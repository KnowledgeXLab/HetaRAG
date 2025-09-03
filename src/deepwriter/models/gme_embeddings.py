import math
import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor
from loguru import logger

from src.deepwriter.models.base_embedding import BaseEmbeddingModel


def custom_collate_fn(batch):
    """Custom collate function for DataLoader."""
    return batch


class GMEmbeddingModel(BaseEmbeddingModel):
    """Embedding model implementation using GME Qwen2-VL."""

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        min_image_tokens: int = 256,
        max_image_tokens: int = 1280,
        max_length: int = 1800,
        **kwargs,
    ):
        """Initialize the GME embedding model.

        Args:
            model_name: Name of the pretrained model
            model_path: Optional path to a local model
            device: Device to run the model on ('cuda' or 'cpu')
            min_image_tokens: Minimum number of image tokens
            max_image_tokens: Maximum number of image tokens
            max_length: Maximum sequence length
            **kwargs: Additional arguments to pass to the model
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model components
        model_name = model_path or model_name
        logger.info(f"Loading GME model from {model_name}")

        self.base = AutoModelForVision2Seq.from_pretrained(
            model_name, torch_dtype=torch.float16, **kwargs
        )
        self.base.to(self.device)
        self.base.eval()
        self.normalize = True

        min_pixels = min_image_tokens * 28 * 28
        max_pixels = max_image_tokens * 28 * 28
        self.max_length = max_length

        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs
        )
        self.processor.tokenizer.padding_side = "right"
        self.default_instruction = "You are a helpful assistant."
        self.sep = " "

    @property
    def dim(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.base.config.hidden_size

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pooling_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key values for efficient inference
            inputs_embeds: Input embeddings
            pixel_values: Pixel values for images
            image_grid_thw: Image grid dimensions
            pooling_mask: Mask for pooling

        Returns:
            Embedding tensor
        """
        if inputs_embeds is None:
            inputs_embeds = self.base.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.base.visual.get_dtype())
                image_embeds = self.base.visual(
                    pixel_values, grid_thw=image_grid_thw
                ).to(inputs_embeds.device)
                image_mask = input_ids == self.base.config.image_token_id
                inputs_embeds[image_mask] = image_embeds
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.base.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        pooling_mask = attention_mask if pooling_mask is None else pooling_mask
        left_padding = pooling_mask[:, -1].sum() == pooling_mask.shape[0]
        if left_padding:
            embeddings = outputs.last_hidden_state[:, -1]
        else:
            sequence_lengths = pooling_mask.sum(dim=1) - 1
            batch_size = outputs.last_hidden_state.shape[0]
            embeddings = outputs.last_hidden_state[
                torch.arange(batch_size, device=outputs.last_hidden_state.device),
                sequence_lengths,
            ]
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()

    def embed(
        self,
        texts: List[str],
        images: List[Image.Image],
        is_query: bool = True,
        instruction: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Embed text and image pairs.

        Args:
            texts: List of text strings
            images: List of images
            is_query: Whether this is a query embedding
            instruction: Optional instruction for the model

        Returns:
            Tensor of embeddings
        """
        # Inputs must be batched
        input_texts, input_images = list(), list()
        for t, i in zip(texts, images):
            if not is_query or instruction is None:
                instruction = self.default_instruction
            input_str = ""
            if i is None:
                input_images = None  # All examples in the same batch are consistent
            else:
                input_str += "<|vision_start|><|image_pad|><|vision_end|>"
                input_images.append(i)
            if t is not None:
                input_str += t
            msg = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"
            input_texts.append(msg)

        inputs = self.processor(
            text=input_texts,
            images=input_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = self.forward(**inputs)
        return embeddings

    def encode_query(self, query: str, **kwargs) -> torch.Tensor:
        """Encode a query string into an embedding vector.

        Args:
            query: The query text to encode

        Returns:
            Embedding vector for the query
        """
        # use batch to improve efficiency
        embeddings = self.get_fused_embeddings(texts=[query], is_query=True, **kwargs)
        # return the first embedding
        return embeddings[0]

    def encode_multimodal_query(self, query: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Encode a multimodal query into an embedding vector.

        Args:
            query: The query to encode, must contain 'text' and 'image' keys

        Returns:
            Embedding vector for the query
        """
        texts = [query.get("text", "")]
        images = [query.get("image")]
        embeddings = self.get_fused_embeddings(
            texts=texts, images=images, is_query=True, **kwargs
        )
        return embeddings[0]

    def encode_documents(self, documents: List[str], **kwargs) -> torch.Tensor:
        """Encode a list of document texts into embedding vectors.

        Returns:
            Array of embedding vectors for the documents
        """
        embeddings = self.get_fused_embeddings(
            texts=documents, is_query=False, **kwargs
        )
        return embeddings

    def encode_multimodal_documents(
        self, documents: List[Dict[str, Any]], batch_size: int = 32
    ) -> torch.Tensor:
        """Encode a list of multimodal documents into embedding vectors.

        Args:
            documents: List of multimodal documents with 'text' and 'image' keys
                      where 'image' can be a path string or PIL.Image
            batch_size: Batch size for processing

        Returns:
            Array of embedding vectors for the multimodal documents
        """
        texts = [doc.get("text", "") for doc in documents]
        images: List[Optional[Image.Image]] = []

        for doc in documents:
            image = doc.get("image")
            if isinstance(image, Image.Image):
                # Already a PIL Image
                images.append(image)
            elif isinstance(image, (str, Path)):
                # Load image from path
                images.append(Image.open(image))
            else:
                # None or unsupported type
                if image is not None and not isinstance(
                    image, (str, Path, Image.Image)
                ):
                    logger.warning(f"Unsupported image type: {type(image)}")
                images.append(None)

        embeddings = self.get_fused_embeddings(
            texts=texts, images=images, batch_size=batch_size, is_query=False
        )
        return embeddings

    def get_fused_embeddings(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[Union[List[Image.Image], DataLoader]] = None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Get embeddings for text-image pairs.

        Args:
            texts: List of text strings
            images: List of images or DataLoader of images
            batch_size: Batch size for processing
            show_progress_bar: Whether to show progress bar

        Returns:
            Tensor of fused embeddings
        """
        if texts is None and images is None:
            raise ValueError("At least one of texts or images must be provided")

        if isinstance(images, list) or images is None:
            image_loader = DataLoader(
                images or [None] * (len(texts) if texts else 0),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=min(math.floor(os.cpu_count() / 2), 8),
            )
        else:
            image_loader = images

        n_batch = 0
        if texts:
            n_batch = len(texts) // batch_size + int(len(texts) % batch_size > 0)
        elif images:
            n_batch = len(image_loader)

        if texts is None:
            texts = [None] * (len(image_loader) * batch_size)

        all_embeddings = list()
        none_batch = [None] * batch_size

        # Create a simple progress iterator
        iterator = zip(range(0, n_batch * batch_size, batch_size), image_loader)

        for n, img_batch in tqdm(iterator, total=n_batch, desc="Encoding documents"):
            text_batch = none_batch if texts is None else texts[n : n + batch_size]
            img_batch = none_batch if img_batch is None else img_batch
            embeddings = self.embed(texts=text_batch, images=img_batch, **kwargs)
            all_embeddings.append(embeddings.cpu())

        if all_embeddings:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            if texts and len(all_embeddings) > len(texts):
                all_embeddings = all_embeddings[: len(texts)]
            return all_embeddings
        return torch.tensor([])

    def similarity_score(
        self, query_embedding: torch.Tensor, doc_embedding: torch.Tensor
    ) -> float:
        """Calculate cosine similarity between query and document embeddings.

        Args:
            query_embedding: Query embedding vector
            doc_embedding: Document embedding vector

        Returns:
            Cosine similarity score between the embeddings
        """
        # Move to CPU for consistency
        query_embedding = query_embedding.to("cpu")
        doc_embedding = doc_embedding.to("cpu")

        # Normalize vectors
        query_norm = torch.norm(query_embedding)
        doc_norm = torch.norm(doc_embedding)

        if query_norm == 0 or doc_norm == 0:
            return 0.0

        query_embedding = query_embedding / query_norm
        doc_embedding = doc_embedding / doc_norm

        # Calculate cosine similarity using torch operations
        similarity = torch.dot(query_embedding, doc_embedding)

        return float(similarity.item())
