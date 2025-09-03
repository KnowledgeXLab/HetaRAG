from typing import List, Dict, Any, Optional


from loguru import logger


from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

from src.deepwriter.llms.abstract_wrapper import LocalVLMWrapper


class QwenVLWrapper(LocalVLMWrapper):

    def __init__(
        self, model_name: str, torch_dtype: str = "auto", device: str = "auto"
    ) -> None:
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def generate_response(
        self,
        query: str,
        context_texts: Optional[List[str]] = None,
        context_images: Optional[List[str]] = None,
        max_new_tokens: int = 1024,
    ) -> str:
        if context_texts is None:
            context = ""
        else:
            context = "\n\n".join([f"Document: {text}" for text in context_texts])

        # Create prompt with context
        prompt = (
            f"You are an AI assistant that answers questions based on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer the question based only on the provided context. "
            f"If the context doesn't contain relevant information, say so."
        )

        # Prepare messages
        messages: List[Dict[str, Any]] = [{"role": "user", "content": []}]

        # Add images to the message
        if context_images:
            for image_path in context_images:
                messages[0]["content"].append(
                    {
                        "type": "image",
                        "image": image_path,
                    }
                )

        # Add the text prompt
        messages[0]["content"].append({"type": "text", "text": prompt})

        # Process input
        logger.info("Processing input for model")
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]
