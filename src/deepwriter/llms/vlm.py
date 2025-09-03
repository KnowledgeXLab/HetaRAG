from typing import List, Dict, Any
from loguru import logger
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def get_qwen_vl_response(
    query: str,
    context_texts: List[str] = [],
    context_images: List[str] = [],
    context_videos: List[str] = [],
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    max_new_tokens: int = 1024,
) -> str:
    """Generate a response using Qwen2.5-VL model with retrieved context.

    Args:
        query: User query
        context_texts: List of retrieved text passages
        context_images: List of retrieved image paths
        model_name: Name of the model to use
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Generated response
    """
    logger.info(f"Loading Qwen model from {model_name}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # Prepare context
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
    for image_path in context_images:
        messages[0]["content"].append(
            {
                "type": "image",
                "image": image_path,
            }
        )

    # Add videos to the message
    for video_path in context_videos:
        messages[0]["content"].append({"type": "video", "video": video_path})

    # Add the text prompt
    messages[0]["content"].append({"type": "text", "text": prompt})

    # Process input
    logger.info("Processing input for model")
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    # Generate response
    logger.info("Generating response")
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]
