import abc
from typing import List, Optional

import torch
from loguru import logger
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from src.deepwriter.utils.image_utils import load_image


class BaseLLMWrapper(abc.ABC):
    @abc.abstractmethod
    def generate_response(self, query: str) -> str:
        raise NotImplementedError("Subclasses must implement this method.")


class LocalLLMWrapper(BaseLLMWrapper):
    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"Initializing LocalLLMWrapper with model {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(self, query: str):
        pass

    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls(model_name)


class LocalVLMWrapper(BaseLLMWrapper):
    def __init__(
        self, model_name: str, torch_dtype: str = "auto", device: str = "auto"
    ):
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(
        self, query: str, image_paths: Optional[List[str]] = None, **kwargs
    ) -> str:
        if image_paths is None:
            pixel_values = None
        else:
            pixel_values = []
            for image_path in image_paths:
                pixel_values.append(load_image(image_path))
            pixel_values = torch.cat(pixel_values, dim=0)
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            query,
            **kwargs,
        )
        return response

    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls(model_name)
