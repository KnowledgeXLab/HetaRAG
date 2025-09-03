import os
import json
from dotenv import load_dotenv
from typing import Union, List, Dict, Type, Optional, Literal
from openai import OpenAI

from src.config.api_config import (
    get_embedding_framework,
    get_llm_framework,
    get_ollama_embedding,
    get_ollama_llm,
    get_vllm_embedding,
    get_vllm_llm,
)

embedding_framework = get_embedding_framework()
llm_framework = get_llm_framework()


class VllmProcessor:
    def __init__(self):
        self.llm = None
        self.embedding_config = get_vllm_embedding()
        self.llm_config = get_vllm_llm()

    def set_up_llm(self):
        llm = OpenAI(
            api_key="your_api_key",  # vLLM 服务不需要 API 密钥，可以使用任意字符串
            base_url=f"http://{self.llm_config['host']}:{self.llm_config['port']}/v1",
            timeout=None,
            max_retries=3,
        )
        return llm

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        **kwargs,
    ):
        self.llm = self.set_up_llm()
        if model is None:
            model = self.llm_config["model_name"]

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content},
        ]
        params = {
            "model": model,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content},
            ],
            "temperature": temperature,
        }
        params.update({k: v for k, v in kwargs.items() if v is not None})

        response = self.llm.chat.completions.create(**params)
        content = response.choices[0].message.content

        return content

    def set_up_embedding(self):
        client = OpenAI(
            api_key="your_api_key",  # vLLM 服务不需要 API 密钥，可以使用任意字符串
            base_url=f"http://{self.embedding_config['host']}:{self.embedding_config['port']}/v1",
        )
        return client

    def get_embedding(self, model=None, prompt=""):
        self.client = self.set_up_embedding()
        if model is None:
            model = self.embedding_config["model_name"]

        response = self.client.embeddings.create(model=model, input=[prompt])
        embedding = response.data[0].embedding

        return embedding

    def get_list_embedding(self, model=None, text_list=[]):
        self.client = self.set_up_embedding()
        if model is None:
            model = self.embedding_config["model_name"]

        embedding = self.client.embeddings.create(
            input=text_list,
            model=model,
        )
        final_embedding = [d.embedding for d in embedding.data]

        return final_embedding

    def get_llm_model(self):
        return self.llm_config["model_name"]

    def get_embedding_model(self):
        return self.embedding_config["model_name"]


class OllamaProcessor:
    def __init__(self):
        self.llm = None
        self.embedding_config = get_ollama_embedding()
        self.llm_config = get_ollama_llm()

    def set_up_llm(self):
        import ollama

        llm = ollama.Client(
            host=f"http://{self.llm_config['host']}:{self.llm_config['port']}"
        )
        return llm

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        **kwargs,
    ):
        self.llm = self.set_up_llm()
        if model is None:
            model = self.llm_config["model_name"]

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content},
        ]
        # Prepare the options
        options = {"temperature": temperature, "seed": seed, **kwargs}
        response = self.llm.chat(model=model, messages=messages, options=options)
        content = response["message"]["content"]

        return content

    def set_up_embedding(self):
        import ollama

        client = ollama.Client(
            host=f"http://{self.embedding_config['host']}:{self.embedding_config['port']}"
        )
        return client

    def get_embedding(self, model=None, prompt=""):
        if model is None:
            model = self.embedding_config["model_name"]
        self.client = self.set_up_embedding()
        embedding = self.client.embeddings(model="bge-m3", prompt=prompt)["embedding"]

        return embedding

    def get_list_embedding(self, model=None, text_list=[]):
        self.client = self.set_up_embedding()
        if model is None:
            model = self.embedding_config["model_name"]

        final_embedding = []
        for t in text_list:
            embedding = self.client.embeddings(model="bge-m3", prompt=t)["embedding"]
            final_embedding.append(embedding)

        return final_embedding

    def get_llm_model(self):
        return self.llm_config["model_name"]

    def get_embedding_model(self):
        return self.embedding_config["model_name"]


class EmbeddingProcessor:
    def __init__(self):
        self.provider = embedding_framework.lower()
        if self.provider == "vllm":
            # print("vllm processor")
            self.processor = VllmProcessor()
        elif self.provider == "ollama":
            # print("ollama processor")
            self.processor = OllamaProcessor()
        else:
            raise ValueError(f"Unsupported embeddingprovider: {embedding_framework}")

    def get_embedding(self, model=None, prompt=""):
        return self.processor.get_embedding(model=model, prompt=prompt)

    def get_list_embedding(self, model=None, text_list=[""]):
        return self.processor.get_list_embedding(model=model, text_list=text_list)


class LLMProcessor:
    def __init__(self):
        self.provider = embedding_framework.lower()
        if self.provider == "vllm":
            # print("vllm processor")
            self.processor = VllmProcessor()
        elif self.provider == "ollama":
            # print("ollama processor")
            self.processor = OllamaProcessor()
        else:
            raise ValueError(f"Unsupported embeddingprovider: {embedding_framework}")

    def get_response(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        **kwargs,
    ):

        return self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            **kwargs,
        )

    def get_llm_model(self):
        return self.processor.get_llm_model()

    def get_embedding_model(self):
        return self.processor.get_embedding_model()
