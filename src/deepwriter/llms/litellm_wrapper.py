import litellm

from src.deepwriter.llms.abstract_wrapper import BaseLLMWrapper


class LitellmWrapper(BaseLLMWrapper):
    def __init__(self, model: str, base_url: str = None):
        self.model = model
        self.base_url = base_url

    def generate_response(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        response = litellm.completion(
            model=self.model, messages=messages, base_url=self.base_url
        )
        return response["choices"][0]["message"]["content"]
