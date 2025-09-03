from src.deepwriter.llms.abstract_wrapper import LocalLLMWrapper


class QwenLMWrapper(LocalLLMWrapper):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def generate_response(self, query: str):
        system_prompt = query.split("\n\n")[0]
        user_prompt = "\n\n".join(query.split("\n\n")[1:])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        return response
