import os
import json
from dotenv import load_dotenv
from typing import Union, List, Dict, Type, Optional, Literal
from openai import OpenAI
import asyncio
from src.utils.pdf2chuck.api_request_parallel_processor import (
    process_api_requests_from_file,
)
from openai.lib._parsing import type_to_response_format_param
import tiktoken
import src.utils.pdf2chuck.prompts as prompts
import requests
from json_repair import repair_json

from src.config.api_config import get_ollama_llm, get_vllm_llm, get_openai_key

ollama_llm_config = get_ollama_llm()
vllm_llm_config = get_vllm_llm()


class BaseOpenaiProcessor:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.default_model = "gpt-4o"

    def set_up_llm(self):
        load_dotenv()
        llm = OpenAI(api_key=get_openai_key(), timeout=None, max_retries=2)

        return llm

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,  # For deterministic ouptputs
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
    ):
        if model is None:
            model = self.default_model
        params = {
            "model": model,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content},
            ],
        }

        # Reasoning models do not support temperature
        if "o3-mini" not in model:
            params["temperature"] = temperature

        if not is_structured:
            completion = self.llm.chat.completions.create(**params)
            content = completion.choices[0].message.content

        elif is_structured:
            params["response_format"] = response_format
            completion = self.llm.beta.chat.completions.parse(**params)

            response = completion.choices[0].message.parsed
            content = response.dict()

        self.response_data = {
            "model": completion.model,
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        }
        print(self.response_data)

        return content

    @staticmethod
    def count_tokens(string, encoding_name="o200k_base"):
        encoding = tiktoken.get_encoding(encoding_name)

        # Encode the string and count the tokens
        tokens = encoding.encode(string)
        token_count = len(tokens)

        return token_count


class BaseIBMAPIProcessor:
    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv("IBM_API_KEY")
        self.base_url = "https://rag.timetoact.at/ibm"
        self.default_model = "meta-llama/llama-3-3-70b-instruct"

    def check_balance(self):
        """Check the current balance for the provided token."""
        balance_url = f"{self.base_url}/balance"
        headers = {"Authorization": f"Bearer {self.api_token}"}

        try:
            response = requests.get(balance_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error checking balance: {err}")
            return None

    def get_available_models(self):
        """Get a list of available foundation models."""
        models_url = f"{self.base_url}/foundation_model_specs"

        try:
            response = requests.get(models_url)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting available models: {err}")
            return None

    def get_embeddings(self, texts, model_id="ibm/granite-embedding-278m-multilingual"):
        """Get vector embeddings for the provided text inputs."""
        embeddings_url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": texts, "model_id": model_id}

        try:
            response = requests.post(embeddings_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting embeddings: {err}")
            return None

    def send_message(
        self,
        # model='meta-llama/llama-3-1-8b-instruct',
        model=None,
        temperature=0.5,
        seed=None,  # For deterministic outputs
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        max_new_tokens=5000,
        min_new_tokens=1,
        **kwargs,
    ):
        if model is None:
            model = self.default_model
        text_generation_url = f"{self.base_url}/text_generation"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        # Prepare the input messages
        input_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content},
        ]

        # Prepare parameters with defaults and any additional parameters
        parameters = {
            "temperature": temperature,
            "random_seed": seed,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            **kwargs,
        }

        payload = {"input": input_messages, "model_id": model, "parameters": parameters}

        try:
            response = requests.post(text_generation_url, headers=headers, json=payload)
            response.raise_for_status()
            completion = response.json()

            content = completion.get("results")[0].get("generated_text")
            self.response_data = {
                "model": completion.get("model_id"),
                "input_tokens": completion.get("results")[0].get("input_token_count"),
                "output_tokens": completion.get("results")[0].get(
                    "generated_token_count"
                ),
            }
            print(self.response_data)
            if is_structured and response_format is not None:
                try:
                    repaired_json = repair_json(content)
                    parsed_dict = json.loads(repaired_json)
                    validated_data = response_format.model_validate(parsed_dict)
                    content = validated_data.model_dump()
                    return content

                except Exception as err:
                    print(
                        "Error processing structured response, attempting to reparse the response..."
                    )
                    reparsed = self._reparse_response(content, system_content)
                    try:
                        repaired_json = repair_json(reparsed)
                        reparsed_dict = json.loads(repaired_json)
                        try:
                            validated_data = response_format.model_validate(
                                reparsed_dict
                            )
                            print("Reparsing successful!")
                            content = validated_data.model_dump()
                            return content

                        except Exception:
                            return reparsed_dict

                    except Exception as reparse_err:
                        print(f"Reparse failed with error: {reparse_err}")
                        print(f"Reparsed response: {reparsed}")
                        return content

            return content

        except requests.HTTPError as err:
            print(f"Error generating text: {err}")
            return None

    def _reparse_response(self, response, system_content):

        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=system_content, response=response
        )

        reparsed_response = self.send_message(
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False,
        )

        return reparsed_response


class BaseOllamaProcessor:
    def __init__(self):
        import ollama

        load_dotenv()
        self.client = ollama.Client(
            host=f"http://{ollama_llm_config['host']}:{ollama_llm_config['port']}"
        )
        self.default_model = "qwen2.5:72b"
        self.response_data = {
            "model": self.default_model,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        **kwargs,
    ):
        if model is None:
            model = self.default_model

        try:
            # Prepare the messages
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content},
            ]

            # Prepare the options
            options = {"temperature": temperature, "seed": seed, **kwargs}

            # Make the API call
            response = self.client.chat(
                model=model,
                messages=messages,
                options=options,
                format=response_format.model_json_schema() if response_format else None,
            )

            content = response["message"]["content"]

            # Set response data
            self.response_data = {
                "model": model,
                "input_tokens": len(system_content)
                + len(human_content),  # Approximate token count
                "output_tokens": len(content),  # Approximate token count
            }
            print("response_data", self.response_data)

            return content

        except Exception as e:
            print(f"Error generating text: {e}")
            # Ensure response_data is set even in case of error
            self.response_data = {
                "model": model,
                "input_tokens": len(system_content) + len(human_content),
                "output_tokens": 0,
                "error": str(e),
            }
            return None


class BaseVllmProcessor:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.default_model = vllm_llm_config["model_name"]

    def set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key="EMPTY",
            base_url=f"http://{vllm_llm_config['host']}:{vllm_llm_config['port']}/v1",
            timeout=None,
            max_retries=2,
        )
        return llm

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,  # For deterministic ouptputs
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
    ):
        if model is None:
            model = self.default_model
        params = {
            "model": model,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content},
            ],
        }

        # Reasoning models do not support temperature
        if "o3-mini" not in model:
            params["temperature"] = temperature

        if not is_structured:
            completion = self.llm.chat.completions.create(**params)
            content = completion.choices[0].message.content

        elif is_structured:
            params["response_format"] = response_format
            completion = self.llm.beta.chat.completions.parse(**params)

            response = completion.choices[0].message.parsed
            content = response.dict()

        self.response_data = {
            "model": completion.model,
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        }
        print(self.response_data)

        return content

    @staticmethod
    def count_tokens(string, encoding_name="o200k_base"):
        encoding = tiktoken.get_encoding(encoding_name)

        # Encode the string and count the tokens
        tokens = encoding.encode(string)
        token_count = len(tokens)

        return token_count


class APIProcessor:
    def __init__(self, provider: Literal["openai", "ibm", "ollama", "vllm"] = "openai"):
        self.provider = provider.lower()
        if self.provider == "openai":
            print("openai processor")
            self.processor = BaseOpenaiProcessor()
        elif self.provider == "ibm":
            self.processor = BaseIBMAPIProcessor()
        elif self.provider == "ollama":
            print("ollama processor")
            self.processor = BaseOllamaProcessor()
        elif self.provider == "vllm":
            self.processor = BaseVllmProcessor()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        **kwargs,
    ):
        """
        Routes the send_message call to the appropriate processor.
        The underlying processor's send_message method is responsible for handling the parameters.
        """
        if model is None:
            model = self.processor.default_model
        return self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            **kwargs,
        )

    def get_answer_from_rag_context(self, question, rag_context, schema, model):
        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(
            schema
        )
        # print("system_prompt", system_prompt)
        # print("user_prompt", user_prompt)
        # print("response_format", response_format)
        answer_dict = self.processor.send_message(
            model=model,
            system_content=system_prompt,
            human_content=user_prompt.format(context=rag_context, question=question),
            is_structured=True,
            response_format=response_format,
        )
        # print("answer_dict", answer_dict)
        # print('--------------------------------')
        # print(type(answer_dict))

        # 如果 answer_dict 是字符串，解析它
        if isinstance(answer_dict, str):
            answer_dict = json.loads(answer_dict)
        # print("answer_dict", answer_dict)
        # print('--------------------------------')
        # print(type(answer_dict))
        self.response_data = self.processor.response_data
        return answer_dict

    def _build_rag_context_prompts(self, schema):
        """Return prompts tuple for the given schema."""
        use_schema_prompt = (
            True if self.provider == "ibm" or self.provider == "gemini" else False
        )

        if schema == "name":
            system_prompt = (
                prompts.AnswerWithRAGContextNamePrompt.system_prompt_with_schema
                if use_schema_prompt
                else prompts.AnswerWithRAGContextNamePrompt.system_prompt
            )
            response_format = prompts.AnswerWithRAGContextNamePrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamePrompt.user_prompt
        elif schema == "number":
            system_prompt = (
                prompts.AnswerWithRAGContextNumberPrompt.system_prompt_with_schema
                if use_schema_prompt
                else prompts.AnswerWithRAGContextNumberPrompt.system_prompt
            )
            response_format = prompts.AnswerWithRAGContextNumberPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNumberPrompt.user_prompt
        elif schema == "boolean":
            system_prompt = (
                prompts.AnswerWithRAGContextBooleanPrompt.system_prompt_with_schema
                if use_schema_prompt
                else prompts.AnswerWithRAGContextBooleanPrompt.system_prompt
            )
            response_format = prompts.AnswerWithRAGContextBooleanPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextBooleanPrompt.user_prompt
        elif schema == "names":
            system_prompt = (
                prompts.AnswerWithRAGContextNamesPrompt.system_prompt_with_schema
                if use_schema_prompt
                else prompts.AnswerWithRAGContextNamesPrompt.system_prompt
            )
            response_format = prompts.AnswerWithRAGContextNamesPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamesPrompt.user_prompt
        elif schema == "comparative":
            system_prompt = (
                prompts.ComparativeAnswerPrompt.system_prompt_with_schema
                if use_schema_prompt
                else prompts.ComparativeAnswerPrompt.system_prompt
            )
            response_format = prompts.ComparativeAnswerPrompt.AnswerSchema
            user_prompt = prompts.ComparativeAnswerPrompt.user_prompt
        else:
            raise ValueError(f"Unsupported schema: {schema}")
        return system_prompt, response_format, user_prompt

    def get_rephrased_questions(
        self, model, original_question: str, companies: List[str]
    ) -> Dict[str, str]:
        """
        Use LLM to break down a comparative question into individual questions.
        Returns a dictionary: {company_name: question}
        """
        # 调用模型
        raw_response = self.processor.send_message(
            model=model,
            system_content=prompts.RephrasedQuestionsPrompt.system_prompt,
            human_content=prompts.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question,
                companies=", ".join([f'"{company}"' for company in companies]),
            ),
            is_structured=True,
            response_format=prompts.RephrasedQuestionsPrompt.RephrasedQuestions,
        )

        print("RAW answer_dict (still a string):")
        print(raw_response)
        print("--------------------------------")

        # 如果返回的是字符串，尝试解析为 dict
        if isinstance(raw_response, str):
            try:
                answer_dict = json.loads(raw_response)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Cannot parse response as JSON:\n{raw_response}"
                ) from e
        elif isinstance(raw_response, dict):
            answer_dict = raw_response
        else:
            raise TypeError(f"Unexpected response type: {type(raw_response)}")

        # 提取 questions 字段
        questions_raw = answer_dict.get("questions")
        if isinstance(questions_raw, str):
            try:
                questions_raw = json.loads(questions_raw)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Cannot parse 'questions' as JSON:\n{questions_raw}"
                ) from e

        # 验证结构
        if not isinstance(questions_raw, list):
            raise TypeError(f"'questions' must be a list, got {type(questions_raw)}")

        # 构建最终字典
        questions_dict = {
            item["company_name"]: item["question"]
            for item in questions_raw
            if isinstance(item, dict) and "company_name" in item and "question" in item
        }

        return questions_dict


class AsyncOpenaiProcessor:

    def _get_unique_filepath(self, base_filepath):
        """Helper method to get unique filepath"""
        if not os.path.exists(base_filepath):
            return base_filepath

        base, ext = os.path.splitext(base_filepath)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        return f"{base}_{counter}{ext}"

    async def process_structured_ouputs_requests(
        self,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        queries=None,
        response_format=None,
        requests_filepath="./temp_async_llm_requests.jsonl",
        save_filepath="./temp_async_llm_results.jsonl",
        preserve_requests=False,
        preserve_results=True,
        request_url="https://api.openai.com/v1/chat/completions",
        max_requests_per_minute=3_500,
        max_tokens_per_minute=3_500_000,
        token_encoding_name="o200k_base",
        max_attempts=5,
        logging_level=20,
        progress_callback=None,
    ):
        # Create requests for jsonl
        jsonl_requests = []
        for idx, query in enumerate(queries):
            request = {
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query},
                ],
                "response_format": type_to_response_format_param(response_format),
                "metadata": {"original_index": idx},
            }
            jsonl_requests.append(request)

        # Get unique filepaths if files already exist
        requests_filepath = self._get_unique_filepath(requests_filepath)
        save_filepath = self._get_unique_filepath(save_filepath)

        # Write requests to JSONL file
        with open(requests_filepath, "w") as f:
            for request in jsonl_requests:
                json_string = json.dumps(request)
                f.write(json_string + "\n")

        # Process API requests
        total_requests = len(jsonl_requests)

        async def monitor_progress():
            last_count = 0
            while True:
                try:
                    with open(save_filepath, "r") as f:
                        current_count = sum(1 for _ in f)
                        if current_count > last_count:
                            if progress_callback:
                                for _ in range(current_count - last_count):
                                    progress_callback()
                            last_count = current_count
                        if current_count >= total_requests:
                            break
                except FileNotFoundError:
                    pass
                await asyncio.sleep(0.1)

        async def process_with_progress():
            await asyncio.gather(
                process_api_requests_from_file(
                    requests_filepath=requests_filepath,
                    save_filepath=save_filepath,
                    request_url=request_url,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    max_requests_per_minute=max_requests_per_minute,
                    max_tokens_per_minute=max_tokens_per_minute,
                    token_encoding_name=token_encoding_name,
                    max_attempts=max_attempts,
                    logging_level=logging_level,
                ),
                monitor_progress(),
            )

        await process_with_progress()

        with open(save_filepath, "r") as f:
            validated_data_list = []
            results = []
            for line_number, line in enumerate(f, start=1):
                raw_line = line.strip()
                try:
                    result = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    print(
                        f"[ERROR] Line {line_number}: Failed to load JSON from line: {raw_line}"
                    )
                    continue

                # Check finish_reason in the API response
                finish_reason = result[1]["choices"][0].get("finish_reason", "")
                if finish_reason != "stop":
                    print(
                        f"[WARNING] Line {line_number}: finish_reason is '{finish_reason}' (expected 'stop')."
                    )

                # Safely parse answer; if it fails, leave answer empty and report the error.
                try:
                    answer_content = result[1]["choices"][0]["message"]["content"]
                    answer_parsed = json.loads(answer_content)
                    answer = response_format(**answer_parsed).model_dump()
                except Exception as e:
                    print(
                        f"[ERROR] Line {line_number}: Failed to parse answer JSON. Error: {e}."
                    )
                    answer = ""

                results.append(
                    {
                        "index": result[2],
                        "question": result[0]["messages"],
                        "answer": answer,
                    }
                )

            # Sort by original index and build final list
            validated_data_list = [
                {"question": r["question"], "answer": r["answer"]}
                for r in sorted(results, key=lambda x: x["index"]["original_index"])
            ]

        if not preserve_requests:
            os.remove(requests_filepath)

        if not preserve_results:
            os.remove(save_filepath)
        else:  # Fix requests order
            with open(save_filepath, "r") as f:
                results = [json.loads(line) for line in f]

            sorted_results = sorted(results, key=lambda x: x[2]["original_index"])

            with open(save_filepath, "w") as f:
                for result in sorted_results:
                    json_string = json.dumps(result)
                    f.write(json_string + "\n")

        return validated_data_list
