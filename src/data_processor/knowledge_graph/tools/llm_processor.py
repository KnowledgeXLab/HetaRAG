from collections import deque
import requests
import json
import threading
from openai import OpenAI
import tiktoken

from src.utils.logging_utils import setup_logger
from src.data_processor.knowledge_graph.tools.triple import Triple

logger = setup_logger("llm_processor")


class OllamaInstanceManager:
    def __init__(self, ports, base_url, model, startup_delay=5):
        self.ports = ports
        self.base_url = base_url
        self.instances = []
        self.lock = threading.Lock()
        self.current_instance = 0  # 用于轮询策略
        self.model = model

        for port in self.ports, self.gpus:
            self.instances.append({"port": port, "load": 0})

    def get_available_instance(self):
        """使用轮询策略获取一个可用的实例"""
        with self.lock:
            instance = self.instances[self.current_instance]
            self.current_instance = (self.current_instance + 1) % len(self.instances)
            return instance["port"]  # 返回端口

    def generate_text(self, prompt, temperature=0):
        """发送请求到选择的实例"""
        port = self.get_available_instance()
        base_url = f"{self.base_url}:{port}"

        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "temperature": temperature},
            timeout=240,  # 设置超时时间，避免无限等待
        )
        response.raise_for_status()
        return response


class VllmInstanceManager:
    def __init__(self, ports, base_url, model, startup_delay=5):
        self.ports = ports
        self.base_url = base_url
        self.instances = []
        self.lock = threading.Lock()
        self.current_instance = 0  # 用于轮询策略
        self.model = model

        for port in self.ports:
            self.instances.append({"port": port, "load": 0})

    def get_available_instance(self):
        """使用轮询策略获取一个可用的实例"""
        with self.lock:
            instance = self.instances[self.current_instance]
            self.current_instance = (self.current_instance + 1) % len(self.instances)
            return instance["port"]  # 返回端口

    def generate_text(self, prompt, max_tokens=4096, output_json=False):
        """发送请求到选择的实例"""
        port = self.get_available_instance()
        base_url = f"http://{self.base_url}:{port}/v1"

        try:
            if output_json:
                # 调用 Chat Completion API 并设置参数
                response = requests.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "response_format": {"type": "json_object"},
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=240,
                )

            else:
                response = requests.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=240,
                )

            response.raise_for_status()
            res = json.loads(response.content)
            response_message = res["choices"][0]["message"]["content"]

            return response_message

        except Exception as e:
            logger.info(f"Error: {e}")
            return None


class LLM_Processor:
    def __init__(self, args):
        self.model = args["llm_model"]
        self.base_host = args["llm_host"]
        self.api_key = args["llm_api_key"]
        self.max_error = args["max_error"]
        self.ports = args["llm_ports"]  # 端口池

        if args["llm_framework"].lower() == "ollama":
            print("ollama processor")
            self.manager = OllamaInstanceManager(self.ports, self.base_host, self.model)
            self.generate_text = self.manager.generate_text
        elif args["llm_framework"].lower() == "vllm":
            print("vllm processor")
            self.manager = VllmInstanceManager(self.ports, self.base_host, self.model)
            self.generate_text = self.manager.generate_text
        else:
            self.generate_text = self.default_generate_text

    def default_generate_text(
        self,
        prompt,
        model="qwen-plus",
        temperature=0,
        max_tokens=4096,
        output_json=False,
    ):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            if output_json:
                # 调用 Chat Completion API 并设置参数
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                )

            return completion.model_dump()["choices"][0]["message"]["content"]

        except Exception as e:
            logger.info(f"Error: {e}")
            return None

    def extract_responses(self, response):
        """从流式响应中提取文本."""
        responses = []
        for line in response.iter_lines():
            if line:
                line_data = json.loads(line.decode("utf-8"))
                responses.append(line_data["response"])
        return "".join(responses)

    def extract_triple_prompt(self, corpus: str, entities: list, ref_kg_path):

        if len(entities) > 0:
            for entity in entities:
                example_triple_list = Triple.get_example(entity, ref_kg_path)
                example_triple = (
                    "\n".join(
                        [
                            f"{i+1}. {item.replace(chr(9), ' | ')}"
                            for i, item in enumerate(example_triple_list)
                        ]
                    )
                    + "\n"
                )

            entities_str = ",".join(entities)

            prompt = (
                f'\n[Text]:\n "{corpus}"\n\n'
                f"[Instruction]:\n A triple is composed of a subject, a predicate, and an object. "
                f"Please extract all triples related to [{entities_str}] from the above text as much as possible. \n"
                f"The triples must have one of  [{entities_str}] as the head entity, the text of the tail entity must be short, "
                f"and be output strictly in triple format.\n\n"
                f"[Examples]:\n"
                f"{example_triple}\n"
            )
        else:
            example_triple_list = Triple.get_example("", ref_kg_path)
            example_triple = (
                "\n".join(
                    [
                        f"{i+1}. {item.replace(chr(9), ' | ')}"
                        for i, item in enumerate(example_triple_list)
                    ]
                )
                + "\n"
            )

            prompt = (
                f'\n[Text]:\n "{corpus}"\n\n'
                f"[Instruction]:\n A triple is composed of a subject, a predicate, and an object. "
                f"The subject and object are entities and must be short."
                f"Please extract all triples from the above text as much as possible. \n"
                f"Entities include proper nouns, discipline terminologies, abstract and collective nouns, etc. "
                f"Entities Do Not include any verbs orwords without specific meanings such as time, location, number, measurement, etc. \n"
                f"and be output strictly in triple format.\n\n"
                f"[Examples]:\n"
                f"{example_triple}\n"
            )
        return prompt

    def extract_description_prompt(self, text: str, triple: str):

        # 处理制表符分隔的字符串并转换格式
        parts = triple.split("\t")
        cleaned_parts = [
            part.strip("<").strip(">").strip() for part in parts
        ]  # 去除尖括号和空格
        triple_str = f"subject: {cleaned_parts[0]}, relation: {cleaned_parts[1]}, object: {cleaned_parts[2]}"

        prompt = f"""
        [Text]:
        {text}

        [Triple]:
        {triple_str}

        [Instruction]:
        Each triple consists of a subject, predicate, and object. Based on the triple and the above text fragment (the triple extracted source), extract:

        - The subject and object entities with the following fields:
            - "name"
            - "description": concise summary (≤ 50 English words) capturing key attributes mentioned in the text

        - The relation with:
            - "name"
            - "description": explain the semantic meaning of the relation in context (≤ 50 English words)

        Special requirements:
        1. Prioritize using exact phrases from the text for name fields
        2. Relation description should explain **why** the connection exists based on text evidence

        [Output format]:
        {{
        "subject": {{
            "name": "xxx",
            "description": "xxx"
        }},
        "relation": {{
            "name": "xxx",
            "description": "xxx"
        }},
        "object": {{
            "name": "xxx",
            "description": "xxx"
        }}
        }}
        """

        return prompt

    def call_api(
        self, user_prompt: str, system_prompt: str = "", output_json=False
    ) -> str:
        response = self.generate_text(user_prompt, output_json=output_json)

        if not isinstance(response, str):  ## 如果不是str，则提取response中的内容
            response = self.extract_responses(response).strip()

        return response.replace("_", " ")

    def infer(self, prompt, output_json=False):
        error_count = deque(maxlen=self.max_error)
        cnt = 0
        while sum(error_count) < self.max_error:
            cnt += 1
            try:
                response = self.call_api(user_prompt=prompt, output_json=output_json)
                if not isinstance(response, str):
                    if isinstance(response, dict) and "data" in response:
                        response = response["data"]["output"]
                return response
            except (Exception, KeyboardInterrupt) as e:
                error_count.append(1)  # 记录一次错误
                logger.info(f"LLM request error:{cnt}, {e}")

        if sum(error_count) >= self.max_error:
            logger.info(f"Maximum error tolerance reached, skipping text id: {id}")

    def entity_evaluate(self, entities):
        entities = "\n".join(entities)
        user_prompt = (
            "Analyze the entity list I provide and extract all entities (e.g., proper nouns, discipline terminologies, abstract and collective nouns, etc.). Follow these rules STRICTLY:\n"
            "1. Extract as many nouns from the text as possible\n"
            "2. Do not include any verbs or words without specific meanings such as time, location, number, measurement, etc.\n"
            "3. Output EXACTLY one entity per line\n"
            "4. Maintain ORIGINAL spelling/case from the input\n"
            "5. Ensure that the entities are not repeated\n"
            "Input entity list:\n"
            f"{entities}\n"
            "Now process this list. \n"
        )

        response = self.call_api(user_prompt)
        verify_entities = response.split("\n")
        verify_entities = list(
            set([item.strip() for item in verify_entities if len(item.strip()) > 0])
        )
        return verify_entities


TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0


def truncate_text(text, max_tokens=4096):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text


class InstanceManager:
    def __init__(self, ports, base_url, model, startup_delay=30):

        self.ports = ports
        self.base_url = base_url
        self.instances = []
        self.lock = threading.Lock()
        self.current_instance = 0  # 用于轮询策略
        self.model = model
        self.TOTAL_TOKEN_COST = 0
        self.TOTAL_API_CALL_COST = 0

        for port in self.ports:
            self.instances.append({"port": port, "load": 0})

    def reset_token_cost(self):
        """重置总的token消耗和API调用次数"""
        self.TOTAL_TOKEN_COST = 0
        self.TOTAL_API_CALL_COST = 0

    def get_tokens_cosumption(self):

        return self.TOTAL_TOKEN_COST, self.TOTAL_API_CALL_COST

    def get_available_instance(self):
        """使用轮询策略获取一个可用的实例"""
        with self.lock:
            instance = self.instances[self.current_instance]
            self.current_instance = (self.current_instance + 1) % len(self.instances)
            return instance["port"]  # 返回端口

    def generate_text(self, prompt, system_prompt=None, history_messages=[], **kwargs):
        """发送请求到选择的实例"""
        port = self.get_available_instance()
        base_url = f"http://{self.base_url}:{port}/v1"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Get the cached response if having-------------------

        if len(history_messages) > 1:
            history_messages[0]["content"] = truncate_text(
                history_messages[0]["content"], max_tokens=3000
            )
            history_messages[1]["content"] = truncate_text(
                history_messages[1]["content"], max_tokens=25000
            )
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        try:
            tokenizer = tiktoken.get_encoding("cl100k_base")
            cur_token_cost = len(tokenizer.encode(messages[0]["content"]))
            if cur_token_cost > 31000:
                cur_token_cost = 31000
                messages[0]["content"] = truncate_text(
                    messages[0]["content"], max_tokens=31000
                )

            # logging api call cost
            self.TOTAL_API_CALL_COST += 1
            response = requests.post(
                f"{base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    **kwargs,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=240,
            )
            response.raise_for_status()
            res = json.loads(response.content)
            self.TOTAL_TOKEN_COST += res["usage"]["prompt_tokens"]
            response_message = res["choices"][0]["message"][
                "content"
            ]  # 对结果进行后处理
        except Exception as e:
            print(f"Retry for Error: {e}")
            response = ""
            response_message = ""

        return response_message

    async def generate_text_asy(
        self, prompt, system_prompt=None, history_messages=[], **kwargs
    ):
        """发送请求到选择的实例"""
        port = self.get_available_instance()
        base_url = f"http://{self.base_url}:{port}/v1"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Get the cached response if having-------------------

        if len(history_messages) > 1:
            history_messages[0]["content"] = truncate_text(
                history_messages[0]["content"], max_tokens=3000
            )
            history_messages[1]["content"] = truncate_text(
                history_messages[1]["content"], max_tokens=25000
            )
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        try:
            tokenizer = tiktoken.get_encoding("cl100k_base")
            cur_token_cost = len(tokenizer.encode(messages[0]["content"]))
            if cur_token_cost > 31000:
                cur_token_cost = 31000
                messages[0]["content"] = truncate_text(
                    messages[0]["content"], max_tokens=31000
                )

            # logging api call cost
            self.TOTAL_API_CALL_COST += 1
            response = requests.post(
                f"{base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    **kwargs,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=240,
            )
            response.raise_for_status()
            res = json.loads(response.content)
            self.TOTAL_TOKEN_COST += res["usage"]["prompt_tokens"]
            response_message = res["choices"][0]["message"][
                "content"
            ]  # 对结果进行后处理
        except Exception as e:
            print(f"Retry for Error: {e}")
            response = ""
            response_message = ""

        return response_message


import yaml


def response(
    prompt, system_prompt=None, history_messages=[], port=8001, **kwargs
) -> str:
    with open("src/config/knowledge_graph/create_kg_conf.yaml", "r") as file:
        config = yaml.safe_load(file)
    MODEL = config["llm_conf"]["llm_model"]
    base_url = config["llm_conf"]["llm_host"]
    port = config["llm_conf"]["llm_ports"]

    global TOTAL_TOKEN_COST
    global TOTAL_API_CALL_COST
    URL = f"http://{base_url}:{port[0]}/v1"
    openai_async_client = OpenAI(api_key="vllm", base_url=URL)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    # -----------------------------------------------------
    retry_time = 3
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        # logging token cost
        cur_token_cost = len(tokenizer.encode(messages[0]["content"]))
        if cur_token_cost > 28672:
            cur_token_cost = 28672
            messages[0]["content"] = truncate_text(
                messages[0]["content"], max_tokens=28672
            )
        TOTAL_TOKEN_COST += cur_token_cost
        # logging api call cost
        TOTAL_API_CALL_COST += 1
        # request
        response = openai_async_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            **kwargs,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as e:
        print(f"Retry for Error: {e}")
        retry_time -= 1
        response = ""

    if response == "":
        return response
    return response.choices[0].message.content
