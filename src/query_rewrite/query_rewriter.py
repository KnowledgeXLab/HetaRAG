"""
Query rewriter implementation for transforming and optimizing queries.
"""

from typing import List, Optional, Dict, Any
import json
import logging
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from src.utils.pdf2chuck.api_requests import APIProcessor


class QueryRewriter:
    """
    A class for rewriting and optimizing queries using LLM.

    This class provides functionality to transform queries into more effective forms
    for better search results and performance.
    """

    def __init__(
        self,
        model: Optional[ChatOllama] = None,
        system_prompt: str = "Generate 3 query variations for improved document retrieval.",
        api_provider: str = "ollama",
    ):
        """
        Initialize the QueryRewriter.

        Args:
            model: Optional ChatOllama instance. If not provided, will create one.
            system_prompt: Instructions for the rewrite task
            api_provider: API provider to use ("ollama", "vllm", etc.)
        """
        self.api_provider = api_provider
        self.system_prompt = system_prompt

        if model is None:
            # 使用现有的API处理器
            self.api_processor = APIProcessor(provider=api_provider)
            self.model = None
        else:
            self.api_processor = None
            self.model = model
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{query}")]
        )
        # 只在使用LangChain时才创建chain
        if self.model is not None:
            self.chain = self._setup_chain()
        else:
            self.chain = None

    def _setup_chain(self) -> RunnableSequence:
        """Set up the processing chain."""
        return self.prompt_template | self.model

    def rewrite(
        self, query: str, max_variations: int = 3, fallback_on_error: bool = True
    ) -> List[str]:
        """
        Generate alternative query formulations.

        Args:
            query: Original user query
            max_variations: Maximum number of variations to return
            fallback_on_error: Whether to return empty list on failure

        Returns:
            List of rewritten queries
        """
        try:
            if self.api_processor is not None:
                # 使用现有的API处理器
                response = self.api_processor.send_message(
                    system_content=self.system_prompt,
                    human_content=query,
                    temperature=0.7,
                )
                parsed = self._parse_response(response, max_variations)
                return parsed
            else:
                # 使用原有的LangChain方式
                if self.chain is not None:
                    response = self.chain.invoke({"query": query})
                    parsed = self._parse_response(response.content, max_variations)
                    return parsed
                else:
                    raise ValueError("No valid processor available")

        except Exception as e:
            logging.warning(f"Query rewrite failed: {str(e)}")
            if fallback_on_error:
                return []
            raise

    def _parse_response(self, response: str, max_variations: int) -> List[str]:
        """
        Parse the LLM response into a list of queries.

        Args:
            response: Raw response from LLM
            max_variations: Maximum number of variations to return

        Returns:
            List of parsed queries
        """
        try:
            data = json.loads(response)
            queries = data.get("queries", [])
            return queries[:max_variations]
        except json.JSONDecodeError:
            return [q.strip() for q in response.split("\n") if q.strip()][
                :max_variations
            ]

    def batch_rewrite(self, queries: list[str]) -> list[str]:
        """
        Rewrite multiple queries in batch.

        Args:
            queries: List of original query strings

        Returns:
            List of rewritten query strings
        """
        return [self.rewrite(query) for query in queries]
