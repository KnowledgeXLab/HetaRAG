"""
Tests for the query rewrite module.
"""

from src.query_rewrite import QueryRewriter, QUERY_REWRITE_PROMPT
from langchain_community.chat_models import ChatOllama

if __name__ == "__main__":
    # 创建重写器实例
    rewriter = QueryRewriter(system_prompt=QUERY_REWRITE_PROMPT)

    # 测试查询重写
    test_queries = [
        "What are the health benefits of green tea?",
        "How does artificial intelligence work?",
        "What is the capital of France?",
    ]

    print("\n=== 查询重写测试 ===")
    for query in test_queries:
        print(f"\n原始查询: {query}")
        variations = rewriter.rewrite(query)
        print("重写变体:")
        for i, var in enumerate(variations, 1):
            print(f"{i}. {var}")
