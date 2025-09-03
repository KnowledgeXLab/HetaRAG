from src.deepsearch.multi_hop import MultiHopAgent

if __name__ == "__main__":
    # 初始化多跳代理
    agent = MultiHopAgent()

    # 执行多跳推理
    answer = agent.answer(
        query="According to the text, what is the key obstacle to enforcing SCM rules on fishing subsidies?",
        collection_name="world_trade_report",
    )
