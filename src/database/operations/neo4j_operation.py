import csv
from py2neo import Graph, Node, Relationship
from neo4j import GraphDatabase
import re, jieba, spacy
import time
from src.database.db_connection import neo4j_connection, neo4j_connection_driver


# 这里是判断一句话是属于什么类型的话
def detect_language(text):
    # 使用正则表达式来匹配中文字符
    chinese_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")

    # 使用正则表达式来匹配英文单词（由字母组成的字符串）
    english_words = re.findall(r"\b[a-zA-Z]+\b", text)

    # 计算英文单词的数量
    english_count = len(english_words)

    if chinese_count >= english_count:
        return "中文"
    elif english_count > chinese_count:
        return "英文"


# 这里是将输入的语言进行分词，分词结果以list输出
def divide(question, user_dict):

    # 首先判断是什么模式，中文模式，使用jieba：
    if detect_language(question) == "中文":
        # 添加自定义词库
        jieba.load_userdict(user_dict)
        # 使用全模式进行切分,可以更全，但是容易出现歧义
        seg_list = jieba.lcut(question, cut_all=True)
        # print("jieba切词模型切完之后如下:",seg_list)

        return seg_list

    # 英文模式使用spacy,需要提前下载模型：
    # 执行该命令自动下载 python -m spacy download en_core_web_sm
    # 详见https://blog.csdn.net/qq_51624702/article/details/132722073
    else:
        nlp = spacy.load("en_core_web_sm")

        # 读取词汇文件并将词汇添加到 SpaCy 的 Vocab
        with open(user_dict, "r", encoding="utf-8") as file:
            for word in file:
                word = word.strip()  # 去除每行的空白字符
                if word:  # 跳过空行
                    lexeme = nlp.vocab[word]
                    lexeme.is_stop = False  # 可以设置是否为停用词，默认是不作为停用词
                    lexeme.is_alpha = True  # 你可以指定它是一个合法的单词（字母组成）

        doc = nlp(question)
        seg_list = [token.text for token in doc]
        # print("spacy切词模型切完之后如下:",seg_list)
        return seg_list


# 这个代码主要用来获取neo4j的所有节点，并以string的形式返回，供后续下游任务的筛选
def get_neo4j_node(driver):
    with driver.session() as session:
        # 获取所有节点的名字（这里假设所有节点都是用name作为key的）
        node_query = "MATCH (n) RETURN DISTINCT n.name AS node_name"
        node_result = session.run(node_query)
        nodes = []
        for record in node_result:
            nodes.append(record["node_name"])
    # 以列表形式返回
    return nodes


# 这个代码主要用来获取neo4j的所有节点，并以string的形式返回，供后续下游任务的筛选
def get_neo4j_relation(driver):
    with driver.session() as session:
        # 获取所有节点的名字（这里假设所有节点都是用name作为key的）
        node_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType as relation"
        node_result = session.run(node_query)
        relations = []
        for record in node_result:
            relations.append(record["relation"])
    # 以列表形式返回
    return relations


class neo4j_key_search_bytoken_with_more_relation:
    def __init__(self, driver, question, top_k, user_dict):
        self.driver = driver  # 获取与neo4j的链接驱动
        self.question = question  # 获取问题
        self.top_k = top_k  # 获取top_k
        self.neo4j_nodes = get_neo4j_node(
            driver
        )  # 获取数据库中的所有节点,列表形式类型 #也即知识库的内部实体表
        self.relation_dict = {}  # 用于存放查询到的relation及其评分
        self.user_dict = user_dict
        self.result_dict = {}

    # 用于过滤掉分词器中的，无关的词语
    def node_filter(self):
        seg_list = divide(self.question, self.user_dict)  # 获取切分后的词语
        key_nodes = []
        for node in seg_list:
            if node in self.neo4j_nodes:
                key_nodes.append(node)

        # print(key_nodes)
        return key_nodes

    # 这个代码是查找两node节点的关键路径，并对路径上的关系进行评分存储
    def search_relation(self, node1, node2):
        # 查询语句,同时考虑出度和入度，并放在同等地位
        relation_query = f"MATCH p = (a)-[*1..4]-(b)\
                            WHERE a.name = '{node1}' AND b.name = '{node2}'\
                            UNWIND relationships(p) AS r\
                            RETURN\
                                startNode(r).name AS node1,\
                                endNode(r).name AS node2,\
                                type(r) AS relation,\
                                    properties(r) AS properties"

        with self.driver.session() as session:
            b = time.time()
            query_result = list(session.run(relation_query))
            e = time.time()
            # print(e-b)
            # 首先处理开头和结尾,每个额外+5分:#先进行分数累计，重要的节点会得到更多分数
            # 开头
            b = time.time()
            if not query_result:
                print(f"未找到从 {node1} 到 {node2} 的路径。")
                return
            item = f"node1:{query_result[0]['node1']},node2:{query_result[0]['node2']},relation:{query_result[0]['relation']},text:{query_result[0]['properties']['text']}."
            if item in self.relation_dict:
                self.relation_dict[item] += 5
            else:
                self.relation_dict[item] = 5
            # 结尾
            item = f"node1:{query_result[-1]['node1']},node2:{query_result[-1]['node2']},relation:{query_result[-1]['relation']},text:{query_result[-1]['properties']['text']}."
            if item in self.relation_dict:
                self.relation_dict[item] += 5
            else:
                self.relation_dict[item] = 5
            # 整条路径+15分
            for record in query_result:
                item = f"node1:{record['node1']},node2:{record['node2']},relation:{record['relation']},text:{record['properties']['text']}."
                if item in self.relation_dict:
                    self.relation_dict[item] += 15
                else:
                    self.relation_dict[item] = 15
            e = time.time()
            # print(e-b)

            b = time.time()
            flag = 0  # 记录起始和结束
            temp_value = 0  # 记录每条路径的值
            result = ""
            for record in query_result:
                index = f"node1:{record['node1']},node2:{record['node2']},relation:{record['relation']},text:{record['properties']['text']}."

                temp_value += self.relation_dict[index] - 5

                result += index + "--"
                if (
                    record["node1"] == node1
                    or record["node1"] == node2
                    or record["node2"] == node1
                    or record["node2"] == node2
                ):
                    flag += 1

                if flag == 2:  # 找到了一条链
                    self.result_dict[result] = temp_value
                    flag = 0
                    temp_value = 0
                    result = ""
            e = time.time()
            # print(e-b)

        return self.result_dict

    # 当只有一个节点的时候，调用这个代码，逻辑是返回所有的一度边
    def search_relation_only1node(self, node):
        relation_query = f"MATCH (n)-[r_1]-(m) WHERE n.name = '{node}' RETURN n.name AS node1,TYPE(r_1) AS relation, m.name AS node2,properties(r_1) AS properties"
        with self.driver.session() as session:
            query_result = session.run(relation_query)
            for record in query_result:
                item = f"node1:{record['node1']},node2:{record['node2']},relation:{record['relation']},text:{record['properties']['text']}"
                if item in self.relation_dict:
                    self.relation_dict[item] += 20
                else:
                    self.relation_dict[item] = 20

    ##################
    # keysearch的总流程，根据用户输入的问题，提取相关主体，而后根据主体，查找节点的一度、二度节点
    # 最后返回评分最高的top_k关系，文档id绑定在关系上，返回文档id即可查找文档
    ##################
    def pipeline(self):
        # 首先将字典归0
        self.relation_dict = {}
        # 过滤关键节点，并按照条件调用查询
        key_nodes = self.node_filter()
        node_num = len(key_nodes)
        if node_num < 2:
            for node in key_nodes:
                self.search_relation_only1node(node)
        else:  # 如果数量多的话，两两去搜索关键路径，并存储分数
            for i in range(node_num - 1):
                for j in range(i + 1, node_num):
                    # print("查询node1:",key_nodes[i],"node2",key_nodes[j])
                    self.search_relation(key_nodes[i], key_nodes[j])

        # #返回top_k,首先排序，按照评分排序：返回的是列表类型，列表中是键值对
        self.sorted_relation_dict = sorted(
            self.result_dict.items(), key=lambda item: item[1], reverse=True
        )

        # 当返回数据少于top_k，直接将所有返回即可
        return self.sorted_relation_dict[: self.top_k]


def import_csv_to_neo4j(csv_file_path, neo4j_graph):
    """
    从 CSV 文件导入数据到 Neo4j 图数据库

    :param csv_file_path: CSV 文件路径
    :param neo4j_graph: Neo4j 图数据库连接对象
    """
    try:
        with open(csv_file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for item in reader:
                if reader.line_num == 1:
                    continue  # 跳过表头

                # 检查字段是否为空
                if not item[0] or not item[1] or not item[2] or not item[4]:
                    print(f"跳过空数据行: {item}")
                    continue  # 跳过当前行

                # 创建节点
                start_node = Node("node", name=item[0])
                end_node = Node("node", name=item[1])

                # 创建关系
                relation = Relationship(start_node, item[2], end_node, chunk_id=item[4])

                # 使用 merge 确保节点存在
                neo4j_graph.merge(start_node, "node", "name")
                neo4j_graph.merge(end_node, "node", "name")

                # 创建关系
                neo4j_graph.create(relation)

        print("数据导入完成")
    except FileNotFoundError:
        print(f"文件未找到: {csv_file_path}")
    except Exception as e:
        print(f"导入数据时出错: {e}")


def delete(neo4j_graph):
    # 执行 Cypher 查询删除所有节点和关系
    neo4j_graph.run("MATCH (n) DETACH DELETE n")


# 这里是判断一句话是属于什么类型的话
def detect_language(text):
    # 使用正则表达式来匹配中文字符
    chinese_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")

    # 使用正则表达式来匹配英文单词（由字母组成的字符串）
    english_words = re.findall(r"\b[a-zA-Z]+\b", text)

    # 计算英文单词的数量
    english_count = len(english_words)

    if chinese_count >= english_count:
        return "中文"
    elif english_count > chinese_count:
        return "英文"


# 这里是将输入的语言进行分词，分词结果以list输出
def divide(question, user_dict):

    # 首先判断是什么模式，中文模式，使用jieba：
    if detect_language(question) == "中文":
        # 添加自定义词库
        jieba.load_userdict(user_dict)
        # 使用全模式进行切分,可以更全，但是容易出现歧义
        seg_list = jieba.lcut(question, cut_all=True)
        # print("jieba切词模型切完之后如下:",seg_list)

        return seg_list

    # 英文模式使用spacy,需要提前下载模型：
    # 详见https://blog.csdn.net/qq_51624702/article/details/132722073
    else:
        nlp = spacy.load("en_core_web_sm")

        # 读取词汇文件并将词汇添加到 SpaCy 的 Vocab
        with open(user_dict, "r", encoding="utf-8") as file:
            for word in file:
                word = word.strip()  # 去除每行的空白字符
                if word:  # 跳过空行
                    lexeme = nlp.vocab[word]
                    lexeme.is_stop = False  # 可以设置是否为停用词，默认是不作为停用词
                    lexeme.is_alpha = True  # 你可以指定它是一个合法的单词（字母组成）

        doc = nlp(question)
        seg_list = [token.text for token in doc]
        # print("spacy切词模型切完之后如下:",seg_list)
        return seg_list


# 这个代码主要用来获取neo4j的所有节点，并以string的形式返回，供后续下游任务的筛选
def get_neo4j_node(driver):
    with driver.session() as session:
        # 获取所有节点的名字（这里假设所有节点都是用name作为key的）
        node_query = "MATCH (n) RETURN DISTINCT n.name AS node_name"
        node_result = session.run(node_query)
        nodes = []
        for record in node_result:
            nodes.append(record["node_name"])
    # 以列表形式返回
    return nodes


class Key_search_bytoken:
    def __init__(self, driver, question, top_k, user_dict):
        self.driver = driver  # 获取与neo4j的链接驱动
        self.question = question  # 获取问题
        self.top_k = top_k  # 获取top_k
        self.neo4j_nodes = get_neo4j_node(driver)  # 获取数据库中的所有节点,列表形式类型
        self.relation_dict = {}  # 用于存放查询到的relation及其评分
        self.user_dict = user_dict

    # 用于过滤掉分词器中的，无关的词语
    def node_filter(self):
        seg_list = divide(self.question, self.user_dict)  # 获取切分后的词语
        key_nodes = []
        for node in seg_list:
            if node in self.neo4j_nodes:
                key_nodes.append(node)

        # print(key_nodes)
        return key_nodes

    # 这个代码是查找两node节点的关键路径，并对路径上的关系进行评分存储
    def search_relation(self, node1, node2):
        # 查询语句,同时考虑出度和入度，并放在同等地位
        relation_query = f"MATCH p = shortestPath((a)-[*1..5]-(b))\
                            WHERE a.name = '{node1}' AND b.name = '{node2}'\
                            UNWIND relationships(p) AS r\
                            RETURN\
                                startNode(r).name AS node1,\
                                endNode(r).name AS node2,\
                                type(r) AS relation,\
                                r.chunk_id AS chunk_id"

        with self.driver.session() as session:
            query_result = list(session.run(relation_query))
            # 首先处理开头和结尾,每个额外+5分:
            # 开头
            item = f"node1:{query_result[0]['node1']},node2:{query_result[0]['node2']},relation:{query_result[0]['relation']},chunk_id:{query_result[0]['chunk_id']}"
            if item in self.relation_dict:
                self.relation_dict[item] += 5
            else:
                self.relation_dict[item] = 5
            # 结尾
            item = f"node1:{query_result[-1]['node1']},node2:{query_result[-1]['node2']},relation:{query_result[-1]['relation']},chunk_id:{query_result[-1]['chunk_id']}"
            if item in self.relation_dict:
                self.relation_dict[item] += 5
            else:
                self.relation_dict[item] = 5
            # 整条路径+15分
            for record in query_result:
                item = f"node1:{record['node1']},node2:{record['node2']},relation:{record['relation']},chunk_id:{record['chunk_id']}"
                if item in self.relation_dict:
                    self.relation_dict[item] += 15
                else:
                    self.relation_dict[item] = 15

        return self.relation_dict

    # 当只有一个节点的时候，调用这个代码，逻辑是返回所有的一度边
    def search_relation_only1node(self, node):
        relation_query = f"MATCH (n)-[r_1]-(m) WHERE n.name = '{node}' RETURN n.name AS node1,TYPE(r_1) AS relation, m.name AS node2,r_1.chunk_id AS chunk_id"
        with self.driver.session() as session:
            query_result = session.run(relation_query)
            for record in query_result:
                item = f"node1:{record['node1']},node2:{record['node2']},relation:{record['relation']},chunk_id:{record['chunk_id']}"
                if item in self.relation_dict:
                    self.relation_dict[item] += 20
                else:
                    self.relation_dict[item] = 20

    ##################
    # keysearch的总流程，根据用户输入的问题，提取相关主体，而后根据主体，查找节点的一度、二度节点
    # 最后返回评分最高的top_k关系，文档id绑定在关系上，返回文档id即可查找文档
    ##################
    def pipeline(self):
        # 首先将字典归0
        self.relation_dict = {}
        # 过滤关键节点，并按照条件调用查询
        key_nodes = self.node_filter()
        node_num = len(key_nodes)
        if node_num < 2:
            for node in key_nodes:
                self.search_relation_only1node(node)
        else:  # 如果数量多的话，两两去搜索关键路径，并存储分数
            for i in range(node_num - 1):
                for j in range(i + 1, node_num):
                    # print("查询node1:",key_nodes[i],"node2",key_nodes[j])
                    self.search_relation(key_nodes[i], key_nodes[j])

        # #返回top_k,首先排序，按照评分排序：返回的是列表类型，列表中是键值对
        self.sorted_relation_dict = sorted(
            self.relation_dict.items(), key=lambda item: item[1], reverse=True
        )

        # 当返回数据少于top_k，直接将所有返回即可
        return self.sorted_relation_dict[: self.top_k]


if __name__ == "__main__":
    # 用户的外部附加字典
    user_dict = "src/resources/temp/database/all_data.csv"

    driver = neo4j_connection_driver()
    key_searcher = Key_search_bytoken(
        driver, "在银屑病治疗过程中，糖皮质激素的作用是什么？", 5, user_dict
    )
    out = key_searcher.pipeline()
    print(out)

    driver.close()
