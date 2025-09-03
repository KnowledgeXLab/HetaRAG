import pymysql
from pymysql import Error
from src.database.db_connection import mysql_connnection


# 创建数据库
def create_database(connection, db_name):
    try:
        with connection.cursor() as cursor:
            # 尝试创建数据库
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"数据库 {db_name} 创建成功")
            return connection
    except pymysql.Error as e:
        # 错误代码 1007 表示数据库已存在
        if e.args[0] == 1007:  # ER_DB_CREATE_EXISTS
            print(f"数据库 {db_name} 已存在")
            return connection
        else:
            print(f"创建数据库时出错: {e}")
            return connection


# 创建表格
def create_table(connection, table_name):
    with connection.cursor() as cursor:
        # 检查表是否已存在
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        result = cursor.fetchone()

        if result:
            print(f"表 {table_name} 已存在，无需创建")
            return

        # 创建表SQL
        create_table_sql = f"""
        CREATE TABLE {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            block_id BIGINT NOT NULL,
            pdf_path VARCHAR(512),
            num_pages INT,
            page_number INT,
            page_height FLOAT,
            page_width FLOAT,
            num_blocks INT,
            block_type VARCHAR(50),
            block_content TEXT,
            block_summary TEXT,
            block_embedding TEXT,  -- 存储向量数据
            image_path VARCHAR(512),
            image_caption TEXT,
            image_footer TEXT,
            block_bbox JSON,  -- 存储边界框坐标
            document_title VARCHAR(512),
            section_title VARCHAR(512)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """

        # 执行创建表
        cursor.execute(create_table_sql)
        connection.commit()
        print(f"成功创建表 {table_name}")


# 插入
from typing import List, Dict, Any
import pickle
import json, csv
import numpy as np


def load_pkl_data(pkl_path: str) -> List[Dict[str, Any]]:
    """
    从PKL文件加载数据
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def prepare_data_for_mysql(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    准备数据以适应MySQL插入
    """
    processed = {}
    for key, value in data.items():
        if value is None:
            processed[key] = None
        elif isinstance(value, np.ndarray):
            processed[key] = value.tolist()
        elif isinstance(value, (list, dict, tuple)):
            processed[key] = json.dumps(value)  # 使用json转换复杂结构
        elif hasattr(value, "value"):  # 枚举类型
            processed[key] = str(value.value)
        else:
            processed[key] = value
    return processed


def convert_value(value: Any) -> Any:
    """转换特殊数据类型为CSV可接受的格式"""
    if isinstance(value, (list, dict, np.ndarray)):
        return str(value)  # 将列表/字典转为字符串
    elif hasattr(value, "value"):  # 处理枚举类型
        return str(value.value)
    return value


def export_to_csv(data_list: List[Dict[str, Any]], csv_path: str):
    """将字典列表导出到CSV文件"""
    if not data_list:
        print("没有数据可导出")
        return

    # 获取所有可能的字段名(合并所有字典的key)
    fieldnames = set()
    for item in data_list:
        fieldnames.update(item.keys())
    fieldnames = sorted(fieldnames)

    # 写入CSV文件
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in data_list:
            # 转换每个值
            row = {key: convert_value(item.get(key)) for key in fieldnames}
            writer.writerow(row)

    print(f"成功导出数据到 {csv_path}")
    print(f"总记录数: {len(data_list)}")
    print(f"包含字段: {', '.join(fieldnames)}")


def import_pkl_to_mysql(
    connection, table_name: str, pkl_path: str, batch_size: int = 100
):
    """
    将PKL文件数据导入MySQL
    """
    # 加载PKL数据
    print(f"正在加载PKL文件: {pkl_path}")
    all_data = load_pkl_data(pkl_path)
    if not all_data:
        print("PKL文件中没有数据")
        return
    csv_file = "src/resources/temp/database/all_data.csv"
    export_to_csv(all_data, csv_file)
    try:
        # 连接MySQL数据库

        with connection.cursor() as cursor:
            # 读取CSV文件
            with open(csv_file, "r", encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader)  # 跳过标题行

                # 准备SQL插入语句
                columns = ", ".join(header)
                placeholders = ", ".join(["%s"] * len(header))
                sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                batch = []

                for row in csv_reader:
                    # 处理空值(将空字符串转换为None)
                    processed_row = [None if value == "" else value for value in row]
                    batch.append(processed_row)

                    if len(batch) >= batch_size:
                        cursor.executemany(sql, batch)
                        connection.commit()
                        batch = []

                # 插入剩余数据
                if batch:
                    cursor.executemany(sql, batch)
                    connection.commit()

                print(f"成功导入 {csv_reader.line_num - 1} 条数据到MySQL数据库")

    except Error as e:
        print(f"导入数据时出错: {e}")


# 删除
def delete_table_alldata(connection, table_name):
    # 安全确认
    confirm = input(
        f"⚠️ 即将清空表 {table_name} 中的所有数据，此操作不可恢复！\n请输入'y'确认操作: "
    )
    if confirm.lower() != "y":
        print("操作已取消")
        return
    try:
        with connection.cursor() as cursor:
            # 方法1: 使用TRUNCATE (更快，不可回滚)
            # cursor.execute(f"TRUNCATE TABLE {table_name}")

            # 方法2: 使用DELETE (可回滚)
            cursor.execute(f"DELETE FROM {table_name}")

            # 重置自增ID计数器(如果使用TRUNCATE则不需要)
            cursor.execute(f"ALTER TABLE {table_name} AUTO_INCREMENT = 1")

        connection.commit()
        print(f"已成功清空表 {table_name} 中的所有数据")

    except Error as e:
        print(f"清空表时出错: {e}")
        if connection:
            connection.rollback()


def drop_table(connection, table_name):

    # 安全确认
    confirm = input(
        f"⚠️ 即将永久删除表 {table_name} 及其所有数据！\n请输入'y'确认操作: "
    )
    if confirm.lower() != "y":
        print("操作已取消")
        return False

    connection = None
    try:
        # 连接数据库
        with connection.cursor() as cursor:

            # 检查表是否存在
            cursor.execute(f"SHOW TABLES LIKE '{table_name}'")

            if not cursor.fetchone():
                print(f"表 {table_name} 不存在")
                return False

            # 执行删除表操作
            cursor.execute(f"DROP TABLE {table_name}")
        connection.commit()
        print(f"表 {table_name} 已成功删除")
        return True

    except Error as e:
        print(f"删除表时出错: {e}")
        if connection:
            connection.rollback()
        return False


from tabulate import tabulate


def query_all_records(connection, table_name, limit=10, order_by="id"):

    if not connection:
        return None

    try:
        with connection.cursor() as cursor:  # pymysql使用with管理游标
            query = f"SELECT * FROM {table_name} ORDER BY {order_by} LIMIT %s"
            cursor.execute(query, (limit,))
            return cursor.fetchall()
    except Error as e:
        print(f"查询出错: {e}")
        return None
    finally:
        pass  # 不在这里关闭连接，统一由主程序控制


def query_with_conditions(connection, table_name, conditions, limit=10):

    if not connection:
        return None

    try:
        with connection.cursor() as cursor:
            where_clause = " AND ".join([f"{k} = %s" for k in conditions.keys()])
            query = f"SELECT * FROM {table_name} WHERE {where_clause} LIMIT %s"
            params = list(conditions.values()) + [limit]
            cursor.execute(query, params)
            return cursor.fetchall()
    except Error as e:
        print(f"条件查询出错: {e}")
        return None
    finally:
        pass  # 不在这里关闭连接，统一由主程序控制


def query_aggregate(connection, table_name, group_by, aggregate_func="COUNT(*)"):

    if not connection:
        return None

    try:
        with connection.cursor() as cursor:
            query = f"""
                SELECT {group_by}, {aggregate_func} as agg_value 
                FROM {table_name} 
                GROUP BY {group_by}
                ORDER BY agg_value DESC
            """
            cursor.execute(query)
            return cursor.fetchall()
    except Error as e:
        print(f"聚合查询出错: {e}")
        return None
    finally:
        pass  # 不在这里关闭连接，统一由主程序控制


# ==================== 高级查询函数 ====================


def query_join_tables(
    connection, table1, table2, join_condition, columns, where_condition=None, limit=5
):

    if not connection:
        return None

    try:
        with connection.cursor() as cursor:
            select_columns = ", ".join(columns)
            where_clause = f"WHERE {where_condition}" if where_condition else ""
            query = f"""
                SELECT {select_columns}
                FROM {table1} b
                JOIN {table2} i ON {join_condition}
                {where_clause}
                LIMIT %s
            """
            cursor.execute(query, (limit,))
            return cursor.fetchall()
    except Error as e:
        print(f"多表联查出错: {e}")
        return None


def query_fulltext_search(connection, table_name, text_column, search_term, limit=5):

    if not connection:
        return None

    try:
        with connection.cursor() as cursor:
            query = f"""
                SELECT id, MATCH({text_column}) AGAINST(%s) as relevance
                FROM {table_name}
                WHERE MATCH({text_column}) AGAINST(%s IN BOOLEAN MODE)
                ORDER BY relevance DESC
                LIMIT %s
            """
            cursor.execute(query, (search_term, search_term, limit))
            return cursor.fetchall()
    except Error as e:
        print(f"全文搜索出错: {e}")
        return None


# ==================== 实用查询函数 ====================


def paginated_query(
    connection, table_name, page=1, page_size=10, filters=None, order_by="id"
):

    if not connection:
        return None

    try:
        with connection.cursor() as cursor:
            # 构建WHERE条件
            where_clause = ""
            params = []
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(f"{key} = %s")
                    params.append(value)
                where_clause = "WHERE " + " AND ".join(conditions)

            # 计算偏移量
            offset = (page - 1) * page_size

            # 执行数据查询
            query = f"""
                SELECT * FROM {table_name}
                {where_clause}
                ORDER BY {order_by}
                LIMIT %s OFFSET %s
            """
            params.extend([page_size, offset])
            cursor.execute(query, params)
            results = cursor.fetchall()

            # 获取总数
            count_query = f"SELECT COUNT(*) as total FROM {table_name} {where_clause}"
            cursor.execute(count_query, params[:-2])  # 去掉LIMIT参数
            total = cursor.fetchone()["total"]

            return {
                "data": results,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size,
            }

    except Error as e:
        print(f"分页查询出错: {e}")
        return None


if __name__ == "__main__":
    # 确定数据库名称
    database_name = "mysqldb_test"  # 若还没创建，则database_name = ""
    connection = mysql_connnection(database_name=database_name)

    # 未创建数据库时，先创建数据库
    # create_database(connection,"mysqldb_test")

    # 确定表名称
    table_name = "pkltomysql"
    # 未创建表时，先创建表
    # create_table(connection,table_name)

    # 从pkl导入数据
    import_pkl_to_mysql(connection, table_name, "src/pkl_files/vector_db.pkl")

    # 基础查询示例
    print("\n=== 基础查询示例 ===")
    print("\n1. 查询前5条记录:")
    records = query_all_records(connection, table_name, limit=5)
    print(tabulate(records, headers="keys", tablefmt="grid"))

    print("\n2. 条件查询(page_number=5且block_type='text'):")
    conditions = {"page_number": 5, "block_type": "text"}
    records = query_with_conditions(connection, table_name, conditions)
    print(tabulate(records, headers="keys", tablefmt="grid"))

    print("\n3. 聚合查询(按block_type分组统计):")
    stats = query_aggregate(connection, table_name, "block_type")
    print(tabulate(stats, headers="keys", tablefmt="grid"))

    # 删除表中所有数据
    # delete_table_alldata(connection,table_name)

    if connection:
        connection.close()
        print("MySQL连接已关闭")
