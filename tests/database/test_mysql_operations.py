from tabulate import tabulate
from src.database.db_connection import mysql_connnection
from src.database.operations.mysql_operations import (
    create_database,
    create_table,
    import_pkl_to_mysql,
    query_all_records,
    query_with_conditions,
    query_aggregate,
    delete_table_alldata,
    drop_table,
)

if __name__ == "__main__":
    # 数据库和表配置
    database_name = "mysqldb_test"
    table_name = "pkltomysql"
    pkl_path = "src/pkl_files/vector_db.pkl"

    # 建立连接
    connection = mysql_connnection(database_name=database_name)

    # 可选：创建数据库（如未手动创建）
    # create_database(connection, database_name)

    # 创建表（如不存在）
    create_table(connection, table_name)

    # 导入数据
    import_pkl_to_mysql(connection, table_name, pkl_path)

    # 查询前5条记录
    print("\n=== 查询前5条记录 ===")
    records = query_all_records(connection, table_name, limit=5)
    print(tabulate(records, headers="keys", tablefmt="grid"))

    # 条件查询
    print("\n=== 条件查询 page_number = 5 且 block_type = 'text' ===")
    conditions = {"page_number": 5, "block_type": "text"}
    result = query_with_conditions(connection, table_name, conditions)
    print(tabulate(result, headers="keys", tablefmt="grid"))

    # 聚合查询
    print("\n=== 按 block_type 分组统计 ===")
    stats = query_aggregate(connection, table_name, "block_type")
    print(tabulate(stats, headers="keys", tablefmt="grid"))

    # ✅ 关闭连接
    if connection:
        connection.close()
        print("\n✅ MySQL连接已关闭")
