import mysql.connector

def get_db_connection():
    """ 创建并返回 MySQL 数据库连接 """
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Xwy021103",  # 个人 MySQL 密码
        database="attendance_system"
    )
    return db
