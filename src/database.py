import os
import psycopg2
from dotenv import load_dotenv

# Загружаем переменные из файла .env
load_dotenv()

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        print("Подключение к БД установлено успешно!")
        return conn
    except Exception as e:
        print(f"Ошибка подключения к БД: {e}")
        return None

def create_table():
    conn = get_db_connection()
    if conn is None:
        return
    
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS db_test (
        id SERIAL PRIMARY KEY,
        date DATE NOT NULL,
        name VARCHAR(25),
        age INTEGER
    );
    '''
    try:
        cur = conn.cursor()
        cur.execute(create_table_query)
        conn.commit() # Подтверждаем изменения в БД
        print("Таблица 'db_test' создана или уже существует.")
        cur.close()
    except Exception as e:
        print(f"Ошибка при создании таблицы: {e}")
    finally:
        if conn is not None:
            conn.close() # Всегда закрываем соединение

def insert_data(date, name, age):
    conn = get_db_connection()
    if conn is None:
        return

    insert_query = '''
    INSERT INTO db_test (date, name, age)
    VALUES (%s, %s, %s);
    '''
    try:
        cur = conn.cursor()
        # Выполняем запрос, подставляя значения
        cur.execute(insert_query, (date, name, age))
        conn.commit()
        print(f"Добавлена запись на {date}.")
        cur.close()
    except Exception as e:
        print(f"Ошибка при вставке данных: {e}")
    finally:
        if conn is not None:
            conn.close()

def read_data():
    conn = get_db_connection()
    if conn is None:
        return

    select_query = '''SELECT * FROM db_test;'''
    try:
        cur = conn.cursor()
        cur.execute(select_query)
        records = cur.fetchall() # Получаем все строки

        print("\nДанные из таблицы 'db_test':")
        for row in records:
            print(row)
        cur.close()
    except Exception as e:
        print(f"Ошибка при чтении данных: {e}")
    finally:
        if conn is not None:
            conn.close()

# Этот блок выполнится только если этот файл запущен напрямую, а не импортирован
if __name__ == "__main__":
    # Демонстрация работы модуля
    create_table()
    # Вставим тестовые данные
    insert_data('2025-09-25', "Андрей", 25)
    insert_data('2025-09-26', "Иван", 28)
    # Прочитаем и выведем что получилось
    read_data()