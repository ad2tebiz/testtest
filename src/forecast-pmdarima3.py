import pandas as pd
import numpy as np
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import warnings
from datetime import datetime

# Фильтрация только важных предупреждений
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Загружаем переменные окружения
load_dotenv()

def create_directories():
    """Создает необходимые директории если они не существуют"""
    os.makedirs('data/pmdarima3', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def load_and_prepare_data():
    """Загрузка и подготовка данных"""
    try:
        data = pd.read_csv('data/airline-passengers.csv', parse_dates=['Month'], index_col='Month')
        data = data.rename(columns={'Passengers': 'passengers'})
        
        print("=" * 60)
        print("АНАЛИЗ ДАННЫХ:")
        print(f"Размер данных: {data.shape}")
        print(f"Период данных: {data.index.min()} - {data.index.max()}")
        print(f"Количество месяцев: {len(data)}")
        print("\nПервые 10 строк исходных данных:")
        print(data.head(10))
        print("\nПоследние 5 строк исходных данных:")
        print(data.tail())
        print(f"\nПроверка на пропущенные значения: {data.isnull().sum().sum()}")
        
        return data
    except FileNotFoundError:
        print("Ошибка: Файл data/airline-passengers.csv не найден!")
        print("Убедитесь, что файл существует по указанному пути")
        return None

def plot_original_data(data):
    """Визуализация исходных данных"""
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data['passengers'], label='Исходные данные', linewidth=2, marker='o', markersize=3)
    plt.title('Количество пассажиров авиалиний по месяцам (1949-1960)', fontsize=14, fontweight='bold')
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Количество пассажиров', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('data/pmdarima3/original_data_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_sarima_model(data):
    """Обучение модели SARIMA"""
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ SARIMA:")
    
    try:
        model = auto_arima(data['passengers'],
                          seasonal=True,
                          m=12,
                          stepwise=True,
                          trace=True,  # Включаем вывод процесса для наглядности
                          error_action='ignore',
                          suppress_warnings=True,
                          information_criterion='aic',
                          max_order=None,
                          test='adf')  # Тест на стационарность
        
        print("\n" + "=" * 50)
        print("ОТЧЕТ ПО МОДЕЛИ:")
        print(model.summary())
        
        return model
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        return None

def make_forecast(model, data, periods=24):
    """Построение прогноза"""
    print("\n" + "=" * 60)
    print("ПОСТРОЕНИЕ ПРОГНОЗА:")
    
    try:
        forecast, conf_int = model.predict(n_periods=periods, return_conf_int=True, alpha=0.05)
        
        # Создание временного индекса для прогноза
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                   periods=periods, 
                                   freq='M')
        
        # Создание DataFrame с прогнозом
        forecast_df = pd.DataFrame({
            'forecast_date': future_dates,
            'predicted_value': forecast,
            'confidence_lower': conf_int[:, 0],
            'confidence_upper': conf_int[:, 1],
            'model_type': 'sarima_auto',
            'created_at': datetime.now()
        })
        
        forecast_df.set_index('forecast_date', inplace=True)
        
        print(f"Прогноз построен на {periods} месяцев")
        print(f"Период прогноза: {future_dates[0]} - {future_dates[-1]}")
        
        return forecast_df, future_dates, forecast, conf_int
        
    except Exception as e:
        print(f"Ошибка при построении прогноза: {e}")
        return None, None, None, None

def plot_forecast_results(data, forecast_df, future_dates, forecast, conf_int):
    """Визуализация результатов прогноза"""
    plt.figure(figsize=(16, 9))
    
    # Исторические данные
    plt.plot(data.index, data['passengers'], 
             label='Исторические данные', color='blue', linewidth=2, marker='o', markersize=3)
    
    # Прогноз
    plt.plot(future_dates, forecast, 
             label='Прогноз SARIMA', color='red', linewidth=2, linestyle='--', marker='s', markersize=4)
    
    # Доверительный интервал
    plt.fill_between(future_dates, 
                     conf_int[:, 0], 
                     conf_int[:, 1], 
                     color='pink', alpha=0.3, label='Доверительный интервал (95%)')
    
    plt.title('Прогноз количества пассажиров авиалиний на 2 года вперед', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Количество пассажиров', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/pmdarima3/forecast_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_forecast_to_csv(forecast_df):
    """Сохранение прогноза в CSV"""
    try:
        # Сохраняем все колонки
        forecast_df.to_csv('data/pmdarima3/forecast_results.csv', index=True)
        print(f"Прогноз сохранен в 'data/pmdarima3/forecast_results.csv'")
        
        # Дополнительно сохраняем краткую версию
        forecast_df[['predicted_value', 'confidence_lower', 'confidence_upper']].to_csv(
            'data/pmdarima3/forecast_simple.csv'
        )
        
    except Exception as e:
        print(f"Ошибка при сохранении в CSV: {e}")

def save_to_database(forecast_df):
    """Сохранение прогноза в базу данных"""
    print("\n" + "=" * 60)
    print("СОХРАНЕНИЕ В БАЗУ ДАННЫХ:")

    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    
    if not all([DB_HOST, DB_NAME, DB_USER]):
        print("Предупреждение: Не все переменные окружения установлены")
        return
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=10
        )
        cursor = conn.cursor()

        # Сначала исправляем структуру таблицы
        print("Проверка и исправление структуры таблицы...")
        
        # Удаляем старое ограничение если оно существует
        try:
            cursor.execute("ALTER TABLE forecasts DROP CONSTRAINT IF EXISTS forecasts_forecast_date_key")
        except:
            pass  # Игнорируем ошибки если ограничения нет
            
        # Добавляем составное ограничение (игнорируем если уже существует)
        try:
            cursor.execute("""
                ALTER TABLE forecasts 
                ADD CONSTRAINT IF NOT EXISTS forecasts_date_model_unique 
                UNIQUE (forecast_date, model_name)
            """)
        except:
            pass  # Игнорируем ошибки если ограничение уже есть
            
        conn.commit()
        
        # Теперь безопасно вставляем данные
        print("Вставка новых записей...")
        
        # Подготовка данных
        records = [
            (row.name.date(),
             float(row['predicted_value']), 
             float(row['confidence_lower']), 
             float(row['confidence_upper']),
             'sarima_auto')
            for _, row in forecast_df.iterrows()
        ]

        # Используем ON CONFLICT с правильным составным ограничением
        insert_query = """
            INSERT INTO forecasts (forecast_date, predicted_value, confidence_lower, confidence_upper, model_name)
            VALUES %s
            ON CONFLICT (forecast_date, model_name) DO UPDATE SET
                predicted_value = EXCLUDED.predicted_value,
                confidence_lower = EXCLUDED.confidence_lower,
                confidence_upper = EXCLUDED.confidence_upper,
                created_at = CURRENT_TIMESTAMP
        """

        execute_values(cursor, insert_query, records)
        conn.commit()
        
        print(f"Успешно добавлено/обновлено {len(records)} записей")

        # Проверка
        cursor.execute("SELECT model_name, COUNT(*) FROM forecasts GROUP BY model_name")
        model_counts = cursor.fetchall()
        print("Статистика по моделям:")
        for model, count in model_counts:
            print(f"   {model}: {count} записей")

    except Exception as e:
        print(f"Ошибка при работе с базой данных: {e}")

    finally:
        if 'conn' in locals() and conn:
            cursor.close()
            conn.close()
            print("Соединение с базой данных закрыто")

def save_to_database_simple(forecast_df):
    """Простое сохранение с уникальным именем модели"""
    print("\n" + "=" * 60)
    print("СОХРАНЕНИЕ В БАЗУ ДАННЫХ:")
    
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'main_db')
    DB_USER = os.getenv('DB_USER', 'user')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        
        # Уникальное имя модели с временной меткой
        model_name = f"sarima_auto_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Подготовка данных
        records = [
            (row.name.date(),
             float(row['predicted_value']), 
             float(row['confidence_lower']), 
             float(row['confidence_upper']),
             model_name)
            for _, row in forecast_df.iterrows()
        ]

        insert_query = """
            INSERT INTO forecasts (forecast_date, predicted_value, confidence_lower, confidence_upper, model_name)
            VALUES %s
        """

        execute_values(cursor, insert_query, records)
        conn.commit()
        
        print(f"Успешно добавлено {len(records)} записей как '{model_name}'")
        
        cursor.execute("SELECT model_name, COUNT(*) FROM forecasts GROUP BY model_name")
        model_counts = cursor.fetchall()
        print("Все модели в базе:")
        for model, count in model_counts:
            print(f"   {model}: {count} записей")

    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        if 'conn' in locals() and conn:
            cursor.close()
            conn.close()

def main():
    """Основная функция"""
    print("ЗАПУСК ПРОГНОЗИРОВАНИЯ ВРЕМЕННЫХ РЯДОВ")
    # Создание директорий
    create_directories()
    
    # Загрузка данных
    data = load_and_prepare_data()
    if data is None:
        return
    
    # Визуализация исходных данных
    plot_original_data(data)
    
    # Обучение модели
    model = train_sarima_model(data)
    if model is None:
        return
    
    # Построение прогноза
    forecast_df, future_dates, forecast, conf_int = make_forecast(model, data)
    if forecast_df is None:
        return
    
    # Визуализация результатов
    plot_forecast_results(data, forecast_df, future_dates, forecast, conf_int)
    
    # Сохранение в CSV
    save_forecast_to_csv(forecast_df)
    
    # Сохранение в базу данных
    save_to_database(forecast_df)
    
    print("\n" + "=" * 60)
    print("Готово! Прогноз успешно построен и сохранен.")
    print(f"Результаты сохранены в:")
    print(f"   - data/pmdarima3/forecast_results.csv")
    print(f"   - data/pmdarima3/forecast_simple.csv") 
    print(f"   - data/pmdarima3/original_data_plot.png")
    print(f"   - data/pmdarima3/forecast_plot.png")

if __name__ == "__main__":
    main()