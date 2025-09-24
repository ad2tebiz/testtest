import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import warnings
import datetime

# Фильтрация всех предупреждений
warnings.filterwarnings('ignore')

# Загружаем переменные окружения
load_dotenv()

def save_to_database_prophet(forecast_df, model_name_suffix=""):
    """Сохранение прогноза Prophet в базу данных"""
    print("\nПодключение к базе данных...")
    
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'main_db')
    DB_USER = os.getenv('DB_USER', 'user')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    
    # Создаем уникальное имя модели
    if model_name_suffix:
        model_name = f"prophet_{model_name_suffix}"
    else:
        model_name = f"prophet_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Подключение к базе данных
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()

        # Сначала удаляем старые записи этой модели
        cursor.execute("DELETE FROM forecasts WHERE model_name LIKE 'prophet%'")
        deleted_count = cursor.rowcount
        print(f"Удалено {deleted_count} старых записей Prophet")

        # Подготовка данных для вставки
        records = [
            (row['period'], row['forecast'], row['confidence_lower'], row['confidence_upper'], model_name)
            for _, row in forecast_df.reset_index().iterrows()
        ]

        # SQL запрос для вставки данных (упрощенный, без ON CONFLICT)
        insert_query = """
        INSERT INTO forecasts (forecast_date, predicted_value, confidence_lower, confidence_upper, model_name)
        VALUES %s
        """

        # Выполнение пакетной вставки
        execute_values(cursor, insert_query, records)
        conn.commit()
        print(f"Успешно добавлено {len(records)} записей модели '{model_name}' в базу данных")

        # Проверка записанных данных
        cursor.execute("SELECT model_name, COUNT(*) FROM forecasts GROUP BY model_name;")
        model_counts = cursor.fetchall()
        print("Статистика по моделям в базе данных:")
        for model, count in model_counts:
            print(f"  {model}: {count} записей")

    except Exception as e:
        print(f"Ошибка при работе с базой данных: {e}")
    finally:
        if 'conn' in locals() and conn:
            cursor.close()
            conn.close()
            print("Соединение с базой данных закрыто")

# 1. Загрузка и подготовка данных
data = pd.read_csv('data/airline-passengers.csv', parse_dates=['Month'], index_col='Month')

# Переименовываем колонку для удобства
data = data.rename(columns={'Passengers': 'y'})
data = data.reset_index()
data = data.rename(columns={'Month': 'ds'})

print("Первые 10 строк исходных данных:")
print(data.head(10))
print("Последние 5 строк исходных данных:")
print(data.tail())

# 2. Визуализация исходных данных
plt.figure(figsize=(12, 8))
plt.plot(data['ds'], data['y'], label='Исходные данные')
plt.title('Количество пассажиров авиалиний по месяцам (1949-1960)')
plt.xlabel('Дата')
plt.ylabel('Количество пассажиров')
plt.grid(True)
plt.legend()
plt.savefig('data/prophet/original_data_plot.png')
plt.show()

# 3. Создание и обучение модели Prophet
print(f"\nРазмер обучающей выборки: {len(data)} месяцев")
print("\nОбучение модели Prophet...")

# Основные параметры для экспериментов:
model = Prophet(
    growth='linear',           # тип тренда: 'linear' или 'logistic'
    seasonality_mode='multiplicative',  # 'additive' или 'multiplicative'
    yearly_seasonality=True,   # автоматическая годовая сезонность
    weekly_seasonality=False,  # недельная сезонность (не нужна для месячных данных)
    daily_seasonality=False,   # дневная сезонность (не нужна)
    changepoint_prior_scale=0.05,  # гибкость тренда (чем выше, тем гибче)
    seasonality_prior_scale=10.0,  # сила сезонности
    holidays_prior_scale=10.0,     # сила влияния праздников
    mcmc_samples=0,            # 0 для точечной оценки, >0 для uncertainty intervals
    interval_width=0.95,       # ширина доверительного интервала
    uncertainty_samples=1000   # количество сэмплов для uncertainty intervals
)

# Обучение модели
model.fit(data)

# 4. Построение прогноза
forecast_periods = 24  # Прогнозируем на 2 года вперед
future = model.make_future_dataframe(periods=forecast_periods, freq='M', include_history=False)
forecast = model.predict(future)

print("\n" + "=" * 50)
print("ОТЧЕТ ПО МОДЕЛИ PROPHET:")
print("Компоненты прогноза доступны в объекте forecast")

# 5. Подготовка данных прогноза
forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
forecast_df = forecast_df.rename(columns={
    'ds': 'period',
    'yhat': 'forecast',
    'yhat_lower': 'confidence_lower',
    'yhat_upper': 'confidence_upper'
})
forecast_df.set_index('period', inplace=True)

# 6. Визуализация результатов
plt.figure(figsize=(14, 7))

# Исторические данные
plt.plot(data['ds'], data['y'], label='Исторические данные', color='blue', linewidth=2)

# Прогноз
plt.plot(forecast_df.index, forecast_df['forecast'], label='Прогноз Prophet', color='red', linewidth=2)

# Доверительный интервал
plt.fill_between(forecast_df.index,
                 forecast_df['confidence_lower'],
                 forecast_df['confidence_upper'],
                 color='pink', alpha=0.3, label='Доверительный интервал (95%)')

plt.title('Прогноз количества пассажиров авиалиний на 2 года (Prophet)')
plt.xlabel('Дата')
plt.ylabel('Количество пассажиров')
plt.legend()
plt.grid(True)
plt.savefig('data/prophet/forecast_plot_prophet.png', dpi=300, bbox_inches='tight')
plt.show()

# Дополнительная визуализация компонентов Prophet
try:
    fig2 = model.plot_components(forecast)
    plt.suptitle('Компоненты прогноза Prophet: тренд и сезонность', y=1.02)
    plt.savefig('data/prophet/prophet_components.png', dpi=300, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Не удалось построить компоненты: {e}")

# 7. Сохранение прогноза в CSV
forecast_df.to_csv('data/prophet/forecast_results_prophet.csv')
print(f"\nПрогноз сохранен в 'data/prophet/forecast_results_prophet.csv'")

# 8. Запись в базу данных PostgreSQL
save_to_database_prophet(forecast_df, "multiplicative")

print("\nГотово! Прогноз построен и сохранен с использованием Prophet.")