import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import warnings

# Фильтрация всех предупреждений
warnings.filterwarnings('ignore')

# Загружаем переменные окружения
load_dotenv()

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
    interval_width=0.99,       # ширина доверительного интервала
    uncertainty_samples=1000   # количество сэмплов для uncertainty intervals
)

# Обучение модели

if model.growth=="logistic":
    data['cap'] = data['y'].max() * 1.2 # для расчета при параметре logistic

model.fit(data)

# 4. Построение прогноза
forecast_periods = 24  # Прогнозируем на 2 года вперед
future = model.make_future_dataframe(periods=forecast_periods, freq='M', include_history=False)
if model.growth=="logistic":
    future['cap'] = data['cap'].max() # для расчета при параметре logistic
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
plt.plot(forecast_df.index, forecast_df['forecast'], label='Прогноз', color='red', linewidth=2)

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
fig2 = model.plot_components(forecast)
plt.suptitle('Компоненты прогноза Prophet: тренд и сезонность', y=1.02)
plt.savefig('data/prophet/prophet_components.png', dpi=300, bbox_inches='tight')
plt.show()



# 7. Сохранение прогноза в CSV
forecast_df[['forecast']].to_csv('data/prophet/forecast_results_prophet.csv')
print(f"\nПрогноз сохранен в 'data/prophet/forecast_results_prophet.csv'")

# 8. Запись в базу данных PostgreSQL
print("\nПодключение к базе данных...")
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Подготовка данных для вставки
records = [
    (row['period'], row['forecast'], row['confidence_lower'], row['confidence_upper'], 'prophet')
    for _, row in forecast_df.reset_index().iterrows()
]

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

    # SQL запрос для вставки данных
    insert_query = """
    INSERT INTO forecasts (forecast_date, predicted_value, confidence_lower, confidence_upper, model_name)
    VALUES %s
    ON CONFLICT (forecast_date, model_name) DO UPDATE
    SET
        predicted_value = EXCLUDED.predicted_value,
        confidence_lower = EXCLUDED.confidence_lower,
        confidence_upper = EXCLUDED.confidence_upper,
        model_name = EXCLUDED.model_name,
        created_at = CURRENT_TIMESTAMP;
    """

    # Выполнение пакетной вставки
    execute_values(cursor, insert_query, records)
    conn.commit()
    print(f"Успешно добавлено/обновлено {len(records)} записей в базе данных")

    # Проверка записанных данных
    cursor.execute("SELECT COUNT(*) FROM forecasts;")
    count = cursor.fetchone()
    print(f"Всего записей в таблице forecasts: {count}")

except Exception as e:
    print(f"Ошибка при работе с базой данных: {e}")
finally:
    if conn:
        cursor.close()
        conn.close()
        print("Соединение с базой данных закрыто")

print("\nГотово! Прогноз построен и сохранен с использованием Prophet.")