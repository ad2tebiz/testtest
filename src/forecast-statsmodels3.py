import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import warnings
from datetime import datetime

# Фильтрация предупреждений
warnings.filterwarnings('ignore')

# Загружаем переменные окружения
load_dotenv()

def create_directories():
    """Создает необходимые директории если они не существуют"""
    os.makedirs('data/statsmodels3', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def prepare_data():
    """Загрузка и подготовка данных"""
    try:
        # Загрузка данных
        df = pd.read_csv('data/airline-passengers.csv')
        df['Month'] = pd.to_datetime(df['Month'])
        df.set_index('Month', inplace=True)
        df.columns = ['passengers']
        
        print("=" * 60)
        print("АНАЛИЗ ДАННЫХ:")
        print(f"Размер данных: {df.shape}")
        print(f"Период данных: {df.index.min()} - {df.index.max()}")
        print(f"Количество месяцев: {len(df)}")
        print("\nПервые 10 строк исходных данных:")
        print(df.head(10))
        print("\nПоследние 5 строк исходных данных:")
        print(df.tail())
        
        return df
    except FileNotFoundError:
        print("Ошибка: Файл data/airline-passengers.csv не найден!")
        print("Убедитесь, что файл существует по указанному пути")
        return None

def plot_original_data(df):
    """Визуализация исходных данных"""
    plt.figure(figsize=(14, 8))
    plt.plot(df.index, df['passengers'], label='Исходные данные', color='blue', linewidth=2, marker='o', markersize=3)
    plt.title('Количество пассажиров авиалиний по месяцам (1949-1960)', fontsize=14, fontweight='bold')
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Количество пассажиров', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('data/statsmodels3/original_data_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_sarima_model(df):
    """Обучение модели SARIMA на всех данных"""
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ SARIMA (на всех данных):")
    
    try:
        # Параметры SARIMA: (p, d, q) × (P, D, Q, s)
        # (1, 1, 1) × (1, 1, 1, 12) - популярные параметры для месячных данных с годовой сезонностью
        model = SARIMAX(
            df['passengers'],
            order=(1, 1, 1),           # Несезонные параметры (p, d, q)
            seasonal_order=(1, 1, 1, 12),  # Сезонные параметры (P, D, Q, s)
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Обучение модели
        fitted_model = model.fit(disp=False)
        
        print("Модель SARIMA успешно обучена на всех данных!")
        print(f"Параметры модели: {fitted_model.params}")
        print(f"AIC: {fitted_model.aic:.2f}")
        print(f"BIC: {fitted_model.bic:.2f}")
        
        return fitted_model
    except Exception as e:
        print(f"Ошибка при обучении модели SARIMA: {e}")
        return None

def make_sarima_forecast(model, df, future_periods=24):
    """Создание прогноза SARIMA на будущие периоды"""
    print("\n" + "=" * 60)
    print("ПОСТРОЕНИЕ ПРОГНОЗА SARIMA НА БУДУЩИЕ ПЕРИОДЫ:")
    
    try:
        # Прогноз на будущие периоды с доверительным интервалом
        forecast_result = model.get_forecast(steps=future_periods)
        forecast = forecast_result.predicted_mean
        confidence_int = forecast_result.conf_int()
        
        # Создание дат для будущего прогноза
        last_actual_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_actual_date + pd.DateOffset(months=1),
            periods=future_periods,
            freq='MS'
        )
        forecast.index = future_dates
        confidence_int.index = future_dates
        
        # Подготовка DataFrame для сохранения в базу
        forecast_df = pd.DataFrame({
            'forecast_date': future_dates,
            'predicted_value': forecast.values,
            'confidence_lower': confidence_int.iloc[:, 0].values,
            'confidence_upper': confidence_int.iloc[:, 1].values,
            'model_type': 'sarima',
            'created_at': datetime.now()
        })
        forecast_df.set_index('forecast_date', inplace=True)
        
        print("БУДУЩИЙ ПРОГНОЗ SARIMA:")
        print(f"Период: {future_dates[0]} - {future_dates[-1]}")
        print(f"Количество периодов: {future_periods} месяцев")
        print(f"Последняя историческая дата: {last_actual_date}")
        print(f"Первая прогнозная дата: {future_dates[0]}")
        print(f"Среднее значение прогноза: {forecast.mean():.2f}")
        
        return forecast, confidence_int, future_dates, forecast_df
        
    except Exception as e:
        print(f"Ошибка при построении прогноза SARIMA: {e}")
        return None, None, None, None

def plot_sarima_results(df, model, forecast, confidence_int, future_dates):
    """Визуализация результатов SARIMA"""
    plt.figure(figsize=(16, 10))
    
    # Исторические данные
    plt.plot(df.index, df['passengers'], 
             label='Исторические данные', color='blue', linewidth=2, marker='o', markersize=3)
    
    # Будущий прогноз
    plt.plot(future_dates, forecast, 
             label='Будущий прогноз SARIMA', color='red', linewidth=2, linestyle='--', marker='s', markersize=4)
    
    # Доверительный интервал
    plt.fill_between(future_dates, 
                    confidence_int.iloc[:, 0], 
                    confidence_int.iloc[:, 1], 
                    color='red', alpha=0.2, label='Доверительный интервал 95%')
    
    # Вертикальная линия разделения
    last_historical_date = df.index[-1]
    plt.axvline(x=last_historical_date, color='gray', linestyle=':', alpha=0.7, 
                label=f'Конец исторических данных ({last_historical_date.strftime("%Y-%m")})')
    
    plt.title('Прогноз количества пассажиров авиалиний (SARIMA)\nНа основе всех исторических данных', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Количество пассажиров', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/statsmodels3/sarima_forecast_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_sarima_diagnostics(model):
    """Построение диагностических графиков SARIMA"""
    try:
        # Диагностические графики
        diagnostics = model.plot_diagnostics(figsize=(15, 12))
        plt.suptitle('Диагностические графики модели SARIMA', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('data/statsmodels3/sarima_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Диагностические графики SARIMA сохранены")
    except Exception as e:
        print(f"Ошибка при построении диагностических графиков: {e}")

def save_forecast_to_csv(forecast_df, forecast, future_dates):
    """Сохранение прогноза в CSV"""
    try:
        # Сохраняем полный прогноз
        forecast_df.to_csv('data/statsmodels3/sarima_forecast_results.csv', index=True)
        print("Прогноз SARIMA сохранен в 'data/statsmodels3/sarima_forecast_results.csv'")
        
        # Сохраняем упрощенную версию
        simple_forecast = pd.DataFrame({
            'date': future_dates,
            'predicted_passengers': forecast.values,
            'confidence_lower': forecast_df['confidence_lower'].values,
            'confidence_upper': forecast_df['confidence_upper'].values
        })
        simple_forecast.to_csv('data/statsmodels3/sarima_simple_forecast.csv', index=False)
        print("Упрощенный прогноз SARIMA сохранен в 'data/statsmodels3/sarima_simple_forecast.csv'")
        
    except Exception as e:
        print(f"Ошибка при сохранении в CSV: {e}")

def save_to_database(forecast_df):
    """Сохранение прогноза в базу данных"""
    print("\n" + "=" * 60)
    print("СОХРАНЕНИЕ ПРОГНОЗА SARIMA В БАЗУ ДАННЫХ:")

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

        # Подготовка данных для вставки
        records = [
            (row.name.date(),
             float(row['predicted_value']), 
             float(row['confidence_lower']), 
             float(row['confidence_upper']),
             'sarima')
            for _, row in forecast_df.iterrows()
        ]

        # SQL запрос для вставки данных
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
        
        print(f"Успешно добавлено/обновлено {len(records)} записей прогноза SARIMA")

        # Проверка записанных данных
        cursor.execute("SELECT model_name, COUNT(*) FROM forecasts GROUP BY model_name")
        model_counts = cursor.fetchall()
        print("Статистика по моделям в базе данных:")
        for model, count in model_counts:
            print(f"   {model}: {count} записей")

    except Exception as e:
        print(f"Ошибка при работе с базой данных: {e}")

    finally:
        if 'conn' in locals() and conn:
            cursor.close()
            conn.close()
            print("Соединение с базой данных закрыто")

def main():
    """Основная функция"""
    print("ЗАПУСК ПРОГНОЗИРОВАНИЯ ВРЕМЕННЫХ РЯДОВ (SARIMA)")
    print("Используются ВСЕ исторические данные для прогноза")
    
    # Создание директорий
    create_directories()
    
    # Загрузка данных
    df = prepare_data()
    if df is None:
        return
    
    # Визуализация исходных данных
    plot_original_data(df)
    
    # Обучение модели SARIMA на всех данных
    model = train_sarima_model(df)
    if model is None:
        return
    
    # Построение прогноза на будущие периоды
    forecast, confidence_int, future_dates, forecast_df = make_sarima_forecast(model, df, future_periods=24)
    if forecast_df is None:
        return
    
    # Визуализация результатов
    plot_sarima_results(df, model, forecast, confidence_int, future_dates)
    
    # Диагностические графики
    plot_sarima_diagnostics(model)
    
    # Сохранение результатов в CSV
    save_forecast_to_csv(forecast_df, forecast, future_dates)
    
    # Сохранение прогноза в базу данных
    save_to_database(forecast_df)
    
    print("\n" + "=" * 60)
    print("Готово! Прогноз SARIMA успешно построен и сохранен.")
    print(f"\nРезультаты сохранены в:")
    print(f"   - data/statsmodels3/sarima_forecast_results.csv (полный прогноз)")
    print(f"   - data/statsmodels3/sarima_simple_forecast.csv (упрощенный прогноз)")
    print(f"   - data/statsmodels3/original_data_plot.png")
    print(f"   - data/statsmodels3/sarima_forecast_plot.png")
    print(f"   - data/statsmodels3/sarima_diagnostics.png")
    print(f"\nПрогноз построен на основе {len(df)} месяцев исторических данных")
    print(f"Спрогнозировано {len(forecast)} будущих месяцев")

if __name__ == "__main__":
    main()