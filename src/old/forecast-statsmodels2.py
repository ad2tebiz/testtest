import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def prepare_data():
    """Загрузка и подготовка данных"""
    # Загрузка данных
    df = pd.read_csv('data/airline-passengers.csv')
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    df.columns = ['Passengers']
    
    # Разделение на обучающую и тестовую выборки
    train_data = df.iloc[:-12]  # Все кроме последних 12 месяцев
    test_data = df.iloc[-12:]   # Последние 12 месяцев для тестирования
    
    return df, train_data, test_data

def train_model(train_data):
    """Обучение модели Holt-Winters"""
    # Параметры модели
    model = ExponentialSmoothing(
        train_data['Passengers'],
        trend='add',           # Аддитивная тенденция
        seasonal='add',        # Аддитивная сезонность
        seasonal_periods=12    # Годовая сезонность (12 месяцев)
    )
    
    # Обучение модели
    fitted_model = model.fit()
    return fitted_model

def make_predictions(model, train_data, test_data, future_periods=12):
    """Создание прогнозов"""
    # Прогноз на тестовый период (последние 12 месяцев)
    test_forecast = model.forecast(steps=12)
    
    # Прогноз на будущие периоды
    future_forecast = model.forecast(steps=future_periods)
    
    # Создание дат для будущего прогноза
    last_date = train_data.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=future_periods,
        freq='MS'
    )
    
    return test_forecast, future_forecast, future_dates

def evaluate_model(test_data, test_forecast):
    """Оценка качества модели"""
    mae = mean_absolute_error(test_data['Passengers'], test_forecast)
    mse = mean_squared_error(test_data['Passengers'], test_forecast)
    rmse = np.sqrt(mse)
    
    print(f"Оценка модели на тестовых данных:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    return mae, mse, rmse

def plot_results(df, test_forecast, future_forecast, future_dates):
    """Визуализация результатов"""
    plt.figure(figsize=(12, 6))
    
    # Исходные данные
    plt.plot(df.index, df['Passengers'], label='Исходные данные', color='blue')
    
    # Прогноз на тестовый период
    test_dates = df.index[-12:]
    plt.plot(test_dates, test_forecast, label='Прогноз на тест', color='red', linestyle='--')
    
    # Прогноз на будущее
    plt.plot(future_dates, future_forecast, label='Будущий прогноз', color='green', linestyle='--')
    
    plt.title('Прогноз пассажиров авиалиний с использованием Holt-Winters')
    plt.xlabel('Дата')
    plt.ylabel('Количество пассажиров')
    plt.legend()
    plt.grid(True)
    
    # Сохранение графика
    os.makedirs('data/statsmodels2', exist_ok=True)
    plt.savefig('data/statsmodels2/forecast_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(test_forecast, future_forecast, future_dates):
    """Сохранение результатов в CSV"""
    # Создание DataFrame с результатами
    results_df = pd.DataFrame({
        'Date': list(test_forecast.index) + list(future_dates),
        'Forecast': list(test_forecast.values) + list(future_forecast.values),
        'Type': ['test'] * len(test_forecast) + ['future'] * len(future_forecast)
    })
    
    # Сохранение в CSV
    os.makedirs('data/statsmodels2', exist_ok=True)
    results_df.to_csv('data/statsmodels2/forecast_results.csv', index=False)
    print("Результаты сохранены в data/statsmodels2/forecast_results.csv")

def main():
    """Основная функция"""
    print("Загрузка данных...")
    df, train_data, test_data = prepare_data()
    
    print("Обучение модели Holt-Winters...")
    model = train_model(train_data)
    
    print("Создание прогнозов...")
    test_forecast, future_forecast, future_dates = make_predictions(model, train_data, test_data)
    
    print("Оценка модели...")
    evaluate_model(test_data, test_forecast)
    
    print("Визуализация результатов...")
    plot_results(df, test_forecast, future_forecast, future_dates)
    
    print("Сохранение результатов...")
    save_results(test_forecast, future_forecast, future_dates)
    
    print("Готово!")

if __name__ == "__main__":
    main()