import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== Демонстрация pmdarima ===")
    
    # 1. Загрузка данных
    print("1. Загрузка данных...")
    df = pd.read_csv('data/airline-passengers.csv')
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    df.columns = ['Passengers']
    
    print(f"Данные за период: {df.index[0]} - {df.index[-1]}")
    print(f"Количество наблюдений: {len(df)}")
    
    # 2. Визуализация исходных данных
    print("2. Визуализация данных...")
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Passengers'])
    plt.title('Исходные данные: Пассажиры авиакомпаний')
    plt.xlabel('Дата')
    plt.ylabel('Количество пассажиров')
    plt.grid(True)
    plt.savefig('data/pmdarima/original_data.png')
    plt.show()
    
    # 3. Автоматический подбор ARIMA модели
    print("3. Автоматический подбор ARIMA модели...")
    print("Подбор параметров (это может занять несколько минут):")
    
    model = auto_arima(
        df['Passengers'],
        seasonal=True,           # Включаем сезонность
        m=12,                    # Период сезонности (12 месяцев)
        stepwise=True,           # Пошаговый поиск для ускорения
        trace=True,              # Вывод процесса подбора
        error_action='ignore',   # Игнорировать ошибки
        suppress_warnings=True,  # Подавить предупреждения
        n_fits=30                # Количество тестируемых моделей
    )
    
    print(f"\n4. Лучшая модель найдена: {model}")
    print(f"Параметры модели (p,d,q)(P,D,Q,s): {model.order} {model.seasonal_order}")
    
    # 4. Прогноз на 12 месяцев вперед
    print("5. Создание прогноза на 12 месяцев...")
    forecast, conf_int = model.predict(
        n_periods=12, 
        return_conf_int=True
    )
    
    # 5. Подготовка результатов
    future_dates = pd.date_range(
        start=df.index[-1] + pd.DateOffset(months=1),
        periods=12,
        freq='MS'
    )
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': forecast,
        'Lower_CI': conf_int[:, 0],
        'Upper_CI': conf_int[:, 1]
    })
    
    # 6. Визуализация прогноза
    print("6. Визуализация результатов...")
    plt.figure(figsize=(14, 8))
    
    # Исторические данные
    plt.plot(df.index, df['Passengers'], 
             label='Исторические данные', linewidth=2, color='blue')
    
    # Прогноз
    plt.plot(forecast_df['Date'], forecast_df['Forecast'], 
             label='Прогноз', linewidth=2, color='red')
    
    # Доверительный интервал
    plt.fill_between(forecast_df['Date'], 
                     forecast_df['Lower_CI'], 
                     forecast_df['Upper_CI'], 
                     color='red', alpha=0.2, label='95% Доверительный интервал')
    
    plt.title('Прогноз количества пассажиров с использованием pmdarima', fontsize=14)
    plt.xlabel('Дата')
    plt.ylabel('Количество пассажиров')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('data/pmdarima/pmdarima_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Сохранение результатов
    print("7. Сохранение результатов...")
    
    # Сохраняем прогноз
    forecast_df.to_csv('data/pmdarima/pmdarima_forecast_results.csv', index=False)
    
    # Сохраняем информацию о модели
    with open('model_info.txt', 'w') as f:
        f.write(f"Лучшая модель: {model}\n")
        f.write(f"Параметры: {model.order} {model.seasonal_order}\n")
        f.write(f"AIC: {model.aic()}\n")
    
    print("✓ Прогноз сохранен в 'pmdarima_forecast_results.csv'")
    print("✓ Информация о модели сохранена в 'model_info.txt'")
    print("✓ Графики сохранены как 'original_data.png' и 'pmdarima_forecast.png'")
    
    # 8. Вывод результатов прогноза
    print("\n8. Результаты прогноза на 12 месяцев:")
    print(forecast_df.round(1))
    
    return forecast_df

if __name__ == "__main__":
    results = main()