import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    def __init__(self, csv_path, output_dir='data/statsmodels'):
        self.data = self.load_data(csv_path)
        self.output_dir = output_dir
        self.train_data = None
        self.test_data = None
        self.model = None
        self.forecast = None
        
        # Создаем директорию для результатов, если она не существует
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self, csv_path):
        """Загрузка и подготовка данных"""
        df = pd.read_csv(csv_path)
        df['Month'] = pd.to_datetime(df['Month'])
        df.set_index('Month', inplace=True)
        df.columns = ['Passengers']
        return df
    
    def get_output_path(self, filename):
        """Получение полного пути к файлу в директории результатов"""
        return os.path.join(self.output_dir, filename)
    
    def explore_data(self):
        """Исследование временного ряда"""
        print("Информация о данных:")
        print(self.data.info())
        print("\nПервые 5 строк:")
        print(self.data.head())
        print("\nСтатистика:")
        print(self.data.describe())
        
        # Сохраняем основную статистику в файл
        stats_file = self.get_output_path('data_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("Статистика данных:\n")
            f.write(str(self.data.describe()))
            f.write(f"\n\nКоличество наблюдений: {len(self.data)}")
            f.write(f"\nПериод: {self.data.index[0]} - {self.data.index[-1]}")
        
        # Визуализация
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Passengers'])
        plt.title('Количество пассажиров авиакомпании (1949-1960)')
        plt.xlabel('Дата')
        plt.ylabel('Пассажиры')
        plt.grid(True)
        plt.savefig(self.get_output_path('data_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Декомпозиция временного ряда
        try:
            decomposition = seasonal_decompose(self.data['Passengers'], model='multiplicative', period=12)
            
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
            decomposition.observed.plot(ax=ax1)
            ax1.set_title('Исходный ряд')
            decomposition.trend.plot(ax=ax2)
            ax2.set_title('Тренд')
            decomposition.seasonal.plot(ax=ax3)
            ax3.set_title('Сезонность')
            decomposition.resid.plot(ax=ax4)
            ax4.set_title('Остатки')
            plt.tight_layout()
            plt.savefig(self.get_output_path('decomposition.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Ошибка при декомпозиции: {e}")
    
    def prepare_train_test(self, test_size=12):
        """Разделение на обучающую и тестовую выборки"""
        self.train_data = self.data.iloc[:-test_size]
        self.test_data = self.data.iloc[-test_size:]
        print(f"Обучающая выборка: {len(self.train_data)} месяцев")
        print(f"Тестовая выборка: {len(self.test_data)} месяцев")
        
        # Сохраняем разделенные данные
        self.train_data.to_csv(self.get_output_path('train_data.csv'))
        self.test_data.to_csv(self.get_output_path('test_data.csv'))
    
    def fit_arima(self, order=(2,1,2), seasonal_order=(1,1,1,12)):
        """Обучение SARIMA модели"""
        try:
            # SARIMA модель с сезонностью
            self.model = SARIMAX(
                self.train_data['Passengers'],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.model_fit = self.model.fit(disp=False)
            print("Модель SARIMA успешно обучена")
            
            # Сохраняем summary модели
            summary_file = self.get_output_path(f'model_summary_arima{order}.txt')
            with open(summary_file, 'w') as f:
                f.write(self.model_fit.summary().as_text())
            
        except Exception as e:
            print(f"Ошибка при обучении SARIMA: {e}")
            
            # Попробуем простую ARIMA модель
            try:
                self.model = ARIMA(
                    self.train_data['Passengers'],
                    order=order
                )
                self.model_fit = self.model.fit()
                print("Модель ARIMA успешно обучена")
                
                # Сохраняем summary модели
                summary_file = self.get_output_path(f'model_summary_arima{order}.txt')
                with open(summary_file, 'w') as f:
                    f.write(self.model_fit.summary().as_text())
                    
            except Exception as e2:
                print(f"Ошибка при обучении ARIMA: {e2}")
                raise
    
    def make_forecast(self, steps=12):
        """Создание прогноза"""
        if self.model_fit is None:
            raise ValueError("Модель не обучена. Сначала вызовите fit_arima()")
        
        # Прогноз на будущее
        self.forecast = self.model_fit.get_forecast(steps=steps)
        forecast_mean = self.forecast.predicted_mean
        forecast_conf_int = self.forecast.conf_int()
        
        return forecast_mean, forecast_conf_int
    
    def evaluate_model(self):
        """Оценка качества модели"""
        if self.model_fit is None or self.test_data is None:
            raise ValueError("Модель не обучена или тестовые данные не подготовлены")
        
        # Прогноз на тестовой выборке
        test_forecast = self.model_fit.get_forecast(steps=len(self.test_data))
        test_pred = test_forecast.predicted_mean
        
        # Метрики качества
        mae = mean_absolute_error(self.test_data['Passengers'], test_pred)
        mse = mean_squared_error(self.test_data['Passengers'], test_pred)
        rmse = np.sqrt(mse)
        
        print(f"\nМетрики качества на тестовой выборке:")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        
        # Сохраняем метрики
        metrics_file = self.get_output_path('model_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"MAE: {mae:.2f}\n")
            f.write(f"MSE: {mse:.2f}\n")
            f.write(f"RMSE: {rmse:.2f}\n")
        
        return mae, mse, rmse
    
    def plot_results(self, forecast_mean, forecast_conf_int, order):
        """Визуализация результатов"""
        plt.figure(figsize=(15, 8))
        
        # Исходные данные
        plt.plot(self.data.index, self.data['Passengers'], label='Фактические данные', color='blue', linewidth=2)
        
        # Прогноз
        future_dates = pd.date_range(start=self.data.index[-1] + pd.DateOffset(months=1), 
                                   periods=len(forecast_mean), freq='M')
        plt.plot(future_dates, forecast_mean, label='Прогноз', color='red', linestyle='--', linewidth=2)
        
        # Доверительный интервал
        plt.fill_between(future_dates, 
                        forecast_conf_int.iloc[:, 0], 
                        forecast_conf_int.iloc[:, 1], 
                        color='pink', alpha=0.3, label='95% доверительный интервал')
        
        plt.title(f'Прогноз количества пассажиров авиакомпании\nARIMA{order}')
        plt.xlabel('Дата')
        plt.ylabel('Пассажиры')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.get_output_path('forecast_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_forecast_to_csv(self, forecast_mean, filename='forecast_results.csv'):
        """Сохранение прогноза в CSV"""
        forecast_df = pd.DataFrame({
            'Date': forecast_mean.index,
            'Forecast_Passengers': forecast_mean.values
        })
        
        output_path = self.get_output_path(filename)
        forecast_df.to_csv(output_path, index=False)
        print(f"Прогноз сохранен в {output_path}")
        
        return forecast_df

def main():
    # Путь к файлу данных
    csv_path = 'data/airline-passengers.csv'
    
    # Создание экземпляра прогнозировщика
    forecaster = TimeSeriesForecaster(csv_path, output_dir='data/statsmodels')
    
    print("=== ИССЛЕДОВАНИЕ ДАННЫХ ===")
    forecaster.explore_data()
    
    print("\n=== ПОДГОТОВКА ДАННЫХ ===")
    forecaster.prepare_train_test(test_size=12)
    
    print("\n=== ОБУЧЕНИЕ МОДЕЛЕЙ ===")
    
    # Параметры ARIMA для тестирования
    orders_to_try = [
        (2, 1, 2),         # Стандартные параметры
        (1, 1, 1),         # Более простая модель
        (3, 1, 3),         # Более сложная модель
        (2, 1, 1),         # Разные комбинации
        (1, 1, 2)
    ]
    
    best_rmse = float('inf')
    best_order = None
    best_forecast_mean = None
    best_forecast_conf_int = None
    
    for order in orders_to_try:
        try:
            print(f"\nПробуем параметры ARIMA{order}:")
            forecaster.fit_arima(order=order)
            mae, mse, rmse = forecaster.evaluate_model()
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_order = order
                # Сохраняем прогноз лучшей модели
                best_forecast_mean, best_forecast_conf_int = forecaster.make_forecast(steps=12)
                print(f"Новый лучший результат: RMSE = {rmse:.2f}")
                
        except Exception as e:
            print(f"Ошибка для параметров {order}: {e}")
            continue
    
    print(f"\nЛучшие параметры: ARIMA{best_order} с RMSE = {best_rmse:.2f}")
    
    # Переобучаем модель с лучшими параметрами для финальных результатов
    print(f"\nФинальное обучение с лучшими параметрами ARIMA{best_order}:")
    forecaster.fit_arima(order=best_order)
    
    # Создание финального прогноза
    print("\n=== СОЗДАНИЕ ПРОГНОЗА ===")
    forecast_mean, forecast_conf_int = forecaster.make_forecast(steps=12)
    
    # Визуализация результатов
    forecaster.plot_results(forecast_mean, forecast_conf_int, best_order)
    
    # Сохранение прогноза в CSV
    forecast_df = forecaster.save_forecast_to_csv(forecast_mean, 'final_forecast.csv')
    
    print("\n=== РЕЗУЛЬТАТЫ ===")
    print("Прогноз на следующие 12 месяцев:")
    print(forecast_df.to_string(index=False))
    
    # Сохраняем информацию о лучшей модели
    best_model_info = {
        'best_order': best_order,
        'best_rmse': best_rmse,
        'forecast_periods': 12
    }
    
    info_file = forecaster.get_output_path('model_info.txt')
    with open(info_file, 'w') as f:
        for key, value in best_model_info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nВсе результаты сохранены в папке: data/statsmodels/")
    print("Содержимое папки:")
    result_files = os.listdir('data/statsmodels')
    for file in result_files:
        print(f"  - {file}")
    
    return forecast_df

if __name__ == "__main__":
    main()