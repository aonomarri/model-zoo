from sklearn import metrics
import numpy as np

def print_metrics(y, y_predict):
    """
    Выводит основные метрики качества регрессионной модели.
    
    Параметры:
        y (array-like): Истинные значения целевой переменной
        y_predict (array-like): Предсказанные значения модели
        
    Выводит:
        - R² (коэффициент детерминации)
        - MAE (средняя абсолютная ошибка)
        - MAPE (средняя абсолютная процентная ошибка)
    """
    # Коэффициент детерминации (R²) - чем ближе к 1, тем лучше
    print('Train R^2: {:.3f}'.format(metrics.r2_score(y, y_predict)))
    
    # Средняя абсолютная ошибка (в единицах целевой переменной)
    print('Train MAE: {:.3f}'.format(metrics.mean_absolute_error(y, y_predict)))
    
    # Средняя абсолютная процентная ошибка (в процентах)
    print('Train MAPE: {:.1f}'.format(metrics.mean_absolute_percentage_error(y, y_predict)*100))
    

def outliers_z_score(data, feature, log_scale=False):
    """
    Обнаружение и удаление выбросов методом z-оценки (правило 3 sigm).
    
    Параметры:
        data (DataFrame): Исходный датафрейм
        feature (str): Название столбца для анализа
        log_scale (bool): Применять логарифмическое преобразование (по умолчанию False)
        
    Возвращает:
        tuple: (outliers - датафрейм с выбросами, cleaned - очищенные данные)
    """
    # Применяем логарифмическое преобразование если нужно
    if log_scale:
        x = np.log(data[feature]+1)  # +1 чтобы избежать log(0)
    else:
        x = data[feature]
    
    # Вычисляем среднее и стандартное отклонение
    mu = x.mean()
    sigma = x.std()
    
    # Устанавливаем границы для выбросов (±3σ)
    lower_bound = mu - 3 * sigma
    upper_bound = mu + 3 * sigma
    
    # Выделяем выбросы и очищенные данные
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x > lower_bound) & (x < upper_bound)]
    
    return outliers, cleaned