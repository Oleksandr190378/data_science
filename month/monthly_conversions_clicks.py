import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from .get_monthly_data import get_daily_data, get_monthly_seasonal_factor
from .get_monthly_shares import get_monthly_shares


def analyze_monthly_search_terms(
    engine,
    start_date: str,
    end_date: str,
    month_number: int,
    year: int,
    search_list: Optional[List[int]] = None
) -> List[pd.DataFrame]:
    """
    Головна функція для аналізу місячних даних пошукових термінів.
    
    :param engine: SQLAlchemy engine
    :param start_date: початкова дата місяця у форматі 'YYYY-MM-DD'
    :param end_date: кінцева дата місяця у форматі 'YYYY-MM-DD'
    :param month_number: номер місяця
    :param year: рік
    :param search_list: список id_amz_search_term (цілі числа)
    :return: список DataFrame з результатами обчислень
    """
    # Отримуємо агреговані щоденні дані для пошукових запитів
    daily_data = get_daily_data(
        engine=engine,
        search_terms=search_list,
        start_date=start_date,
        end_date=end_date
    )
    
    # Якщо немає даних, повертаємо порожній список
    if daily_data.empty:
        return []
    
    # Отримуємо частки кліків та конверсій з місячної таблиці
    monthly_shares = get_monthly_shares(
        engine=engine,
        search_terms=daily_data['Search_Term'].tolist(),
        month_number=month_number,
        year=year
    )
    
    # Отримуємо сезонний фактор для місяця
    seasonal_factor = get_monthly_seasonal_factor(start_date, end_date)
    
    # Додаємо сезонний фактор до даних
    daily_data['Seasonal_Factor'] = seasonal_factor
    
    # Поєднуємо щоденні дані з частками кліків та конверсій
    combined_data = pd.merge(
        daily_data,
        monthly_shares,
        on='Search_Term',
        how='left'
    )
    
    # Заповнюємо пропущені значення нулями
    combined_data.fillna(0, inplace=True)
    
    # Для кожного пошукового терміну обчислюємо конверсії та кліки
    result_dfs = []
    
    for _, row in combined_data.iterrows():
        # Розрахунок конверсій
        conversion_result, _ = find_monthly_conversions(
            row=row, 
            daily_total_orders=row['Total_Orders_Daily'] if 'Total_Orders_Daily' in row else 0
        )
        
        # Розрахунок кліків на основі конверсій
        click_result, _ = find_monthly_clicks(
            row=row, 
            conv_row=pd.Series(conversion_result),
            daily_total_clicks=row['Total_Clicks_Daily'] if 'Total_Clicks_Daily' in row else 0
        )
        
        # Створюємо DataFrame з результатами
        result_df = pd.DataFrame({
            'Search_Term': [row['Search_Term']],
            'SFR': [row['SFR']],
            'Total_Orders': [conversion_result['Total_Orders']],
            'Other_Orders': [conversion_result['Other_Orders']],
            'ASIN1_Orders': [conversion_result['ASIN1_Orders']],
            'ASIN2_Orders': [conversion_result['ASIN2_Orders']],
            'ASIN3_Orders': [conversion_result['ASIN3_Orders']],
            'Total_Clicks': [click_result['Total_Clicks']],
            'Other_Clicks': [click_result['Other_Clicks']],
            'ASIN1_Clicks': [click_result['ASIN1_Clicks']],
            'ASIN2_Clicks': [click_result['ASIN2_Clicks']],
            'ASIN3_Clicks': [click_result['ASIN3_Clicks']]
        })
        
        result_dfs.append(result_df)
    
    return result_dfs


def find_monthly_conversions(row: pd.Series, daily_total_orders: float = 0) -> Tuple[Dict[str, int], float]:
    """
    Розраховує метрики конверсій для місяця на основі рядка даних.
    Коефіцієнти збільшені в 30 разів у порівнянні з щоденними даними.
    
    :param row: рядок даних з часткою конверсій
    :param daily_total_orders: сума щоденних конверсій за місяць (для обмеження)
    :return: словник з кількістю конверсій, delta
    """
    row = row.copy()
    conversion_cols = ['Conversion_Share_1', 'Conversion_Share_2', 'Conversion_Share_3']
    row.loc[conversion_cols] = row.loc[conversion_cols].fillna(0)
    row.loc[conversion_cols] = pd.to_numeric(row.loc[conversion_cols], errors='coerce')
        
    # Перевіряємо, чи всі значення конверсії нульові
    if (row['Conversion_Share_1'] == 0 and 
        row['Conversion_Share_2'] == 0 and 
        row['Conversion_Share_3'] == 0):
        return {
            'Total_Orders': 0,
            'Other_Orders': 0,
            'ASIN1_Orders': 0,
            'ASIN2_Orders': 0,
            'ASIN3_Orders': 0, 
            'SFR': row['SFR'] if 'SFR' in row else None
        }, 0

    # Розрахунок часток конверсії та співвідношень
    conv_shares = [row['Conversion_Share_1'], 
                  row['Conversion_Share_2'], 
                  row['Conversion_Share_3']]
    min_share = min(share for share in conv_shares if share > 0)
    ratios = [share/min_share if share > 0 else 0 for share in conv_shares]
    
    # Отримуємо сезонний фактор (передбачається, що він вже обчислений для місяця)
    seasonal_factor = 1.0
    if 'Seasonal_Factor' in row and row['Seasonal_Factor'] > 0:
        seasonal_factor = row['Seasonal_Factor']

    # Розрахунок мінімального порогу замовлень на основі даних за місяць
    # Використовуємо щоденні дані, помножені на коефіцієнт, як обмеження
    min_order_threshold = daily_total_orders * 0.65 
    max_order_threshold = daily_total_orders * 1.9 
    
    # Дефолтні значення для пошуку
    delta = 0.025
    start = 30  # Збільшуємо стартове значення в порівнянні з щоденним
    
    # Визначення параметрів delta і start на основі SFR та min_share
    # Коефіцієнти збільшені в 30 разів у порівнянні з щоденними даними
    sfr_share_config = [
        # SFR range, min_share range, has_zero_share, delta, start
        ((0, 1700), (20, float('inf')), True, 0.005, 500),
        ((0, 1700), (8, 20), True, 0.005, 400),
        ((0, 1700), (2, 8), True, 0.005, 350),
        ((0, 1700), (1, 2), True, 0.005, 150),
        
        ((1700, 4000), (20, float('inf')), True, 0.01, 300),
        ((1700, 4000), (2, 20), True, 0.01, 170),
        ((1700, 4000), (1, 2), True, 0.01, 60),
        
        ((4000, 10000), (2, float('inf')), True, 0.01, 210),
        ((4000, 10000), (1, 2), True, 0.01, 60),
        
        ((10000, 40000), (2.5, float('inf')), True, 0.01, 60),
        
        ((40000, 100000), (9, float('inf')), True, 0.01, 60)
    ]
    
    # Спецкейс для високих значень min_share
    if any(share == 0 for share in conv_shares) and min_share >= 50:
        if row['SFR'] <= 10000:
            delta, start = 0.005, 900
        elif 10000 < row['SFR'] <= 30000:
            delta, start = 0.005, 600
        elif 30000 < row['SFR'] <= 70000:
            delta, start = 0.005, 300
    
    # Застосовуємо конфігурацію з таблиці пошуку, якщо можливо
    has_zero_share = any(share == 0 for share in conv_shares)
    for sfr_range, share_range, requires_zero, d, s in sfr_share_config:
        if (sfr_range[0] < row['SFR'] <= sfr_range[1] and
            share_range[0] <= min_share < share_range[1] and
            (requires_zero == has_zero_share)):
            delta, start = d, s
            break
    
    # Визначаємо параметри a та b за допомогою таблиці пошуку
    # Коефіцієнти збільшені в 30 разів у порівнянні з щоденними даними
    ab_lookup = [
        # SFR range, a, b
        ((0, 100), 63000, 2550),
        ((100, 300), 54000, 2400),
        ((300, 500), 43500, 2400),
        ((500, 1000), 34500, 1950),
        ((1000, 2000), 28500, 1800),
        ((2000, 2700), 23400, 1500),
        ((2700, 3500), 18000, 1500),
        ((3500, 5000), 14100, 1350),
        ((5000, 15000), 11850, 1200),
        ((15000, 35000), 8100, 1050),
        ((35000, 70000), 6600, 900),
        ((70000, 100000), 4800, 750),
        ((100000, 200000), 3000, 600),
        ((200000, float('inf')), 1500, 600)
    ]
    
    a, b = 6600, 900  # Дефолтні значення в 30 разів більші
    for sfr_range, a_val, b_val in ab_lookup:
        if sfr_range[0] < row['SFR'] <= sfr_range[1]:
            a, b = int(a_val * seasonal_factor), int(b_val * seasonal_factor)
            break
            
    # Пошук валідних комбінацій замовлень
    while delta <= 3:
        for i in range(start, b):
            predicted = [ratio * i for ratio in ratios]
            
            if all(abs(pred - round(pred)) < delta for pred in predicted if pred > 0):
                orders = [round(pred) for pred in predicted]
                sum_ratio = sum(conv_shares)
                other_orders = round((100-sum_ratio)*sum(orders)/sum_ratio) if sum_ratio > 0 else 0
                total_orders = sum(orders) + other_orders
                
                if total_orders > a and delta < 1.5:
                    break
                    
                if (total_orders % 1000 != 0 and 
                    (min_order_threshold <= total_orders <= max_order_threshold)):
                    return {
                        'Total_Orders': total_orders,
                        'Other_Orders': other_orders,
                        'ASIN1_Orders': orders[0],
                        'ASIN2_Orders': orders[1],
                        'ASIN3_Orders': orders[2],
                        'SFR': row['SFR'] if 'SFR' in row else None
                    }, delta
        delta += 0.02
        if delta > 2.5:
            min_order_threshold = daily_total_orders * 0.5 
            max_order_threshold = daily_total_orders * 2.3 

        
    # Якщо немає підібраних даних, повертаємо нулі 
    return {
        'Total_Orders': 0,
        'Other_Orders': 0,
        'ASIN1_Orders': 0,
        'ASIN2_Orders': 0,
        'ASIN3_Orders': 0,
        'SFR': row['SFR'] if 'SFR' in row else None
    }, delta


def find_monthly_clicks(row: pd.Series, conv_row: pd.Series, daily_total_clicks: float = 0) -> Tuple[Dict[str, int], float]:
    """
    Розраховує метрики кліків для місяця на основі рядка даних та конверсій.
    Коефіцієнти збільшені в 30 разів у порівнянні з щоденними даними.
    
    :param row: рядок даних з часткою кліків
    :param conv_row: рядок даних з розрахованими конверсіями
    :param daily_total_clicks: сума щоденних кліків за місяць (для обмеження)
    :return: словник з кількістю кліків, delta
    """
    click_shares = [
        row['Click_Share_1'], 
        row['Click_Share_2'], 
        row['Click_Share_3']
    ]
    
    # Обробка випадку, коли всі частки кліків дорівнюють нулю
    if all(share == 0 for share in click_shares):
        return {
            'Total_Clicks': 1,
            'Other_Clicks': 1,
            'ASIN1_Clicks': 0,
            'ASIN2_Clicks': 0,
            'ASIN3_Clicks': 0
        }, 0
        
    min_click_share = min(share for share in click_shares if share > 0)
    click_ratios = [share/min_click_share if share > 0 else 0 for share in click_shares]
    
    # Отримуємо сезонний фактор (передбачається, що він вже обчислений для місяця)
    seasonal_factor = 1.0
    if 'Seasonal_Factor' in row and row['Seasonal_Factor'] > 0:
        seasonal_factor = row['Seasonal_Factor']
    
    # Розрахунок мінімального порогу кліків на основі даних за місяць
    # Використовуємо щоденні дані, помножені на коефіцієнт, як обмеження
    min_click_threshold = daily_total_clicks * 0.65 
    max_click_threshold = daily_total_clicks * 2.0 
    
    # Визначення таблиці параметрів на основі SFR
    # Коефіцієнти збільшені в 30 разів у порівнянні з щоденними даними
    click_params = [
        # SFR range, a (max iterations), b (clicks to orders ratio), delta
        ((0, 100), 15000, 4, 0.02),
        ((100, 500), 12900, 3.8, 0.02),
        ((500, 1000), 10800, 3.6, 0.02),
        ((1000, 5000), 7800, 3.5, 0.025),
        ((5000, 20000), 5400, 3.3, 0.025),
        ((20000, 50000), 3900, 2.7, 0.025),
        ((50000, 100000), 2400, 2.0, 0.025),
        ((100000, 200000), 1800, 1.5, 0.025),
        ((200000, float('inf')), 1800, 1, 0.025)
    ]
    
    # Отримання параметрів з таблиці пошуку
    a, b, delta = 1800, 1, 0.025  # Дефолтні значення в 30 разів більші
    for sfr_range, a_val, b_val, d_val in click_params:
        if sfr_range[0] < row['SFR'] <= sfr_range[1]:
            a, b, delta = int(a_val * seasonal_factor), b_val, d_val
            break
    
    # Пошук валідних комбінацій кліків
    while delta <= 3.6:
        for i in range(1, a):
            predicted_clicks = [ratio * i for ratio in click_ratios]
            
            if all(abs(click - round(click)) < delta for click in predicted_clicks if click > 0):
                clicks = [round(click) for click in predicted_clicks]
                sum_ratio = sum(click_shares)
                other_clicks = round((100-sum_ratio)*sum(clicks)/sum_ratio) if sum_ratio > 0 else 0
                total_clicks = sum(clicks) + other_clicks
                
                if (total_clicks % 1000 != 0 and
                    min_click_threshold <= total_clicks <= max_click_threshold and
                    total_clicks >= conv_row['Total_Orders'] * b and
                    other_clicks >= conv_row['Other_Orders'] and
                    all(c >= o for c, o in zip(clicks, [
                        conv_row['ASIN1_Orders'],
                        conv_row['ASIN2_Orders'],
                        conv_row['ASIN3_Orders']
                    ]))):
                    
                    return {
                        'Total_Clicks': total_clicks,
                        'Other_Clicks': other_clicks,
                        'ASIN1_Clicks': clicks[0],
                        'ASIN2_Clicks': clicks[1],
                        'ASIN3_Clicks': clicks[2]
                    }, delta
        delta += 0.02
        if delta > 2.8:
            a += 50        
            max_click_threshold = daily_total_clicks * 2.5 * seasonal_factor    
    # Якщо не знайдено валідних комбінацій
    return {
        'Total_Clicks': 1,
        'Other_Clicks': 1,
        'ASIN1_Clicks': 0,
        'ASIN2_Clicks': 0,
        'ASIN3_Clicks': 0
    }, delta

