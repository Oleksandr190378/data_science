import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from get_id_terms import get_unique_search_terms_for_period, get_date_range
from get_data import get_data_from_db, load_data_for_term_and_date
from seasonal_factor import calculate_seasonal_factor


pd.set_option('future.no_silent_downcasting', True)


def rename_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Перейменовує стовпці DataFrame відповідно до наданого відображення.
    """
    return df.rename(columns=column_mapping)

    
def find_conversions(row: pd.Series, prev_avg_orders: float = 1) -> Tuple[Dict[str, int], float]:
    """
    Розраховує метрики конверсії для заданого рядка.
    """
    # Копіюємо оригінальну функцію без змін
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
    date_str = row['Reporting_Date']
    seasonal_factor = 1.0
    if date_str:
        seasonal_factor = calculate_seasonal_factor(date_str)
    

    # Розрахунок мінімального порогу замовлень на основі даних попереднього місяця
    if row['SFR'] <= 5000:
        min_order_threshold = max(1, prev_avg_orders * 0.77 * seasonal_factor)
    elif row['SFR'] <= 10000 and row['SFR'] > 5000:
        min_order_threshold = max(1, prev_avg_orders * 0.74 * seasonal_factor)
    elif row['SFR'] > 10000 and row['SFR'] <= 40000:
        min_order_threshold = max(1, prev_avg_orders * 0.7 * seasonal_factor)
    elif row['SFR'] > 40000 and row['SFR'] <= 100000:
        min_order_threshold = max(1, prev_avg_orders * 0.65 * seasonal_factor)    
    else:
        min_order_threshold = max(1, prev_avg_orders * 0.55 * seasonal_factor)
    # Дефолтні значення
    delta = 0.025
    start = 1
    
    # Визначення стартового значення на основі SFR та min_share
    if row['SFR'] <= 200 and min_share >= 1:
        start = 3
    elif 200 < row['SFR'] <= 2000 and min_share >= 1:
        start = 2
    
    # Визначення параметрів delta і start на основі SFR та min_share
    sfr_share_config = [
        # SFR range, min_share range, has_zero_share, delta, start
        ((0, 1700), (20, float('inf')), True, 0.005, 20),
        ((0, 1700), (8, 20), True, 0.005, 15),
        ((0, 1700), (2, 8), True, 0.005, 12),
        ((0, 1700), (1, 2), True, 0.005, 5),
        
        ((1700, 4000), (20, float('inf')), True, 0.01, 10),
        ((1700, 4000), (2, 20), True, 0.01, 6),
        ((1700, 4000), (1, 2), True, 0.01, 2),
        
        ((4000, 10000), (2, float('inf')), True, 0.01, 8),
        ((4000, 10000), (1, 2), True, 0.01, 2),
        
        ((10000, 40000), (2.5, float('inf')), True, 0.01, 2),
        
        ((40000, 100000), (9, float('inf')), True, 0.01, 2)
    ]
    
    # Спецкейс для високих значень min_share
    if any(share == 0 for share in conv_shares) and min_share >= 50:
        if row['SFR'] <= 10000:
            delta, start = 0.005, 30
        elif 10000 < row['SFR'] <= 30000:
            delta, start = 0.005, 20
        elif 30000 < row['SFR'] <= 70000:
            delta, start = 0.005, 10
    
    # Застосовуємо конфігурацію з таблиці пошуку, якщо можливо
    has_zero_share = any(share == 0 for share in conv_shares)
    for sfr_range, share_range, requires_zero, d, s in sfr_share_config:
        if (sfr_range[0] < row['SFR'] <= sfr_range[1] and
            share_range[0] <= min_share < share_range[1] and
            (requires_zero == has_zero_share)):
            delta, start = d, s
            break
    
    # Визначаємо параметри a та b за допомогою таблиці пошуку
    ab_lookup = [
        # SFR range, a, b
        ((0, 100), 2100, 95),
        ((100, 300), 1800, 90),
        ((300, 500), 1400, 90),
        ((500, 1000), 1100, 75),
        ((1000, 2000), 920, 70),
        ((2000, 2700), 750, 60),
        ((2700, 3500), 580, 60),
        ((3500, 5000), 460, 55),
        ((5000, 15000), 375, 50),
        ((15000, 35000), 260, 35),
        ((35000, 70000), 200, 30),
        ((70000, 100000), 150, 25),
        ((100000, 200000), 100, 20),
        ((200000, float('inf')), 50, 20)
    ]
    
    a, b = 220, 30  # Дефолтні значення
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
                    total_orders >= min_order_threshold and
                   (total_orders < min_order_threshold*8 if min_order_threshold > 50 else True)):
                    return {
                        'Total_Orders': total_orders,
                        'Other_Orders': other_orders,
                        'ASIN1_Orders': orders[0],
                        'ASIN2_Orders': orders[1],
                        'ASIN3_Orders': orders[2],
                        'SFR': row['SFR'] if 'SFR' in row else None
                    }, delta
        delta += 0.02
        
    # Якщо не знайдено валідних комбінацій
    return {
        'Total_Orders': 0,
        'Other_Orders': 0,
        'ASIN1_Orders': 0,
        'ASIN2_Orders': 0,
        'ASIN3_Orders': 0,
        'SFR': row['SFR'] if 'SFR' in row else None
    }, delta


def find_clicks(row: pd.Series, conv_row: pd.Series, prev_avg_clicks: float = 1) -> Tuple[Dict[str, int], float]:
    """
    Розраховує метрики кліків для заданого рядка на основі даних конверсії.
    """
    # Копіюємо оригінальну функцію без змін
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
    date_str = row['Reporting_Date']
    seasonal_factor = 1.0
    if date_str:
        seasonal_factor = calculate_seasonal_factor(date_str)
    # Розрахунок мінімального порогу кліків на основі даних попереднього місяця
    min_click_threshold = max(1, prev_avg_clicks * 0.65 * seasonal_factor) 
    
    # Визначення таблиці параметрів на основі SFR
    click_params = [
        # SFR range, a (max iterations), b (clicks to orders ratio), delta
        ((0, 100), 600, 4, 0.02),
        ((100, 500), 530, 3.8, 0.02),
        ((500, 1000), 460, 3.6, 0.02),
        ((1000, 5000), 360, 3.5, 0.025),
        ((5000, 20000), 280, 3.3, 0.025),
        ((20000, 50000), 190, 2.7, 0.025),
        ((50000, 100000), 120, 2.0, 0.025),
        ((100000, 200000), 80, 1.5, 0.025),
        ((200000, float('inf')), 60, 1, 0.025)
    ]
    
    # Отримання параметрів з таблиці пошуку
    a, b, delta = 60, 1, 0.025  # Дефолтні значення
    for sfr_range, a_val, b_val, d_val in click_params:
        if sfr_range[0] < row['SFR'] <= sfr_range[1]:
            a, b, delta = int(a_val * seasonal_factor), b_val, d_val
            break
    
    # Пошук валідних комбінацій кліків
    while delta <= 3:
        for i in range(1, a):
            predicted_clicks = [ratio * i for ratio in click_ratios]
            
            if all(abs(click - round(click)) < delta for click in predicted_clicks if click > 0):
                clicks = [round(click) for click in predicted_clicks]
                sum_ratio = sum(click_shares)
                other_clicks = round((100-sum_ratio)*sum(clicks)/sum_ratio) if sum_ratio > 0 else 0
                total_clicks = sum(clicks) + other_clicks
                
                if (total_clicks % 1000 != 0 and
                    total_clicks >= min_click_threshold and
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
            
    # Якщо не знайдено валідних комбінацій
    return {
        'Total_Clicks': 1,
        'Other_Clicks': 1,
        'ASIN1_Clicks': 0,
        'ASIN2_Clicks': 0,
        'ASIN3_Clicks': 0
    }, delta


def process_search_term(
    search_term: int,
    all_dfs: List[pd.DataFrame],
    dates: List[datetime],
    previous_month_data: pd.DataFrame,
    column_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Обробляє один пошуковий термін по всіх DataFrame і повертає результати.
    """
    # Отримуємо дані за попередній місяць для цього пошукового терміну
    prev_data_row = previous_month_data[previous_month_data['Search_Term'] == search_term]
    prev_avg_orders = prev_data_row['Average_Orders'].iloc[0] if not prev_data_row.empty else 1
    prev_avg_clicks = prev_data_row['Average_Clicks'].iloc[0] if not prev_data_row.empty else 1
    
    conversion_results = []
    click_results = []
    
    for df, date in zip(all_dfs, dates):
        # Перейменовуємо стовпці перед обробкою
        df_mapped = rename_columns(df, column_mapping)
        
        # Шукаємо рядок для поточного пошукового терміну
        row = df_mapped[df_mapped['Search_Term'] == search_term]
        
        if row.empty:
            conv_result = {
                'Total_Orders': 0,
                'Other_Orders': 0,
                'ASIN1_Orders': 0,
                'ASIN2_Orders': 0,
                'ASIN3_Orders': 0,
                'SFR': 0
            }
            click_result = {
                'Total_Clicks': 0,
                'Other_Clicks': 0,
                'ASIN1_Clicks': 0,
                'ASIN2_Clicks': 0,
                'ASIN3_Clicks': 0
            }
        else:
            conv_result, _ = find_conversions(row.iloc[0], prev_avg_orders)
            click_result, _ = find_clicks(row.iloc[0], pd.Series(conv_result), prev_avg_clicks)
        
        conversion_results.append(conv_result)
        click_results.append(click_result)
    
    conv_df = pd.DataFrame(conversion_results, index=dates)
    click_df = pd.DataFrame(click_results, index=dates)
    conv_df['Search_Term'] = search_term
    result_df = pd.concat([conv_df, click_df], axis=1)
    cols = ['Search_Term', 'SFR'] + [col for col in result_df.columns if col not in ['Search_Term', 'SFR']]
    result_df = result_df[cols]
    
    return result_df


def analyze_search_terms(
    engine,
    start_date: str,
    end_date: str,
    columns_to_import: List[str],
    column_mapping: Dict[str, str],
    previous_month_data: Optional[pd.DataFrame] = None,
    search_list: Optional[List[int]] = None,  # Змінюємо тип на List[int]
    chunk_size: int = 10000
) -> List[pd.DataFrame]:
    """
    Головна функція для аналізу пошукових термінів з бази даних.
    
    :param engine: SQLAlchemy engine
    :param start_date: початкова дата у форматі 'YYYY-MM-DD'
    :param end_date: кінцева дата у форматі 'YYYY-MM-DD'
    :param columns_to_import: список стовпців для вибірки
    :param column_mapping: словник для перейменування стовпців
    :param previous_month_data: DataFrame з даними за попередній місяць
    :param search_list: список id_amz_search_term (цілі числа)
    :param chunk_size: розмір чанка для зчитування
    :return: список DataFrame з результатами обчислень
    """
    # Якщо не передано дані попереднього місяця, використовуємо порожній DataFrame
    if previous_month_data is None:
        previous_month_data = pd.DataFrame(columns=["Search_Term", "Average_Orders", "Average_Clicks"])
    
    # Генеруємо список дат для аналізу
    dates = pd.date_range(start=start_date, end=end_date)
    date_strings = [date.strftime('%Y-%m-%d') for date in dates]
    
    # Якщо search_list не надано, отримуємо унікальні id_amz_search_term за вказаний період
    if search_list is None:
        search_list = get_unique_search_terms_for_period(
            engine=engine,
            start_date=start_date,
            end_date=end_date,
            chunk_size=chunk_size
        )
    
    # Переконуємося, що всі елементи search_list є цілими числами
    if hasattr(search_list, 'tolist'):
        search_list = search_list.tolist()
    
    search_list = [int(term) for term in search_list]
    
    # Завантажуємо дані для кожної дати
    all_dfs = []
    
    for date_str in date_strings:
        # Завантажуємо дані для поточної дати
        df = load_data_for_term_and_date(
            engine=engine,
            columns_to_import=columns_to_import,
            search_terms=search_list,
            date=date_str,
            chunk_size=chunk_size
        )
        all_dfs.append(df)
    
    # Обробляємо кожен пошуковий термін
    result_dfs = []
    for search_term in search_list:
        result_df = process_search_term(
            search_term=int(search_term),  # Переконуємося, що передаємо ціле число
            all_dfs=all_dfs, 
            dates=dates,
            previous_month_data=previous_month_data,
            column_mapping=column_mapping
        )
        result_dfs.append(result_df)
    
    return result_dfs
    

