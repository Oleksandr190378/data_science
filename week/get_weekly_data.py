import pandas as pd
import numpy as np
from sqlalchemy import text
from typing import List
from datetime import datetime, timedelta
from seasonal_factor import calculate_seasonal_factor


def get_daily_data(
    engine,
    search_terms: List[int],
    start_date: str,
    end_date: str,
    chunk_size: int = 10000
) -> pd.DataFrame:
    """
    Отримує агреговані дані з таблиці ad_amz_search_term_daily_data
    для списку пошукових термінів за вказаний тиждень.
    
    Фокусується тільки на id_amz_search_term, total_clicks, total_orders
    для розрахунку сумарних значень за тиждень.
    
    Якщо для термінів немає даних за деякі дні тижня, 
    використовує середнє арифметичне за наявні дні.
    
    :param engine: SQLAlchemy engine
    :param search_terms: список id_amz_search_term
    :param start_date: початкова дата тижня у форматі 'YYYY-MM-DD'
    :param end_date: кінцева дата тижня у форматі 'YYYY-MM-DD'
    :param chunk_size: розмір чанка для зчитування
    :return: DataFrame з агрегованими даними за тиждень
    """
    # Конвертуємо дати в datetime об'єкти для подальших операцій
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Кількість днів у тижні (зазвичай 7, але може бути менше для неповних тижнів)
    days_in_period = (end_date_obj - start_date_obj).days + 1
    
    # Формуємо список всіх дат у цьому тижні
    dates_in_period = [(start_date_obj + timedelta(days=i)).strftime('%Y-%m-%d') 
                      for i in range(days_in_period)]
    
    # Конвертуємо numpy array в список, якщо потрібно
    if hasattr(search_terms, 'tolist'):
        search_terms = search_terms.tolist()
    
    # Переконуємося, що всі елементи search_terms є цілими числами
    search_terms = [int(term) for term in search_terms]
    
    # Формуємо запит для отримання щоденних даних з таблиці
    # Фокусуємося тільки на необхідних полях: id_amz_search_term, total_clicks, total_orders
    base_query = """
    SELECT 
        id_amz_search_term as "Search_Term",
        date,
        total_clicks,
        total_orders
    FROM ad_amz_search_term_daily_data
    WHERE date BETWEEN :start_date AND :end_date
    AND id_amz_marketplace = 2
    AND id_amz_search_term IN :search_terms
    ORDER BY id_amz_search_term
    """
    
    # Додаємо ліміт та офсет для пагінації
    paginated_query = base_query + " LIMIT :limit OFFSET :offset"
    
    # Завантажуємо дані чанками
    chunks = []
    offset = 0
    
    while True:
        # Виконуємо запит для поточного чанку
        with engine.connect() as connection:
            # Перевіряємо, чи маємо один елемент чи багато
            terms_param = tuple(search_terms) if len(search_terms) > 1 else (search_terms[0],)
            
            query = text(paginated_query).bindparams(
                start_date=start_date,
                end_date=end_date,
                search_terms=terms_param,
                limit=chunk_size,
                offset=offset
            )
            chunk = pd.read_sql(query, connection)
        
        # Якщо чанк порожній, виходимо з циклу
        if len(chunk) == 0:
            break
        
        chunks.append(chunk)
        offset += chunk_size
        
        # Якщо отримали менше рядків, ніж розмір чанку, значить дані закінчились
        if len(chunk) < chunk_size:
            break
    
    # Якщо немає даних, повертаємо порожній DataFrame
    if not chunks:
        print("Не знайдено щоденних даних для вказаних пошукових термінів за цей період")
        return pd.DataFrame(columns=[
            "Search_Term", "Total_Clicks_Daily", "Total_Orders_Daily"
        ])
    
    # Об'єднуємо всі чанки
    daily_data = pd.concat(chunks, ignore_index=True)
    
    # Результати для кожного пошукового терміну
    weekly_results = []
    
    # Обробляємо кожний пошуковий термін окремо
    for term in search_terms:
        term_data = daily_data[daily_data["Search_Term"] == term]
        
        # Якщо для цього терміну немає даних, пропускаємо його
        if term_data.empty:
            continue
        
        # Перевіряємо, чи є дані за всі дні тижня
        days_with_data = term_data["date"].nunique()
        
        # Розрахунок сумарних кліків та замовлень
        total_clicks_sum = term_data["total_clicks"].sum()
        total_orders_sum = term_data["total_orders"].sum()
        
        # Якщо немає даних за деякі дні, заповнюємо середніми значеннями
        if days_with_data < days_in_period:
            # Розраховуємо середні значення за доступні дні
            avg_clicks = round(term_data["total_clicks"].median())
            avg_orders = round(term_data["total_orders"].median())
            
            # Додаємо середні значення для відсутніх днів
            missing_days = days_in_period - days_with_data
            total_clicks_sum += avg_clicks * missing_days
            total_orders_sum += avg_orders * missing_days
        
        # Усереднений SFR за тиждень
        #avg_sfr = term_data["SFR"].mean()
        
        # Створюємо рядок з результатами для цього терміну
        result_row = {
            "Search_Term": term,
            "Total_Clicks_Daily": total_clicks_sum,  # Сума щоденних кліків
            "Total_Orders_Daily": total_orders_sum   # Сума щоденних замовлень
        }
        
        weekly_results.append(result_row)
    
    # Створюємо DataFrame з результатами
    result_df = pd.DataFrame(weekly_results)
    
    # Якщо немає результатів, повертаємо порожній DataFrame з потрібними стовпцями
    if result_df.empty:
        return pd.DataFrame(columns=[
            "Search_Term", "Total_Clicks_Daily", "Total_Orders_Daily"
        ])
    
    return result_df


def check_daily_data_availability(
    engine,
    search_terms: List[int],
    start_date: str,
    end_date: str
) -> bool:
    """
    Перевіряє, чи є щоденні дані (конверсії та кліки) у таблиці ad_amz_search_term_daily_data
    для вказаних пошукових термінів за вказаний період.
    
    :param engine: SQLAlchemy engine
    :param search_terms: список id_amz_search_term
    :param start_date: початкова дата у форматі 'YYYY-MM-DD'
    :param end_date: кінцева дата у форматі 'YYYY-MM-DD'
    :return: True, якщо дані доступні, False - в іншому випадку
    """
    # Конвертуємо numpy array в список, якщо потрібно
    if hasattr(search_terms, 'tolist'):
        search_terms = search_terms.tolist()
    
    # Переконуємося, що всі елементи search_terms є цілими числами
    search_terms = [int(term) for term in search_terms]
    
    # Формуємо запит для перевірки наявності даних
    query = text("""
    SELECT COUNT(*) as data_count
    FROM ad_amz_search_term_daily_data
    WHERE date BETWEEN :start_date AND :end_date
    AND id_amz_marketplace = 2
    AND id_amz_search_term IN :search_terms
    AND total_clicks IS NOT NULL
    AND total_orders IS NOT NULL
    """)
    
    # Виконуємо запит
    with engine.connect() as connection:
        # Перевіряємо, чи маємо один елемент чи багато
        terms_param = tuple(search_terms) if len(search_terms) > 1 else (search_terms[0],)
        
        result = connection.execute(
            query,
            {
                "start_date": start_date,
                "end_date": end_date,
                "search_terms": terms_param
            }
        )
        
        data_count = result.fetchone()[0]
    
    # Якщо є хоча б один запис з даними, повертаємо True
    return data_count > 0


def get_weekly_seasonal_factor(start_date, end_date):
    """
    Розраховує середній сезонний фактор для тижня.
    
    :param start_date: початкова дата тижня
    :param end_date: кінцева дата тижня
    :return: середній сезонний фактор
    """
    # Перетворюємо дати в об'єкти datetime, якщо вони не є ними
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Розраховуємо сезонний фактор для кожного дня тижня
    current_date = start_date
    total_factor = 0
    days_count = 0
    
    while current_date <= end_date:
        factor = calculate_seasonal_factor(current_date)
        total_factor += factor
        days_count += 1
        current_date = current_date + timedelta(days=1)
    
    # Обчислюємо середній фактор
    avg_factor = total_factor / days_count if days_count > 0 else 1.0
    
    return avg_factor

