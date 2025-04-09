from datetime import datetime, date, timedelta
from typing import List
import pandas as pd
from sqlalchemy import  text
import numpy as np


def get_data_from_db(
    engine,
    columns_to_import: List[str],
    date: str = None,
    search_list: List[int] = None,  # Змінюємо тип на List[int]
    chunk_size: int = 10000
) -> pd.DataFrame:
    """
    Отримує дані з PostgreSQL бази даних з опціями фільтрації.
    
    :param engine: SQLAlchemy engine
    :param columns_to_import: список стовпців для вибірки
    :param date: дата у форматі 'YYYY-MM-DD'
    :param search_list: список id_amz_search_term (цілі числа)
    :param chunk_size: розмір чанка для зчитування
    :return: DataFrame з даними
    """
    # Створюємо базовий SQL запит
    base_query = """
    SELECT 
        id_amz_search_term,
        amz_search_frequency_rank,
        amz_top_1_item_name,
        amz_top_1_click_share,
        amz_top_1_conversion_share,
        amz_top_2_item_name,
        amz_top_2_click_share,
        amz_top_2_conversion_share,
        amz_top_3_item_name,
        amz_top_3_click_share,
        amz_top_3_conversion_share,
        date
    FROM ad_amz_search_term_daily_data
    WHERE id_amz_marketplace = 2
    """
    
    # Додаємо умови фільтрації
    params = {}
    
    # Фільтр за датою
    if date is not None:
        base_query += " AND date = :date"
        params['date'] = date
    
    # Фільтр за списком пошукових термінів (точне співпадіння)
    if search_list is not None and len(search_list) > 0:
        # Конвертуємо numpy array у список, якщо потрібно
        if hasattr(search_list, 'tolist'):
            search_list = search_list.tolist()
        
        # Переконуємося, що всі елементи списку є цілими числами
        search_list = [int(item) for item in search_list]
            
        if len(search_list) == 1:
            base_query += " AND id_amz_search_term = :term_list"
            params['term_list'] = search_list[0]  # Тут не потрібно додаткове конвертування
        else:
            base_query += " AND id_amz_search_term IN :term_list"
            params['term_list'] = tuple(search_list)  # Тут не потрібно додаткове конвертування
    
    # Додаємо ліміт та офсет для пагінації
    paginated_query = base_query + " LIMIT :limit OFFSET :offset"
    
    # Завантажуємо дані чанками
    chunks = []
    offset = 0
    
    while True:
        # Оновлюємо параметри для пагінації
        params_with_pagination = params.copy()
        params_with_pagination['limit'] = chunk_size
        params_with_pagination['offset'] = offset
        
        # Виконуємо запит для поточного чанку
        with engine.connect() as connection:
            query = text(paginated_query).bindparams(**params_with_pagination)
            chunk = pd.read_sql(query, connection)
        
        # Якщо чанк порожній, виходимо з циклу
        if len(chunk) == 0:
            break
        
        chunks.append(chunk)
        offset += chunk_size
        
        # Якщо отримали менше рядків, ніж розмір чанку, значить дані закінчились
        if len(chunk) < chunk_size:
            break
    
    # Об'єднуємо всі чанки в один DataFrame
    if not chunks:
        # Повертаємо порожній DataFrame з правильними стовпцями
        return pd.DataFrame(columns=columns_to_import)
    
    df = pd.concat(chunks, ignore_index=True)
    
    # Переконуємося, що id_amz_search_term має тип int
    if 'id_amz_search_term' in df.columns:
        df['id_amz_search_term'] = df['id_amz_search_term'].astype(int)
    
    # Перевіряємо, чи отримали всі необхідні стовпці
    for col in columns_to_import:
        if col not in df.columns:
            df[col] = np.nan  # Заповнюємо NaN відсутні стовпці
    
    return df


def load_data_for_term_and_date(
    engine,
    columns_to_import: List[str],
    search_terms: List[int],
    date: str,
    chunk_size: int = 10000
) -> pd.DataFrame:
    """
    Отримує дані з БД для конкретних пошукових термінів за вказану дату.
    Якщо термін за дату не знайдено, створює рядок з дефолтними значеннями.
    
    :param engine: SQLAlchemy engine
    :param columns_to_import: список стовпців для вибірки
    :param search_terms: список id_amz_search_term (цілі числа)
    :param date: дата у форматі 'YYYY-MM-DD'
    :param chunk_size: розмір чанка для зчитування
    :return: DataFrame з даними
    """
    # Отримуємо дані з БД
    df = get_data_from_db(
        engine=engine,
        columns_to_import=columns_to_import,
        date=date,
        search_list=search_terms,
        chunk_size=chunk_size
    )
    
    # Стовпці, які повинні мати значення 1 за замовчуванням
    top_click_columns = [
        'amz_top_1_click_share',
        'amz_top_2_click_share',
        'amz_top_3_click_share'
    ]
    
    # Ініціалізуємо список для результатів
    result_rows = []
    
    # Обробляємо кожен пошуковий термін
    for term in search_terms:
        # Переконуємося, що term є цілим числом
        term = int(term)
        
        # Шукаємо точне співпадіння
        matched_row = df[df['id_amz_search_term'] == term]
        
        if len(matched_row) > 0:
            # Якщо знайдено, додаємо перший співпадаючий рядок
            result_rows.append(matched_row.iloc[0])
        else:
            # Створюємо дефолтні значення для всіх стовпців
            default_values = {}
            default_values['id_amz_search_term'] = term
            default_values['date'] = pd.to_datetime(date)  # Додаємо дату
            
            # Встановлюємо дефолтні значення для інших стовпців
            for col in columns_to_import:
                if col == 'id_amz_search_term' or col == 'date':
                    continue
                elif col in top_click_columns:
                    default_values[col] = 1  # 1 для top clicked product columns
                else:
                    default_values[col] = 0  # 0 для всіх інших стовпців
            
            default_row = pd.Series(default_values)
            result_rows.append(default_row)
    
    # Об'єднуємо всі рядки у фінальний DataFrame
    result_df = pd.DataFrame(result_rows)
    
    return result_df
    

def load_previous_data(
    engine,
    search_terms: List[int],
    current_start_date: str,
    days_back: int = 7,
    chunk_size: int = 10000
) -> pd.DataFrame:
    """
    Завантажує та розраховує середні значення за попередній період для вказаних пошукових термінів.
    Враховує сезонні піки продажів.
    
    :param engine: SQLAlchemy engine для підключення до бази даних
    :param search_terms: список пошукових термінів для аналізу
    :param current_start_date: початкова дата поточного аналізу у форматі 'YYYY-MM-DD'
    :param days_back: кількість днів для аналізу назад від поточної дати
    :param chunk_size: розмір чанка для зчитування даних
    :return: DataFrame з середніми значеннями замовлень і кліків за попередній період
    """
    try:
        # Перетворюємо поточну дату початку в об'єкт datetime
        current_start_date_obj = datetime.strptime(current_start_date, '%Y-%m-%d')
        current_month = current_start_date_obj.month
        current_day = current_start_date_obj.day
        
        # Перевіряємо, чи поточна дата припадає на сезонний пік
        is_peak_season = False
        
        # Black Friday / Cyber Monday
        if (current_month == 11 and current_day >= 20 and current_day <= 30):
            is_peak_season = True
        # Різдвяний сезон
        elif (current_month == 12 and current_day <= 20):
            is_peak_season = True
        # Prime Day
        elif (current_month == 7 and 10 <= current_day <= 20):
            is_peak_season = True
        # Deal Days / October Prime Day
        elif (current_month == 10 and 10 <= current_day <= 15):
            is_peak_season = True
        
        # Створюємо порожній DataFrame для спеціальних періодів
        if (current_month == 1 and current_day <= 26) or (current_month == 12 and current_day > 25) :
            # Для періодів 01.01-01.26  12.25-12.31 повертаємо порожні значення
            if not search_terms or len(search_terms) == 0:
                return pd.DataFrame(columns=["Search_Term", "Average_Orders", "Average_Clicks"])
            
            # Конвертуємо numpy array у список, якщо потрібно
            if hasattr(search_terms, 'tolist'):
                search_terms = search_terms.tolist()
                
            # Повертаємо DataFrame з порожніми значеннями для всіх пошукових термінів
            default_data = {
                "Search_Term": search_terms,
                "Average_Orders": [1 for _ in range(len(search_terms))],
                "Average_Clicks": [1 for _ in range(len(search_terms))]
            }
            return pd.DataFrame(default_data)
        
        # Визначаємо специфічні періоди порівняння залежно від поточної дати
        if is_peak_season:
            # Для піків продажів використовуємо дані перед піком або з аналогічного періоду минулого місяця
            if current_month == 11 and current_day >= 20:  # Black Friday
                # Використовуємо початок листопада перед Black Friday
                previous_start_date = f"{current_start_date_obj.year}-11-01"
                previous_end_date = f"{current_start_date_obj.year}-11-07"
            elif current_month == 12:  # Різдвяний сезон
                # Використовуємо початок грудня
                previous_start_date = f"{current_start_date_obj.year}-11-01"
                previous_end_date = f"{current_start_date_obj.year}-11-07"
            elif current_month == 7 and 10 <= current_day <= 20:  # Prime Day
                # Використовуємо початок липня перед Prime Day
                previous_start_date = f"{current_start_date_obj.year}-07-01"
                previous_end_date = f"{current_start_date_obj.year}-07-06"
            elif current_month == 10 and 10 <= current_day <= 15:  # Deal Days
                # Використовуємо початок жовтня перед Deal Days
                previous_start_date = f"{current_start_date_obj.year}-09-23"
                previous_end_date = f"{current_start_date_obj.year}-09-30"
            else:
                # Для інших піків використовуємо стандартну логіку
                previous_start_date = (current_start_date_obj - timedelta(days=days_back)).strftime('%Y-%m-%d')
                previous_end_date = (current_start_date_obj - timedelta(days=3)).strftime('%Y-%m-%d')
        elif current_month >= 1 and current_month < 3:
            previous_start_date = f"{current_start_date_obj.year}-01-20"
            previous_end_date = f"{current_start_date_obj.year}-01-25"
        elif current_month >= 3 and current_month < 5:
            previous_start_date = f"{current_start_date_obj.year}-02-25"
            previous_end_date = f"{current_start_date_obj.year}-03-01"
        elif current_month >= 5 and current_month < 7:
            previous_start_date = f"{current_start_date_obj.year}-04-20"
            previous_end_date = f"{current_start_date_obj.year}-04-27"    
        elif current_month >= 7 and current_month < 10:
            previous_start_date = f"{current_start_date_obj.year}-06-23"
            previous_end_date = f"{current_start_date_obj.year}-06-29"
        elif current_month == 10 :
            previous_start_date = f"{current_start_date_obj.year}-09-18"
            previous_end_date = f"{current_start_date_obj.year}-09-25"        
        else:
            previous_start_date = f"{current_start_date_obj.year}-10-25"
            previous_end_date = f"{current_start_date_obj.year}-11-01"   
        
        #print(f"Завантаження попередніх даних за період: {previous_start_date} - {previous_end_date}")
        
        # Якщо список пошукових термінів порожній, повертаємо порожній DataFrame
        if not search_terms or len(search_terms) == 0:
            return pd.DataFrame(columns=["Search_Term", "Average_Orders", "Average_Clicks"])
        
        # Конвертуємо numpy array у список, якщо потрібно
        if hasattr(search_terms, 'tolist'):
            search_terms = search_terms.tolist()
        
        # Формуємо запит для отримання середніх значень за попередній період
        base_query = """
        SELECT 
            id_amz_search_term as "Search_Term",
            AVG(CASE WHEN total_orders IS NOT NULL THEN total_orders ELSE 0 END) as "Average_Orders",
            AVG(CASE WHEN total_clicks IS NOT NULL THEN total_clicks ELSE 0 END) as "Average_Clicks"
        FROM ad_amz_search_term_daily_data
        WHERE date BETWEEN :start_date AND :end_date
        AND id_amz_marketplace = 2
        AND id_amz_search_term IN :search_terms
        GROUP BY id_amz_search_term
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
                    start_date=previous_start_date,
                    end_date=previous_end_date,
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
        
        # Об'єднуємо всі чанки в один DataFrame
        if not chunks:
            # Якщо дані не знайдено, повертаємо DataFrame з дефолтними значеннями для всіх пошукових термінів
            default_data = {
                "Search_Term": search_terms,
                "Average_Orders": [1 for _ in range(len(search_terms))],
                "Average_Clicks": [1 for _ in range(len(search_terms))]
            }
            return pd.DataFrame(default_data)
        
        result_df = pd.concat(chunks, ignore_index=True)
        
        # Перевіряємо, чи для всіх пошукових термінів є дані
        missing_terms = set(search_terms) - set(result_df["Search_Term"].tolist())
        
        # Якщо є відсутні терміни, додаємо їх з дефолтними значеннями
        if missing_terms:
            missing_df = pd.DataFrame({
                "Search_Term": list(missing_terms),
                "Average_Orders": [1 for _ in range(len(missing_terms))],
                "Average_Clicks": [1 for _ in range(len(missing_terms))]
            })
            result_df = pd.concat([result_df, missing_df], ignore_index=True)
        
        return result_df
    except Exception as e:
        print(f"Помилка при завантаженні попередніх даних: {e}")
        # Повертаємо DataFrame з дефолтними значеннями для всіх пошукових термінів
        default_data = {
            "Search_Term": search_terms,
            "Average_Orders": [1 for _ in range(len(search_terms))],
            "Average_Clicks": [1 for _ in range(len(search_terms))]
        }
        return pd.DataFrame(default_data)    
            

