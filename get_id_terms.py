from datetime import  date, timedelta
from typing import List, Dict, Tuple
import pandas as pd
from sqlalchemy import create_engine, text


def get_date_range(period: str, value: int, year: int) -> Tuple[str, str]:
    """
    Генерує діапазон дат на основі періоду та значення.
    
    :param period: 'week' або 'month'
    :param value: номер тижня або місяця
    :param year: рік
    :return: початкова дата, кінцева дата
    """
    if period == 'month':
        start_date = date(year, value, 1)
        # Знаходимо останній день місяця
        if value == 12:
            end_date = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = date(year, value + 1, 1) - timedelta(days=1)
    elif period == 'week':
        # Система, де тиждень починається з неділі
        # Знаходимо перший день року
        first_day = date(year, 1, 1)
        
        # Знаходимо першу неділю року (або останню неділю попереднього року)
        # У datetime.weekday() неділя має індекс 6
        day_of_week = first_day.weekday()
        # Конвертуємо в систему, де неділя має індекс 0
        sunday_based_dow = (day_of_week + 1) % 7
        
        # Зсуваємося до першої неділі (або назад до останньої неділі попереднього року)
        first_sunday = first_day - timedelta(days=sunday_based_dow)
        
        # Перший тиждень року - це тиждень, що містить 1 січня
        # Але якщо 1 січня - це не неділя, то перший тиждень починається з неділі перед 1 січня
        
        # Додаємо потрібну кількість тижнів
        start_date = first_sunday + timedelta(weeks=value-1)
        # Кінець тижня (субота)
        end_date = start_date + timedelta(days=6)
    else:
        raise ValueError("Period must be 'week' or 'month'")
        
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def get_unique_search_terms_for_period(
    engine,
    start_date: str,
    end_date: str,
    chunk_size: int = 10000
) -> List[int]:  # Змінюємо повертаємий тип на List[int]
    """
    Отримує список унікальних id_amz_search_term за вказаний період.
    
    :param engine: SQLAlchemy engine
    :param start_date: початкова дата у форматі 'YYYY-MM-DD'
    :param end_date: кінцева дата у форматі 'YYYY-MM-DD'
    :param chunk_size: розмір чанка для зчитування
    :return: список унікальних id_amz_search_term як цілих чисел
    """
    try:
        # Формуємо SQL-запит для пагінації
        paginated_query = text("""
        SELECT DISTINCT id_amz_search_term
        FROM ad_amz_search_term_daily_data
        WHERE date BETWEEN :start_date AND :end_date
        AND id_amz_marketplace = 2
        ORDER BY id_amz_search_term  -- Додаємо сортування для послідовності
        LIMIT :limit OFFSET :offset
        """)
                
        # Завантажуємо дані чанками
        chunks = []
        offset = 0
        
        while True:
            # Виконуємо запит для поточного чанку
            with engine.connect() as connection:
                chunk = pd.read_sql(
                    paginated_query, 
                    connection, 
                    params={
                        "start_date": start_date,
                        "end_date": end_date,
                        "limit": chunk_size,
                        "offset": offset
                    }
                )
            
            # Якщо чанк порожній, виходимо з циклу
            if len(chunk) == 0:
                break
            
            chunks.append(chunk)
            offset += chunk_size
            
            # Якщо отримали менше рядків, ніж розмір чанку, значить дані закінчились
            if len(chunk) < chunk_size:
                break
        
        # Об'єднуємо всі чанки
        if not chunks:
            return []
        
        df = pd.concat(chunks, ignore_index=True)
        # Конвертуємо до цілих чисел перед поверненням
        return df['id_amz_search_term'].astype(int).tolist()
    
    except Exception as e:
        print(f"Помилка при отриманні унікальних id_amz_search_term: {e}")
        return []

        
if __name__ == '__main__':
    db_params = {
    'host': 'localhost',
    'port': 5432,
    'database': 'postgres',
    'user': 'postgres',
    'password': 'mysecretpassword'  
}

    # Створення з'єднання один раз
    engine = create_engine(f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")

    # Визначення періоду
    period = 'week'
    value = 5
    year = 2025
    start_date, end_date = get_date_range(period, value, year)
    print(f"Аналіз даних з {start_date} по {end_date}")

    # Отримуємо список пошукових термінів за весь період
    search_terms = get_unique_search_terms_for_period(engine, start_date, end_date)
    print(f"Знайдено {len(search_terms)} унікальних пошукових термінів") 

