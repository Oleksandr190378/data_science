import pandas as pd
import argparse
from sqlalchemy import create_engine
from get_id_terms import get_unique_search_terms_for_period, get_date_range
from get_data import load_previous_data
from conversions_clicks import analyze_search_terms
from columns import column_mapping, columns_to_import
from adjust_conv_click import adjust_clicks_and_orders, process_orders_and_clicks
from update_database import update_table_with_results
from logger import get_logger

# Отримуємо логер для main модуля
logger = get_logger("main")

pd.set_option('future.no_silent_downcasting', True)

def parse_search_range(range_str):
    """
    Перетворює рядок з діапазоном у кортеж (start, end)
    Наприклад: "10:20" -> (10, 20)
    """
    if not range_str:
        return None
    
    try:
        parts = range_str.split(':')
        if len(parts) != 2:
            raise ValueError("Формат діапазону має бути 'start:end'")
        
        start = int(parts[0])
        end = int(parts[1])
        
        if start < 0 or end <= start:
            raise ValueError("Початок має бути >= 0, кінець > початку")
        
        return (start, end)
    except Exception as e:
        logger.error(f"Помилка при обробці діапазону '{range_str}': {str(e)}")
        return None

def main():
    # Парсинг аргументів командного рядка
    parser = argparse.ArgumentParser(description='Аналіз даних Amazon')
    parser.add_argument('--period', type=str, default='week',
                        choices=['day', 'week', 'month', 'quarter', 'year'],
                        help='Період для аналізу (day, week, month, quarter, year)')
    parser.add_argument('--value', type=int, default=1,
                        help='Значення періоду (наприклад, номер тижня, місяця)')
    parser.add_argument('--year', type=int, default=2025,
                        help='Рік для аналізу')
    parser.add_argument('--search_range', type=str, default=None,
                        help='Діапазон пошукових термінів у форматі "start:end" (наприклад, "10:20")')
    parser.add_argument('--days_back', type=int, default=7,
                        help='Кількість днів назад для завантаження попередніх даних')
    parser.add_argument('--chunk_size', type=int, default=10000,
                        help='Розмір чанка для запитів до бази даних')
    parser.add_argument('--update_chunk', type=int, default=100,
                        help='Розмір чанка для оновлення бази даних')
    parser.add_argument('--db_host', type=str, default='localhost',
                        help='Хост бази даних')
    parser.add_argument('--db_port', type=int, default=5432,
                        help='Порт бази даних')
    parser.add_argument('--db_name', type=str, default='postgres',
                        help='Назва бази даних')
    parser.add_argument('--db_user', type=str, default='postgres',
                        help='Користувач бази даних')
    parser.add_argument('--db_password', type=str, default='mysecretpassword',
                        help='Пароль бази даних')
    
    args = parser.parse_args()
    
    # Параметри підключення до бази даних
    db_params = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    # Створення з'єднання з базою даних
    try:
        engine = create_engine(f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")
    except Exception as e:
        logger.error(f"Помилка при підключенні до бази даних: {str(e)}")
        return
    
    # Визначення періоду
    try:
        start_date, end_date = get_date_range(args.period, args.value, args.year)
        logger.info(f"Аналіз даних з {start_date} по {end_date}")
    except Exception as e:
        logger.error(f"Помилка при визначенні дат: {str(e)}")
        return
    
    # Отримуємо список пошукових термінів за весь період
    try:
        search_terms = get_unique_search_terms_for_period(engine, start_date, end_date)
        logger.info(f"Знайдено {len(search_terms)} унікальних пошукових термінів")
    except Exception as e:
        logger.error(f"Помилка при отриманні пошукових термінів: {str(e)}")
        return
    
    # Визначення діапазону пошукових термінів для аналізу
    search_range = parse_search_range(args.search_range)
    if search_range:
        start_idx, end_idx = search_range
        if end_idx > len(search_terms):
            logger.warning(f"Кінцевий індекс {end_idx} більший за кількість термінів {len(search_terms)}. Використовуємо доступні терміни.")
            end_idx = len(search_terms)
        
        if start_idx >= len(search_terms):
            logger.error(f"Початковий індекс {start_idx} більший або рівний кількості термінів {len(search_terms)}. Неможливо продовжити.")
            return
            
        selected_terms = search_terms[start_idx:end_idx]
        logger.info(f"Використовуємо підмножину пошукових термінів: {start_idx}:{end_idx} ({len(selected_terms)} термінів)")
    else:
        selected_terms = search_terms
        logger.info(f"Використовуємо всі {len(search_terms)} пошукових термінів")
    
    # Завантажуємо дані за попередній період
    try:
        logger.info(f"Завантаження даних за попередні {args.days_back} днів...")
        previous_data = load_previous_data(
            engine=engine,
            search_terms=selected_terms,
            current_start_date=start_date,
            days_back=args.days_back
        )
    except Exception as e:
        logger.error(f"Помилка при завантаженні попередніх даних: {str(e)}")
        return
    
    # Аналіз пошукових термінів
    try:
        logger.info("Початок аналізу пошукових термінів...")
        results = analyze_search_terms(
            engine=engine,
            start_date=start_date,
            end_date=end_date,
            columns_to_import=columns_to_import,
            column_mapping=column_mapping,
            previous_month_data=previous_data,
            search_list=selected_terms,
            chunk_size=args.chunk_size
        )
        logger.info(f"Отримано результати для {len(results)} пошукових термінів")
    except Exception as e:
        logger.error(f"Помилка при аналізі пошукових термінів: {str(e)}")
        return
    
    # Обробка результатів
    try:
        logger.info("Коригування кліків та замовлень...")
        adjusted_results = adjust_clicks_and_orders(results)
        
        logger.info("Обробка кліків та замовлень...")
        processed_results = process_orders_and_clicks(adjusted_results)
        
        if processed_results and len(processed_results) > 0:
            sample_df = processed_results[0] if processed_results else None
            if sample_df is not None and not sample_df.empty:
                logger.info(f"Приклад результатів:\n{sample_df.head()}")
        else:
            logger.warning("Немає результатів після обробки.")
    except Exception as e:
        logger.error(f"Помилка при обробці результатів: {str(e)}")
        return
    
    # Оновлення бази даних
    try:
        logger.info("Оновлення бази даних результатами...")
        update_table_with_results(
            engine=engine,
            result_dfs=processed_results,
            chunk_size=args.update_chunk
        )
        logger.info("Оновлення бази даних завершено.")
    except Exception as e:
        logger.error(f"Помилка при оновленні бази даних: {str(e)}")
        return
    
    logger.info("Аналіз даних успішно завершено.")

if __name__ == '__main__':
    main()


    