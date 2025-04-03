import pandas as pd
import argparse
import os
import gc  # Додаємо для виклику збирача сміття
from sqlalchemy import create_engine, text
from get_id_terms import get_unique_search_terms_for_period, get_date_range
from week.get_weekly_data import get_daily_data, check_daily_data_availability
from week.weekly_conversions_clicks import analyze_weekly_search_terms
from week.update_weekly_database import update_weekly_table_with_results
from config import get_config
from logger import get_logger  # Import the logger module used in main.py


# Get logger for main_weekly module
logger = get_logger("main_weekly")


def process_search_terms_chunk(engine, start_date, end_date, search_terms_chunk, week_number, year, update_chunk):
    """
    Обробляє один чанк пошукових термінів для тижневих даних
    
    :param engine: з'єднання з базою даних
    :param start_date: початкова дата аналізу
    :param end_date: кінцева дата аналізу
    :param search_terms_chunk: список пошукових термінів для обробки
    :param week_number: номер тижня
    :param year: рік
    :param update_chunk: розмір чанка для оновлення бази даних
    """
    try:
        # Аналіз пошукових термінів для тижневих даних
        logger.info(f"Початок аналізу для чанка з {len(search_terms_chunk)} пошукових термінів...")
        results = analyze_weekly_search_terms(
            engine=engine,
            start_date=start_date,
            end_date=end_date,
            week_number=week_number,
            year=year,
            search_list=search_terms_chunk
        )
        
        if not results:
            logger.warning("Не отримано результатів аналізу для чанка.")
            return
        
        logger.info(f"Отримано результати для {len(results)} пошукових термінів у чанку.")
        
        # Оновлюємо тільки поля кліків та замовлень в таблиці
        logger.info(f"Оновлення бази даних результатами чанка...")
        update_weekly_table_with_results(
            engine=engine,
            result_dfs=results,
            week_number=week_number,
            year=year,
            chunk_size=update_chunk
        )
        
        # Звільняємо пам'ять від результатів
        del results
        gc.collect()
        
        logger.info(f"Обробка чанка успішно завершена.")
    except Exception as e:
        logger.error(f"Помилка при обробці чанка пошукових термінів: {str(e)}")


def main():
    # Отримуємо конфігурацію для поточного середовища
    current_config = get_config()
    db_config = current_config['db']
    app_config = current_config['app']
    
    # Парсинг аргументів командного рядка
    parser = argparse.ArgumentParser(description='Аналіз тижневих даних Amazon')
    parser.add_argument('--week', type=int, required=True,
                      help='Номер тижня для аналізу')
    parser.add_argument('--year', type=int, default=2025,
                      help='Рік для аналізу')
    parser.add_argument('--search_range', type=str, default=None,
                      help='Діапазон пошукових термінів у форматі "start:end" (наприклад, "10:20")')
    parser.add_argument('--list_ids', type=str, default=None,
                      help='Список ID пошукових запитів для аналізу (наприклад, "2108,2316,2596")')
    parser.add_argument('--chunk_size', type=int, default=app_config['chunk_size'],
                      help='Розмір чанка для запитів до бази даних')
    parser.add_argument('--update_chunk', type=int, default=app_config['update_chunk'],
                      help='Розмір чанка для оновлення бази даних')
    parser.add_argument('--max_terms', type=int, default=2000,
                      help='Максимальна кількість пошукових термінів для обробки за один раз')
    parser.add_argument('--db_host', type=str, default=db_config['host'],
                      help='Хост бази даних')
    parser.add_argument('--db_port', type=int, default=db_config['port'],
                      help='Порт бази даних')
    parser.add_argument('--db_name', type=str, default=db_config['database'],
                      help='Назва бази даних')
    parser.add_argument('--db_user', type=str, default=db_config['user'],
                      help='Користувач бази даних')
    parser.add_argument('--db_password', type=str, default=db_config['password'],
                      help='Пароль бази даних')
    parser.add_argument('--env', type=str, default=os.environ.get('APP_ENV', 'development'),
                      choices=['development', 'production'],
                      help='Середовище для запуску (development, production)')
    
    args = parser.parse_args()
    
    # Оновлюємо змінну середовища на основі аргумента --env
    if args.env:
        os.environ['APP_ENV'] = args.env
    
    # Параметри підключення до бази даних
    db_params = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    logger.info(f"Запуск у середовищі: {os.environ.get('APP_ENV', 'development')}")
    logger.info(f"Підключення до бази даних: {db_params['host']}:{db_params['port']}/{db_params['database']}")
    
    # Створення з'єднання з базою даних
    try:
        engine = create_engine(f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")
    except Exception as e:
        logger.error(f"Помилка при підключенні до бази даних: {str(e)}")
        return
    
    # Отримуємо діапазон дат для вказаного тижня
    try:
        start_date, end_date = get_date_range('week', args.week, args.year)
        logger.info(f"Аналіз даних для тижня {args.week} ({start_date} - {end_date})")
    except Exception as e:
        logger.error(f"Помилка при визначенні дат: {str(e)}")
        return
    
    # Перевіряємо, чи є щоденні дані (конверсії та кліки) для цього періоду
    try:
        # Отримуємо список пошукових термінів за весь період
        search_terms = get_unique_search_terms_for_period(engine, start_date, end_date)
        logger.info(f"Знайдено {len(search_terms)} унікальних пошукових термінів")
        
        # Парсимо діапазон пошукових термінів, якщо вказано
        if args.search_range:
            try:
                start_idx, end_idx = map(int, args.search_range.split(':'))
                if end_idx > len(search_terms):
                    end_idx = len(search_terms)
                if start_idx >= len(search_terms):
                    logger.error(f"Початковий індекс {start_idx} більший або рівний кількості термінів {len(search_terms)}. Неможливо продовжити.")
                    return
                search_terms = search_terms[start_idx:end_idx]
                logger.info(f"Використовуємо підмножину пошукових термінів: {start_idx}:{end_idx} ({len(search_terms)} термінів)")
            except Exception as e:
                logger.error(f"Помилка при парсингу діапазону пошукових термінів: {str(e)}")
                return
        
        # Якщо передано список ID, фільтруємо пошукові терміни за цими ID
        if args.list_ids:
            try:
                list_ids = [int(id) for id in args.list_ids.split(',')]
                search_terms = [term for term in search_terms if term in list_ids]
                logger.info(f"Використовуємо пошукові терміни зі списку ID: {args.list_ids} ({len(search_terms)} термінів)")
            except Exception as e:
                logger.error(f"Помилка при обробці списку ID: {str(e)}")
                return
        
        # Перевіряємо наявність щоденних даних
        has_daily_data = check_daily_data_availability(engine, search_terms, start_date, end_date)
        
        if not has_daily_data:
            logger.warning("Для цього періоду ще немає щоденних даних конверсій та кліків. Спочатку запустіть main.py для обробки щоденних даних.")
            return
        
        # Перевіряємо, чи існують записи в тижневій таблиці
        check_weekly_query = text(f"""
        SELECT COUNT(*) FROM ad_amz_search_term_weekly_data 
        WHERE week = {args.week} AND year = {args.year}
        """)
        
        with engine.connect() as connection:
            weekly_count = connection.execute(check_weekly_query).scalar()
        
        if weekly_count == 0:
            logger.warning("В тижневій таблиці відсутні записи для цього тижня. Спочатку внесіть частки кліків та конверсій.")
            return
            
        logger.info("Знайдено щоденні дані. Починаємо обробку тижневих даних.")
        
        # Перевіряємо розмір списку термінів і обробляємо його по частинах, якщо потрібно
        if len(search_terms) > args.max_terms:
            logger.info(f"Список пошукових термінів перевищує максимальний розмір ({len(search_terms)} > {args.max_terms})")
            logger.info(f"Розбиваємо на частини по {args.max_terms//2} термінів")
            
            # Розбиваємо на частини
            chunk_size = args.max_terms // 2  # Розмір чанка вдвічі менший за максимальний розмір
            term_chunks = [search_terms[i:i+chunk_size] for i in range(0, len(search_terms), chunk_size)]
            
            # Обробляємо кожну частину окремо
            for i, chunk in enumerate(term_chunks):
                logger.info(f"Обробка чанка {i+1}/{len(term_chunks)} з {len(chunk)} термінів")
                process_search_terms_chunk(
                    engine=engine,
                    start_date=start_date,
                    end_date=end_date,
                    search_terms_chunk=chunk,
                    week_number=args.week,
                    year=args.year,
                    update_chunk=args.update_chunk
                )
                
                # Звільняємо пам'ять після обробки чанка
                gc.collect()
                logger.info(f"Обробка чанка {i+1}/{len(term_chunks)} завершена")
        else:
            # Обробляємо весь список термінів як один чанк
            logger.info(f"Обробка всіх {len(search_terms)} термінів разом")
            process_search_terms_chunk(
                engine=engine,
                start_date=start_date,
                end_date=end_date,
                search_terms_chunk=search_terms,
                week_number=args.week,
                year=args.year,
                update_chunk=args.update_chunk
            )
        
        logger.info("Аналіз тижневих даних успішно завершено.")
    except Exception as e:
        logger.error(f"Помилка при обробці даних: {str(e)}")
        return

if __name__ == '__main__':
    main()

