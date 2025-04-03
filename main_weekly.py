import pandas as pd
import argparse
import os
import gc  # Додаємо для виклику збирача сміття
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from sqlalchemy import create_engine, text
from get_id_terms import get_unique_search_terms_for_period, get_date_range
from week.get_weekly_data import get_daily_data, check_daily_data_availability
from week.weekly_conversions_clicks import analyze_weekly_search_terms
from week.update_weekly_database import update_weekly_table_with_results
from config import get_config
from logger import get_logger  # Import the logger module used in main.py


# Get logger for main_weekly module
logger = get_logger("main_weekly")


def worker_process(db_params, start_date, end_date, search_terms_chunk, week_number, year, update_chunk, worker_id):
    """
    Функція, що виконується в окремому процесі для обробки частини пошукових термінів
    
    :param db_params: параметри підключення до бази даних
    :param start_date: початкова дата аналізу
    :param end_date: кінцева дата аналізу
    :param search_terms_chunk: список пошукових термінів для обробки
    :param week_number: номер тижня для аналізу
    :param year: рік для аналізу
    :param update_chunk: розмір чанка для оновлення бази даних
    :param worker_id: ідентифікатор робочого процесу
    :return: статус завершення (True/False)
    """
    # Налаштовуємо логер для воркера
    import logging
    worker_logger = logging.getLogger(f"worker_weekly_{worker_id}")
    worker_logger.setLevel(logging.INFO)
    
    if not worker_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        worker_logger.addHandler(handler)
    
    try:
        # Створюємо нове з'єднання з базою даних для кожного процесу
        engine = create_engine(f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")
        
        worker_logger.info(f"Воркер {worker_id}: Початок обробки {len(search_terms_chunk)} термінів")
        
        # Аналіз пошукових термінів для тижневих даних
        results = analyze_weekly_search_terms(
            engine=engine,
            start_date=start_date,
            end_date=end_date,
            week_number=week_number,
            year=year,
            search_list=search_terms_chunk
        )
        
        if not results:
            worker_logger.warning(f"Воркер {worker_id}: Не отримано результатів аналізу")
            return False
        
        worker_logger.info(f"Воркер {worker_id}: Отримано {len(results)} результатів")
        
        # Оновлюємо базу даних результатами
        update_weekly_table_with_results(
            engine=engine,
            result_dfs=results,
            week_number=week_number,
            year=year,
            chunk_size=update_chunk
        )
        
        # Звільняємо пам'ять
        del results
        gc.collect()
        
        worker_logger.info(f"Воркер {worker_id}: Обробку завершено успішно")
        return True
    except Exception as e:
        worker_logger.error(f"Воркер {worker_id}: Помилка при обробці даних: {str(e)}")
        return False


def process_chunk_parallel(db_params, start_date, end_date, terms_chunk, week_number, year, update_chunk, num_workers):
    """
    Обробляє один чанк термінів паралельно, використовуючи кілька процесів
    
    :param db_params: параметри підключення до бази даних
    :param start_date: початкова дата аналізу
    :param end_date: кінцева дата аналізу
    :param terms_chunk: чанк пошукових термінів
    :param week_number: номер тижня для аналізу
    :param year: рік для аналізу
    :param update_chunk: розмір чанка для оновлення бази даних
    :param num_workers: кількість паралельних процесів
    :return: успішність операції (True/False)
    """
    logger.info(f"Паралельна обробка чанка з {len(terms_chunk)} термінів, {num_workers} робочих процесів")
    
    # Розбиваємо чанк термінів на підчанки для паралельної обробки
    worker_chunk_size = max(1, len(terms_chunk) // num_workers)
    worker_chunks = [terms_chunk[i:i+worker_chunk_size] for i in range(0, len(terms_chunk), worker_chunk_size)]
    
    logger.info(f"Розділено на {len(worker_chunks)} підчанки")
    
    success_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Створюємо завдання для кожного підчанка
        futures = {
            executor.submit(
                worker_process, 
                db_params, 
                start_date, 
                end_date, 
                worker_chunk,
                week_number,
                year,
                update_chunk,
                i  # Ідентифікатор воркера
            ): i for i, worker_chunk in enumerate(worker_chunks)
        }
        
        # Обробляємо результати в міру їх завершення
        for future in as_completed(futures):
            worker_idx = futures[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                    logger.info(f"Підчанк {worker_idx+1}/{len(worker_chunks)} оброблено успішно")
                else:
                    logger.warning(f"Підчанк {worker_idx+1}/{len(worker_chunks)} завершено з помилками")
            except Exception as e:
                logger.error(f"Помилка при обробці підчанка {worker_idx+1}/{len(worker_chunks)}: {str(e)}")
    
    logger.info(f"Паралельну обробку завершено: {success_count}/{len(worker_chunks)} підчанків успішно")
    return success_count > 0


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
    parser.add_argument('--max_terms', type=int, default=20000,
                      help='Максимальна кількість пошукових термінів для обробки за один раз')
    # Додаємо нові параметри для паралельної обробки
    parser.add_argument('--parallel', action='store_true',
                      help='Використовувати паралельну обробку даних')
    parser.add_argument('--workers', type=int, default=None,
                      help='Кількість процесів для паралельної обробки (за замовчуванням - кількість ядер CPU)')
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
    
    # Визначаємо кількість процесів для паралельної обробки
    if args.parallel:
        num_workers = args.workers or max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Увімкнено паралельну обробку з {num_workers} процесами")
    
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
            memory_chunk_size = args.max_terms // 2  # Розмір чанка вдвічі менший за максимальний розмір
            memory_chunks = [search_terms[i:i+memory_chunk_size] for i in range(0, len(search_terms), memory_chunk_size)]
            
            # Обробляємо кожну частину окремо
            for i, memory_chunk in enumerate(memory_chunks):
                logger.info(f"Обробка чанка {i+1}/{len(memory_chunks)} з {len(memory_chunk)} термінів")
                
                if args.parallel:
                    # Паралельна обробка чанка
                    logger.info(f"Використовуємо паралельну обробку з {num_workers} процесами для чанка {i+1}")
                    success = process_chunk_parallel(
                        db_params=db_params,
                        start_date=start_date,
                        end_date=end_date,
                        terms_chunk=memory_chunk,
                        week_number=args.week,
                        year=args.year,
                        update_chunk=args.update_chunk,
                        num_workers=num_workers
                    )
                    if not success:
                        logger.error(f"Помилка при паралельній обробці чанка {i+1}/{len(memory_chunks)}")
                else:
                    # Послідовна обробка чанка
                    process_search_terms_chunk(
                        engine=engine,
                        start_date=start_date,
                        end_date=end_date,
                        search_terms_chunk=memory_chunk,
                        week_number=args.week,
                        year=args.year,
                        update_chunk=args.update_chunk
                    )
                
                # Звільняємо пам'ять після обробки чанка
                gc.collect()
                logger.info(f"Обробка чанка {i+1}/{len(memory_chunks)} завершена")
        else:
            # Обробляємо весь список термінів як один чанк
            logger.info(f"Обробка всіх {len(search_terms)} термінів разом")
            
            if args.parallel:
                # Паралельна обробка всіх термінів
                logger.info(f"Використовуємо паралельну обробку з {num_workers} процесами для всіх {len(search_terms)} термінів")
                success = process_chunk_parallel(
                    db_params=db_params,
                    start_date=start_date,
                    end_date=end_date,
                    terms_chunk=search_terms,
                    week_number=args.week,
                    year=args.year,
                    update_chunk=args.update_chunk,
                    num_workers=num_workers
                )
                if not success:
                    logger.error("Помилка при паралельній обробці термінів")
            else:
                # Послідовна обробка всіх термінів
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

    

