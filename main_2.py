import pandas as pd
import argparse
import os
import gc  # Збирач сміття
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from sqlalchemy import create_engine
from get_id_terms import get_unique_search_terms_for_period, get_date_range
from get_data import load_previous_data
from conversions_clicks import analyze_search_terms
from columns import column_mapping, columns_to_import
from adjust_conv_click import adjust_clicks_and_orders, process_orders_and_clicks
from update_database import update_table_with_results
from logger import get_logger
from config import get_config
from datetime import datetime, timedelta
from seasonal_factor import calculate_seasonal_factor


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
    

def is_hot_period(start_date, end_date):
    """
    Перевіряє, чи входить період в "гарячий" сезон (коефіцієнт >= 1.4)
    
    :param start_date: початкова дата періоду
    :param end_date: кінцева дата періоду
    :return: True, якщо період є "гарячим", False - в іншому випадку
    """
    # Перетворюємо дати в об'єкти datetime, якщо вони не є ними
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Розраховуємо середній сезонний коефіцієнт для всього періоду
    current_date = start_date
    total_factor = 0
    days_count = 0
    factors = []
    
    while current_date <= end_date:
        factor = calculate_seasonal_factor(current_date)
        factors.append(factor)
        total_factor += factor
        days_count += 1
        current_date = current_date + timedelta(days=1)
    
    # Обчислюємо середній коефіцієнт
    avg_factor = total_factor / days_count if days_count > 0 else 0
    max_min_factor = max(factors) - min(factors)
    # Перевіряємо, чи є період "гарячим"
    is_hot = (avg_factor >= 1.25 or max_min_factor > 0.1)
    
    if is_hot:
        logger.info(f"Період з {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')} є гарячим (середній коефіцієнт: {avg_factor:.2f})")
    else:
        logger.info(f"Період з {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')} не є гарячим (середній коефіцієнт: {avg_factor:.2f})")
    
    return is_hot


def worker_process(db_params, start_date, end_date, search_terms_chunk, days_back, chunk_size, is_hot, worker_id):
    """
    Функція, що виконується в окремому процесі для обробки частини пошукових термінів
    
    :param db_params: параметри підключення до бази даних
    :param start_date: початкова дата аналізу
    :param end_date: кінцева дата аналізу
    :param search_terms_chunk: список пошукових термінів для обробки
    :param days_back: кількість днів назад для попередніх даних
    :param chunk_size: розмір чанка для запитів до бази даних
    :param is_hot: чи є період "гарячим"
    :param worker_id: ідентифікатор робочого процесу
    :return: оброблені результати для подальшого оновлення бази даних
    """
    # Налаштовуємо логер для воркера - перенаправляємо на NullHandler, щоб не виводити зайві повідомлення
    import logging
    worker_logger = logging.getLogger(f"worker_{worker_id}")
    worker_logger.setLevel(logging.WARNING)  # Тільки попередження та помилки
    
    # Якщо обробники вже налаштовані, не додаємо нові
    if not worker_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.WARNING)
        worker_logger.addHandler(handler)
    
    # Створюємо нове з'єднання з базою даних для кожного процесу
    engine = create_engine(f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")
    
    try:
        # Завантажуємо дані за попередній період
        previous_data = load_previous_data(
            engine=engine,
            search_terms=search_terms_chunk,
            current_start_date=start_date,
            days_back=days_back
        )
        
        # Аналіз пошукових термінів
        results = analyze_search_terms(
            engine=engine,
            start_date=start_date,
            end_date=end_date,
            columns_to_import=columns_to_import,
            column_mapping=column_mapping,
            previous_month_data=previous_data,
            search_list=search_terms_chunk,
            chunk_size=chunk_size
        )
        
        # Звільняємо пам'ять від попередніх даних
        del previous_data
        gc.collect()
        
        # Обробка результатів залежно від того, чи є період "гарячим"
        if is_hot:
            adjusted_results = results
        else:
            adjusted_results = adjust_clicks_and_orders(results)
        
        # Звільняємо пам'ять від результатів аналізу
        del results
        gc.collect()
        
        # Обробка кліків та замовлень
        processed_results = process_orders_and_clicks(adjusted_results)
        
        # Звільняємо пам'ять від відкоригованих результатів
        del adjusted_results
        gc.collect()
        
        return processed_results
    except Exception as e:
        worker_logger.error(f"Процес {worker_id}: Помилка при обробці чанка: {str(e)}")
        return None


def process_chunk_parallel(db_params, start_date, end_date, terms_chunk, days_back, chunk_size, update_chunk, is_hot, num_workers):
    """
    Обробляє один чанк термінів паралельно, використовуючи кілька процесів
    
    :param db_params: параметри підключення до бази даних
    :param start_date: початкова дата аналізу
    :param end_date: кінцева дата аналізу
    :param terms_chunk: чанк пошукових термінів
    :param days_back: кількість днів назад для попередніх даних
    :param chunk_size: розмір чанка для запитів до бази даних
    :param update_chunk: розмір чанка для оновлення бази даних
    :param is_hot: чи є період "гарячим"
    :param num_workers: кількість паралельних процесів
    :return: оброблені результати для оновлення бази даних
    """
    # Створюємо з'єднання з базою даних
    engine = create_engine(f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")
    
    # Розбиваємо чанк термінів на підчанки для паралельної обробки
    worker_chunk_size = max(1, len(terms_chunk) // num_workers)
    worker_chunks = [terms_chunk[i:i+worker_chunk_size] for i in range(0, len(terms_chunk), worker_chunk_size)]
    
    logger.info(f"Розділяємо чанк з {len(terms_chunk)} термінів на {len(worker_chunks)} підчанків для паралельної обробки")
    
    # Запускаємо паралельну обробку
    all_results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Створюємо завдання для кожного підчанка
        futures = {
            executor.submit(
                worker_process, 
                db_params, 
                start_date, 
                end_date, 
                worker_chunk, 
                days_back, 
                chunk_size,
                is_hot,
                i  # Ідентифікатор воркера
            ): i for i, worker_chunk in enumerate(worker_chunks)
        }
        
        # Обробляємо результати в міру їх завершення
        for future in as_completed(futures):
            worker_idx = futures[future]
            try:
                result = future.result()
                if result:
                    all_results.extend(result)
                    logger.info(f"Підчанк {worker_idx+1}/{len(worker_chunks)} оброблено успішно ({len(result)} результатів)")
                else:
                    logger.warning(f"Підчанк {worker_idx+1}/{len(worker_chunks)} не повернув результатів")
            except Exception as e:
                logger.error(f"Помилка при обробці підчанка {worker_idx+1}/{len(worker_chunks)}: {str(e)}")
    
    # Оновлення бази даних для цього чанка
    if all_results:
        logger.info(f"Оновлення бази даних {len(all_results)} результатами для поточного чанка...")
        update_table_with_results(
            engine=engine,
            result_dfs=all_results,
            chunk_size=update_chunk
        )
        logger.info("Оновлення бази даних для поточного чанка завершено.")
    else:
        logger.warning("Немає результатів для оновлення бази даних для поточного чанка.")
    
    return True


def main():
    # Отримуємо конфігурацію для поточного середовища
    current_config = get_config()
    db_config = current_config['db']
    app_config = current_config['app']
    
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
    parser.add_argument('--days_back', type=int, default=app_config['days_back'],
                        help='Кількість днів назад для завантаження попередніх даних')
    parser.add_argument('--chunk_size', type=int, default=app_config['chunk_size'],
                        help='Розмір чанка для запитів до бази даних')
    parser.add_argument('--update_chunk', type=int, default=app_config['update_chunk'],
                        help='Розмір чанка для оновлення бази даних')
    parser.add_argument('--max_terms', type=int, default=2000,
                        help='Максимальна кількість пошукових термінів для обробки за один раз')
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
    parser.add_argument('--list_ids', type=str, default=None,
                        help='Список ID пошукових запитів для аналізу (наприклад, "2108,2316,2596")')
    
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
    
    # Якщо передано список ID, фільтруємо пошукові терміни за цими ID
    if args.list_ids:
        try:
            list_ids = [int(id) for id in args.list_ids.split(',')]
            selected_terms = [term for term in selected_terms if term in list_ids]
            logger.info(f"Використовуємо пошукові терміни зі списку ID: {args.list_ids} ({len(selected_terms)} термінів)")
        except Exception as e:
            logger.error(f"Помилка при обробці списку ID: {str(e)}")
            return
    
    # Перевіряємо, чи є період "гарячим" (один раз на весь аналіз)
    is_hot = is_hot_period(start_date, end_date)
    
    # Визначаємо кількість процесів для паралельної обробки
    num_workers = args.workers or max(1, multiprocessing.cpu_count() - 1)
    
    # Обробка випадку, коли кількість термінів перевищує максимальний розмір
    if len(selected_terms) > args.max_terms:
        logger.info(f"Список пошукових термінів перевищує максимальний розмір ({len(selected_terms)} > {args.max_terms})")
        logger.info(f"Розбиваємо на частини по {args.max_terms//2} термінів")
        
        # Розбиваємо на частини
        memory_chunk_size = args.max_terms // 2  # Розмір чанка для контролю пам'яті
        memory_chunks = [selected_terms[i:i+memory_chunk_size] for i in range(0, len(selected_terms), memory_chunk_size)]
        
        # Обробляємо кожну частину
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
                    days_back=args.days_back,
                    chunk_size=args.chunk_size,
                    update_chunk=args.update_chunk,
                    is_hot=is_hot,
                    num_workers=num_workers
                )
                if not success:
                    logger.error(f"Помилка при паралельній обробці чанка {i+1}/{len(memory_chunks)}")
            else:
                # Послідовна обробка чанка
                try:
                    # Завантажуємо дані за попередній період
                    previous_data = load_previous_data(
                        engine=engine,
                        search_terms=memory_chunk,
                        current_start_date=start_date,
                        days_back=args.days_back
                    )
                    
                    # Аналіз пошукових термінів
                    results = analyze_search_terms(
                        engine=engine,
                        start_date=start_date,
                        end_date=end_date,
                        columns_to_import=columns_to_import,
                        column_mapping=column_mapping,
                        previous_month_data=previous_data,
                        search_list=memory_chunk,
                        chunk_size=args.chunk_size
                    )
                    
                    # Звільняємо пам'ять від попередніх даних
                    del previous_data
                    gc.collect()
                    
                    # Обробка результатів залежно від того, чи є період "гарячим"
                    if is_hot:
                        adjusted_results = results
                    else:
                        adjusted_results = adjust_clicks_and_orders(results)
                    
                    # Звільняємо пам'ять від результатів аналізу
                    del results
                    gc.collect()
                    
                    # Обробка кліків та замовлень
                    processed_results = process_orders_and_clicks(adjusted_results)
                    
                    # Оновлення бази даних
                    logger.info(f"Оновлення бази даних результатами для чанка {i+1}...")
                    update_table_with_results(
                        engine=engine,
                        result_dfs=processed_results,
                        chunk_size=args.update_chunk
                    )
                    
                    # Звільняємо пам'ять від оброблених результатів
                    del adjusted_results, processed_results
                    gc.collect()
                    
                    logger.info(f"Обробка чанка {i+1}/{len(memory_chunks)} завершена")
                except Exception as e:
                    logger.error(f"Помилка при обробці чанка {i+1}/{len(memory_chunks)}: {str(e)}")
            
            # Звільняємо пам'ять після обробки чанка
            gc.collect()
    else:
        # Коли кількість термінів не перевищує максимальний розмір
        if args.parallel:
            # Паралельна обробка всіх термінів
            logger.info(f"Використовуємо паралельну обробку з {num_workers} процесами для всіх {len(selected_terms)} термінів")
            success = process_chunk_parallel(
                db_params=db_params,
                start_date=start_date,
                end_date=end_date,
                terms_chunk=selected_terms,
                days_back=args.days_back,
                chunk_size=args.chunk_size,
                update_chunk=args.update_chunk,
                is_hot=is_hot,
                num_workers=num_workers
            )
            if not success:
                logger.error("Помилка при паралельній обробці термінів")
        else:
            # Послідовна обробка всіх термінів
            logger.info(f"Обробка всіх {len(selected_terms)} термінів послідовно")
            
            try:
                # Завантажуємо дані за попередній період
                previous_data = load_previous_data(
                    engine=engine,
                    search_terms=selected_terms,
                    current_start_date=start_date,
                    days_back=args.days_back
                )
                
                # Аналіз пошукових термінів
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
                
                # Звільняємо пам'ять від попередніх даних
                del previous_data
                gc.collect()
                
                # Обробка результатів залежно від того, чи є період "гарячим"
                if is_hot:
                    adjusted_results = results
                else:
                    adjusted_results = adjust_clicks_and_orders(results)
                
                # Звільняємо пам'ять від результатів аналізу
                del results
                gc.collect()
                
                # Обробка кліків та замовлень
                processed_results = process_orders_and_clicks(adjusted_results)
                
                # Оновлення бази даних
                logger.info("Оновлення бази даних результатами...")
                update_table_with_results(
                    engine=engine,
                    result_dfs=processed_results,
                    chunk_size=args.update_chunk
                )
                logger.info("Оновлення бази даних завершено.")
            except Exception as e:
                logger.error(f"Помилка при обробці даних: {str(e)}")
    
    logger.info("Аналіз даних успішно завершено.")

if __name__ == '__main__':
    main()

    