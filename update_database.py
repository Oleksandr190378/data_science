import pandas as pd
import time
from typing import List
from datetime import datetime, date
from sqlalchemy import text
from logger import get_logger

# Отримуємо логер для модуля update_database
logger = get_logger("update_database")

def update_table_with_results(
    engine,
    result_dfs: List[pd.DataFrame],
    chunk_size: int = 1000
) -> None:
    """
    Оновлює таблицю ad_amz_search_term_daily_data результатами аналізу.
    
    :param engine: SQLAlchemy engine для підключення до бази даних
    :param result_dfs: список DataFrame з результатами аналізу
    :param chunk_size: розмір батчу для оновлень (щоб не перевантажувати базу)
    """
    total_updates = 0
    total_errors = 0
    
    # Об'єднуємо всі DataFrame для зручності обробки
    if not result_dfs:
        logger.warning("Немає даних для оновлення таблиці.")
        return
    
    try:
        logger.info(f"Початок оновлення даних для {len(result_dfs)} пошукових термінів")
        
        # Для кожного DataFrame з результатами
        for df_index, df in enumerate(result_dfs):
            if df.empty:
                continue
                
            search_term = df['Search_Term'].iloc[0]
            
            # Переконуємося, що search_term є цілим числом
            try:
                search_term = int(search_term)
            except (ValueError, TypeError):
                logger.error(f"Значення search_term '{search_term}' не є цілим числом. Пропускаємо.")
                continue
            
            # Обробляємо кожен рядок DataFrame
            updates_for_term = 0
            errors_for_term = 0
            
            for idx, row in df.iterrows():
                # Пропускаємо рядки без дати
                if not isinstance(idx, (datetime, date)):
                    continue
                
                date_str = idx.strftime('%Y-%m-%d')
                
                # Підготовка даних для оновлення
                update_data = {
                    'top_1_clicks': int(row.get('ASIN1_Clicks', 0)),
                    'top_2_clicks': int(row.get('ASIN2_Clicks', 0)),
                    'top_3_clicks': int(row.get('ASIN3_Clicks', 0)),
                    'other_clicks': int(row.get('Other_Clicks', 0)),
                    'total_clicks': int(row.get('Total_Clicks', 0)),
                    'top_1_orders': int(row.get('ASIN1_Orders', 0)),
                    'top_2_orders': int(row.get('ASIN2_Orders', 0)),
                    'top_3_orders': int(row.get('ASIN3_Orders', 0)),
                    'other_orders': int(row.get('Other_Orders', 0)),
                    'total_orders': int(row.get('Total_Orders', 0)),
                    'id_amz_search_term': search_term,
                    'date': date_str
                }
                
                # Формування та виконання SQL-запиту для оновлення
                update_query = text("""
                UPDATE ad_amz_search_term_daily_data
                SET 
                    top_1_clicks = :top_1_clicks,
                    top_2_clicks = :top_2_clicks,
                    top_3_clicks = :top_3_clicks,
                    other_clicks = :other_clicks,
                    total_clicks = :total_clicks,
                    top_1_orders = :top_1_orders,
                    top_2_orders = :top_2_orders,
                    top_3_orders = :top_3_orders,
                    other_orders = :other_orders,
                    total_orders = :total_orders
                WHERE 
                    id_amz_search_term = :id_amz_search_term
                    AND date = :date
                """)
                
                try:
                    with engine.connect() as connection:
                        result = connection.execute(update_query, update_data)
                        connection.commit()
                        
                    rows_updated = result.rowcount
                    if rows_updated > 0:
                        updates_for_term += rows_updated
                        total_updates += rows_updated
                        
                except Exception as e:
                    logger.debug(f"Помилка при оновленні даних для ID={search_term} на дату {date_str}: {str(e)}")
                    errors_for_term += 1
                    total_errors += 1
            
            # Вивід статистики тільки періодично
            if (df_index + 1) % 10000 == 0 or df_index == len(result_dfs) - 1:
                logger.info(f"Оброблено {df_index + 1}/{len(result_dfs)} термінів. Поточна статистика: {total_updates} оновлень, {total_errors} помилок")
            
            # Обмеження частоти запитів
            if (df_index + 1) % chunk_size == 0:
                logger.info(f"Пауза після обробки {df_index + 1} термінів...")
                time.sleep(1)
        
        logger.info(f"Оновлення завершено. Всього оновлено: {total_updates} рядків, помилок: {total_errors}")
        
    except Exception as e:
        logger.error(f"Критична помилка при оновленні таблиці: {str(e)}")

        
