import pandas as pd
import time
from typing import List
from sqlalchemy import text


def update_weekly_table_with_results(
    engine,
    result_dfs: List[pd.DataFrame],
    week_number: int,
    year: int,
    marketplace: str = 2,
    chunk_size: int = 500
):
    """
    Оновлює таблицю ad_amz_search_term_weekly_data результатами аналізу.
    Оновлює тільки поля кліків та замовлень.
    
    :param engine: SQLAlchemy engine
    :param result_dfs: список DataFrame з результатами аналізу
    :param week_number: номер тижня
    :param year: рік
    :param marketplace: ідентифікатор маркетплейсу (за замовчуванням 'US')
    :param chunk_size: розмір чанка для оновлення
    """
    if not result_dfs:
        return
    
    # Лічильник оновлених рядків
    total_rows_updated = 0
    total_errors = 0
    
    # Обробляємо кожен DataFrame з результатами
    for df_index, df in enumerate(result_dfs):
        if df.empty:
            continue
            
        search_term = df['Search_Term'].iloc[0]
        
        # Переконуємося, що search_term є цілим числом
        try:
            search_term = int(search_term)
        except (ValueError, TypeError):
            total_errors += 1
            continue
        
        # Підготовка даних для оновлення
        update_data = {
            'id_amz_search_term': search_term,
            'week_number': week_number,
            'year': year,
            'top_1_clicks': int(df['ASIN1_Clicks'].iloc[0]) if pd.notna(df['ASIN1_Clicks'].iloc[0]) else 0,
            'top_2_clicks': int(df['ASIN2_Clicks'].iloc[0]) if pd.notna(df['ASIN2_Clicks'].iloc[0]) else 0,
            'top_3_clicks': int(df['ASIN3_Clicks'].iloc[0]) if pd.notna(df['ASIN3_Clicks'].iloc[0]) else 0,
            'other_clicks': int(df['Other_Clicks'].iloc[0]) if pd.notna(df['Other_Clicks'].iloc[0]) else 0,
            'total_clicks': int(df['Total_Clicks'].iloc[0]) if pd.notna(df['Total_Clicks'].iloc[0]) else 0,
            'top_1_orders': int(df['ASIN1_Orders'].iloc[0]) if pd.notna(df['ASIN1_Orders'].iloc[0]) else 0,
            'top_2_orders': int(df['ASIN2_Orders'].iloc[0]) if pd.notna(df['ASIN2_Orders'].iloc[0]) else 0,
            'top_3_orders': int(df['ASIN3_Orders'].iloc[0]) if pd.notna(df['ASIN3_Orders'].iloc[0]) else 0,
            'other_orders': int(df['Other_Orders'].iloc[0]) if pd.notna(df['Other_Orders'].iloc[0]) else 0,
            'total_orders': int(df['Total_Orders'].iloc[0]) if pd.notna(df['Total_Orders'].iloc[0]) else 0
        }
        
        # Формування та виконання SQL-запиту для оновлення
        update_query = """
        UPDATE ad_amz_search_term_weekly_data
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
            AND week = :week_number 
            AND year = :year
            AND id_amz_marketplace = 2
        """
        
        # Виконуємо запит
        try:
            with engine.begin() as connection:
                result = connection.execute(text(update_query), update_data)
                rows_updated = result.rowcount
                total_rows_updated += rows_updated
        except Exception as e:
            total_errors += 1
        
        # Обмеження частоти запитів
        if (df_index + 1) % chunk_size == 0:
            time.sleep(1)
    
    print(f"Оновлення завершено. Всього оновлено: {total_rows_updated} рядків, помилок: {total_errors}")

    