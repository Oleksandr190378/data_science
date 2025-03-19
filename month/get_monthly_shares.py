from sqlalchemy import text
from typing import List
import pandas as pd


def get_monthly_shares(
    engine,
    search_terms: List[int],
    month_number: int,
    year: int,
    marketplace: int = 2
) -> pd.DataFrame:
    """
    Отримує частки кліків та конверсій з місячної таблиці ad_amz_search_term_monthly_data.
    
    :param engine: SQLAlchemy engine
    :param search_terms: список id_amz_search_term
    :param month_number: номер місяця
    :param year: рік
    :param marketplace: ідентифікатор маркетплейсу (за замовчуванням  2 - 'US')
    :return: DataFrame з частками кліків та конверсій
    """
    try:
        # Конвертуємо список в tuple для SQL параметрів
        if hasattr(search_terms, 'tolist'):
            search_terms = search_terms.tolist()
        
        search_terms_tuple = tuple(int(term) for term in search_terms)
        
        # Якщо тільки один пошуковий термін, коригуємо синтаксис IN
        if len(search_terms_tuple) == 1:
            search_terms_condition = f"id_amz_search_term = {search_terms_tuple[0]}"
        else:
            search_terms_condition = f"id_amz_search_term IN {search_terms_tuple}"
        
        # Формуємо запит для отримання часток з місячної таблиці
        shares_query = text(f"""
        SELECT 
            id_amz_search_term as "Search_Term",
            amz_search_frequency_rank as "SFR",
            amz_top_1_click_share as "Click_Share_1",
            amz_top_1_conversion_share as "Conversion_Share_1",
            amz_top_2_click_share as "Click_Share_2",
            amz_top_2_conversion_share as "Conversion_Share_2",
            amz_top_3_click_share as "Click_Share_3",
            amz_top_3_conversion_share as "Conversion_Share_3"                               
        FROM ad_amz_search_term_monthly_data
        WHERE {search_terms_condition}
        AND id_amz_marketplace = :marketplace
        AND month = :month_number
        AND year = :year
        """)
        
        # Виконуємо запит
        with engine.connect() as connection:
            shares_df = pd.read_sql(
                shares_query,
                connection,
                params={
                    "marketplace": marketplace,
                    "month_number": month_number,
                    "year": year
                }
            )
        
        # Заповнюємо пропущені значення нулями
        shares_df.fillna(0, inplace=True)
        
        return shares_df
        
    except Exception as e:
        print(f"Помилка при отриманні часток з місячної таблиці: {e}")
        # У випадку помилки, повертаємо порожній DataFrame
        return pd.DataFrame(columns=[
            "Search_Term", "SFR", "Click_Share_1", "Conversion_Share_1", 
            "Click_Share_2", "Conversion_Share_2", "Click_Share_3", "Conversion_Share_3"
        ])

        