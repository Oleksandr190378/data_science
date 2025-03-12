import pandas as pd
import datetime
from datetime import date, timedelta


def calculate_seasonal_factor(date_obj):
    """
    Розраховує сезонний коефіцієнт для заданої дати з автоматичним обчисленням святкових дат.
    
    :param date_obj: об'єкт datetime.date або datetime.datetime
    :return: сезонний коефіцієнт (множник для корегування)
    """
    # Дефолтний коефіцієнт (звичайний день)
    factor = 1.0
    # Переконаємося, що маємо об'єкт datetime.date
    if isinstance(date_obj, datetime.datetime):
        date_obj = date_obj.date()
    elif isinstance(date_obj, pd.Timestamp):
        date_obj = date_obj.date()
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    
    # Функція для знаходження Black Friday (4-й четвер листопада)
    def get_black_friday(year):
        # Знаходимо перший день листопада
        november_first = date(year, 11, 1)
        # Знаходимо перший четвер листопада
        first_thursday = november_first + timedelta(days=((3 - november_first.weekday()) % 7))
        # 4-й четвер - це Black Friday
        black_friday = first_thursday + timedelta(days=21)  # 3 тижні після першого четверга
        return black_friday
    
    # Функція для знаходження Cyber Monday (понеділок після Black Friday)
    def get_cyber_monday(year):
        black_friday = get_black_friday(year)
        # Cyber Monday - понеділок після Black Friday
        cyber_monday = black_friday + timedelta(days=3)  # +3 дні від п'ятниці до понеділка
        return cyber_monday
    
    # Функція для знаходження Mother's Day (друга неділя травня)
    def get_mothers_day(year):
        # Знаходимо перший день травня
        may_first = date(year, 5, 1)
        # Знаходимо першу неділю травня
        first_sunday = may_first + timedelta(days=((6 - may_first.weekday()) % 7))
        # Друга неділя травня
        mothers_day = first_sunday + timedelta(days=7)
        return mothers_day
    
    # Конкретні дати для Prime Day (змінюються щороку)
    prime_day_dates = {
        2024: [(7, 16), (7, 17)],           # 16-17 липня 2024
        2025: [(7, 15), (7, 16)]            # Прогноз на 2025
    }
    
    # Конкретні дати для Deal Days / October Prime Day
    deal_days_dates = {
        2024: [(10, 8), (10, 9)],           # 8-9 жовтня 2024
        2025: [(10, 7), (10, 8)]            # Прогноз на 2025
    }
    
    # Обчислюємо Black Friday для поточного року
    black_friday_date = get_black_friday(year)
    
    # Обчислюємо Cyber Monday для поточного року
    cyber_monday_date = get_cyber_monday(year)
    
    # Обчислюємо Mother's Day для поточного року
    mothers_day_date = get_mothers_day(year)
    
    # Перевіряємо чи дата відповідає Prime Day
    if year in prime_day_dates:
        exact_prime_days = prime_day_dates[year]
        if (month, day) in exact_prime_days:
            return 2.8  # Пік Prime Day
        
        # Дні до Prime Day (підготовка)
        for pd_month, pd_day in exact_prime_days:
            if month == pd_month and pd_day - 5 <= day < pd_day:
                return 1.5  # Підготовка до Prime Day
        
        # Дні після Prime Day (хвіст)
        for pd_month, pd_day in exact_prime_days:
            if month == pd_month and pd_day < day <= pd_day + 3:
                return 1.8  # Хвіст після Prime Day
    
    # Перевіряємо чи дата відповідає Deal Days
    if year in deal_days_dates:
        exact_deal_days = deal_days_dates[year]
        if (month, day) in exact_deal_days:
            return 2.2  # Пік Deal Days
        
        # Дні до Deal Days (підготовка)
        for dd_month, dd_day in exact_deal_days:
            if month == dd_month and dd_day - 5 <= day < dd_day:
                return 1.4  # Підготовка до Deal Days
        
        # Дні після Deal Days (хвіст)
        for dd_month, dd_day in exact_deal_days:
            if month == dd_month and dd_day < day <= dd_day + 3:
                return 1.6  # Хвіст після Deal Days
    
    # Перевіряємо Black Friday
    bf_date = black_friday_date
    
    # Сам Black Friday
    if date_obj == bf_date:
        return 3.0  # Пік Black Friday
    
    # Тиждень до Black Friday
    week_before_bf = bf_date - timedelta(days=7)
    if week_before_bf <= date_obj < bf_date:
        return 1.8  # Підготовка до Black Friday
    
    # Вихідні після Black Friday
    weekend_after_bf = bf_date + timedelta(days=2)
    if bf_date < date_obj <= weekend_after_bf:
        return 2.4  # Вихідні після Black Friday
    
    # Перевіряємо Cyber Monday
    cm_date = cyber_monday_date
    
    # Сам Cyber Monday
    if date_obj == cm_date:
        return 2.6  # Пік Cyber Monday
    
    # Дні після Cyber Monday
    days_after_cm = cm_date + timedelta(days=3)
    if cm_date < date_obj <= days_after_cm:
        return 1.8  # Дні після Cyber Monday
    
    # Різдвяний сезон (грудень)
    if month == 12:
        if 1 <= day <= 15:
            return 1.9  # Перша половина грудня
        elif 16 <= day <= 20:
            return 2.2  # Пік перед Різдвом
        elif 21 <= day <= 24:
            return 2.4  # Останні дні перед Різдвом
    
    # Back to School (серпень)
    if month == 8:
        if 1 <= day <= 15:
            return 1.3  # Початок Back to School
        elif 16 <= day <= 31:
            return 1.5  # Пік Back to School
    
    # Valentine's Day
    if month == 2:
        if 1 <= day <= 7:
            return 1.2  # Початок лютого
        elif 8 <= day <= 13:
            return 1.3  # Дні перед Valentine's Day
        elif day == 14:
            return 1.5  # Valentine's Day
    
    # Mother's Day
    md_date = mothers_day_date
    
    # Тиждень перед Mother's Day
    week_before_md = md_date - timedelta(days=7)
    if week_before_md <= date_obj < md_date:
        return 1.3
    
    # Сам Mother's Day
    if date_obj == md_date:
        return 1.5
    
    # Для всіх інших дат
    return factor

