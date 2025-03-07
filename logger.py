import logging
import os
from datetime import datetime

# Створюємо папку для логів, якщо вона не існує
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Створюємо унікальне ім'я файлу з поточною датою
current_date = datetime.now().strftime('%Y-%m-%d')
log_file = os.path.join(log_dir, f'amazon_data_{current_date}.log')

# Налаштування логування
def setup_logger():
    # Налаштовуємо базову конфігурацію для кореневого логера
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Повертаємо кореневий логер
    return logging.getLogger()

# Створюємо глобальний логер
logger = setup_logger()

def get_logger(name=None):
    """
    Повертає логер для певного модуля.
    Якщо ім'я не вказано, повертає кореневий логер.
    """
    if name:
        return logging.getLogger(name)
    return logger