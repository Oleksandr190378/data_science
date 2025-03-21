import os
from pathlib import Path
from dotenv import load_dotenv

# Пошук .env файлу: спочатку в поточній директорії, потім у батьківській
env_paths = [
    Path.cwd() / '.env',                # поточна директорія
    Path.cwd().parent / '.env',         # батьківська директорія 
    Path.cwd().parent.parent / '.env',       # директорія на два рівні вище
    Path(__file__).parent / '.env',     # директорія скрипта
    Path(__file__).parent.parent / '.env'  # батьківська директорія скрипта
]

# Спроба завантажити перший знайдений .env файл
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Завантажено .env з {env_path}")
        break
else:
    print("Файл .env не знайдено, використовуємо змінні середовища або значення за замовчуванням")


# Визначення середовища
ENV = os.environ.get('APP_ENV', 'development')
# Визначення середовища (production)
#ENV = os.environ.get('APP_ENV', 'production')
# Конфігурації для різних середовищ
# Конфігурації для різних середовищ
config = {
    'development': {
        'db': {
            'host': os.environ.get('DB_HOST', 'localhost'),
            'port': int(os.environ.get('DB_PORT', 5432)),
            'database': os.environ.get('DB_NAME', 'postgres'),
            'user': os.environ.get('DB_USER', 'postgres'),
            'password': os.environ.get('DB_PASSWORD', 'mysecretpassword')
        },
        'app': {
            'chunk_size': int(os.environ.get('CHUNK_SIZE', 10000)),
            'update_chunk': int(os.environ.get('UPDATE_CHUNK', 100)),
            'days_back': int(os.environ.get('DAYS_BACK', 10))
        }
    },
    'production': {
        'db': {
            'host': os.environ.get('DB_HOST', 'server_hostname'),
            'port': int(os.environ.get('DB_PORT', 5432)),
            'database': os.environ.get('DB_NAME', 'postgres'),
            'user': os.environ.get('DB_USER', 'postgres'),
            'password': os.environ.get('DB_PASSWORD', 'production_password')
        },
        'app': {
            'chunk_size': int(os.environ.get('CHUNK_SIZE', 10000)),
            'update_chunk': int(os.environ.get('UPDATE_CHUNK', 10000)),
            'days_back': int(os.environ.get('DAYS_BACK', 10))
        }
    }
}

# Функція для отримання поточної конфігурації
def get_config():
    """
    Отримує поточну конфігурацію на основі значення ENV.
    
    :return: словник з налаштуваннями для поточного середовища
    """
    return config[ENV]

