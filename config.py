import os

# Визначення середовища (development або production)
ENV = os.environ.get('APP_ENV', 'development')
# Визначення середовища (production)
#ENV = os.environ.get('APP_ENV', 'production')
# Конфігурації для різних середовищ
config = {
    'development': {
        'db': {
            'host': 'localhost',
            'port': 5432,
            'database': 'postgres',
            'user': 'postgres',
            'password': 'mysecretpassword'
        },
        'app': {
            'chunk_size': 10000,
            'update_chunk': 100,
            'days_back': 7
        }
    },
    'production': {
        'db': {
            'host': 'server_hostname',  # Замініть на реальну адресу сервера
            'port': 5432,
            'database': 'postgres',
            'user': 'postgres',
            'password': 'production_password'  # Змініть на реальний пароль
        },
        'app': {
            'chunk_size': 10000,
            'update_chunk': 10000,
            'days_back': 7
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

