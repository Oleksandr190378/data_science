# Amazon Search Terms Analysis

Проект для аналізу пошукових термінів Amazon, розрахунку кліків, замовлень та оновлення відповідних даних у базі даних.

## Встановлення

1. Клонуйте репозиторій:
```bash
git clone https://github.com/Oleksandr190378/data-computing.git
cd data-computing
```

2. Створіть та активуйте віртуальне середовище:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

3. Встановіть залежності:
```bash
pip install -r requirements.txt
```

4. Налаштуйте параметри підключення до бази даних у файлі `config.py`.

## Налаштування середовищ

Проект підтримує два середовища: `development` (локальне розробництво) та `production` (сервер). 

### Перемикання між середовищами

1. Через аргумент командного рядка:
```bash
python main.py --env production
```

2. Через змінну середовища:
```bash
# Windows
set APP_ENV=production
python main.py

# Linux/macOS
export APP_ENV=production
python main.py
```

## Використання

### Базове використання:
```bash
python main.py --period week --value 5 --year 2025
```

### Аналіз підмножини пошукових термінів:
```bash
python main.py --period week --value 5 --year 2025 --search_range 100:120
```

### Повний список параметрів:
- `--period`: Період для аналізу (day, week, month, quarter, year)
- `--value`: Значення періоду (наприклад, номер тижня, місяця)
- `--year`: Рік для аналізу
- `--search_range`: Діапазон пошукових термінів у форматі "start:end"
- `--days_back`: Кількість днів назад для завантаження попередніх даних
- `--chunk_size`: Розмір чанка для запитів до бази даних
- `--update_chunk`: Розмір чанка для оновлення бази даних
- `--db_host`: Хост бази даних
- `--db_port`: Порт бази даних
- `--db_name`: Назва бази даних
- `--db_user`: Користувач бази даних
- `--db_password`: Пароль бази даних
- `--env`: Середовище для запуску (development, production)

## Структура проекту

- `main.py`: Головний скрипт для запуску аналізу
- `get_id_terms.py`: Отримання унікальних пошукових термінів
- `get_data.py`: Завантаження даних з бази
- `conversions_clicks.py`: Аналіз конверсій та кліків
- `adjust_conv_click.py`: Коригування та обробка даних про кліки та замовлення
- `update_database.py`: Оновлення бази даних результатами аналізу
- `columns.py`: Визначення колонок та їхнього мапінгу
- `logger.py`: Налаштування логування
- `config.py`: Конфігурація для різних середовищ

## Логування

Логи зберігаються у папці `logs/` з іменем файлу, що включає поточну дату.
