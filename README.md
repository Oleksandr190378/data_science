# Amazon Search Terms Analysis

Проект для аналізу пошукових термінів Amazon, розрахунку кліків, замовлень та оновлення відповідних даних у базі даних.

## Встановлення

1. Клонуйте репозиторій:
```bash
git clone https://github.com/Oleksandr190378/data_science.git
cd data_science
git checkout search_terms
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

4. Налаштуйте параметри підключення до бази даних у файлі `config.py` або через `.env` файл.

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

3. Через `.env` файл:
```
APP_ENV=production
DB_HOST=ваш_хост_бази_даних
DB_PORT=5432
DB_NAME=назва_бази_даних
DB_USER=користувач_бази_даних
DB_PASSWORD=пароль_бази_даних
CHUNK_SIZE=10000
UPDATE_CHUNK=10000
DAYS_BACK=10
```

Модифікований config.py автоматично шукає .env файл у таких локаціях:
* Поточній директорії
* Батьківській директорії
* Директорії скрипта
* Батьківській директорії скрипта

Якщо .env файл не знайдено, скрипт використовуватиме системні змінні середовища або значення за замовчуванням.

## Production налаштування на сервері

### Підготовка сервера

1. **Встановлення необхідних пакетів**:
```bash
# Оновлення списку пакетів
sudo apt update

# Встановлення Python 3
sudo apt install -y python3

# Перевірка версії Python (опційно)
python3 --version

# Встановлення додаткових компонентів
sudo apt install -y python3-pip python3-venv 

# Встановлення Git
sudo apt install -y git-all
```

2. **Клонування репозиторію**:
```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/Oleksandr190378/data_science.git
cd data_science
git checkout search_terms
```

3. **Створення та налаштування віртуального середовища**:
```bash
python3 -m venv venv
source venv/bin/activate
#pip install --upgrade pip
pip install -r requirements.txt
```

4. **Налаштування .env файлу**:

Створіть `.env` файл у кореневій директорії проекту:
```bash
nano .env
```

Додайте необхідні параметри:
```
APP_ENV=production
DB_HOST=ваш_хост_бази_даних
DB_PORT=5432
DB_NAME=назва_бази_даних
DB_USER=користувач_бази_даних
DB_PASSWORD=пароль_бази_даних
CHUNK_SIZE=10000
UPDATE_CHUNK=10000
DAYS_BACK=10
```

### Запуск проекту на сервері

#### Ручний запуск

1. **Активація віртуального середовища**:
```bash
cd ~/projects/data_science
source venv/bin/activate
```

2. **Запуск аналізу**:
```bash
# Для щоденного аналізу (поточний тиждень)
python main_daily.py --period week --value 5 --year 2025

# Для тижневого аналізу (конкретний тиждень)
python main_weekly.py --week 5 --year 2025

# Для місячного аналізу
python main_monthly.py --month 2 --year 2025
```

3. **Після завершення деактивуйте віртуальне середовище**:
```bash
deactivate
```

#### Запуск у фоновому режимі

Для довготривалих операцій, які повинні продовжуватись після закриття SSH сесії:

```bash
# Створення директорії для логів
mkdir -p logs

# Запуск процесу у фоновому режимі
nohup python main_daily.py --period week --value $(date +%V) --year $(date +%Y) > logs/nohup.log 2>&1 &
```

Перевірка статусу процесу:
```bash
ps aux | grep python
```

Перегляд логів:
```bash
tail -f logs/nohup.log
```

#### Автоматизація запусків через cron

1. **Відкрийте редактор crontab**:
```bash
crontab -e
```

2. **Додайте рядок для автоматичного запуску** (наприклад, щодня о 2:00):
```
0 2 * * * cd ~/projects/data_science && source venv/bin/activate && python main.py --period week --value $(date +\%V) --year $(date +\%Y) >> logs/cron_$(date +\%Y-\%m-\%d).log 2>&1 && deactivate
```

## Використання

### Аналіз щоденних даних:
```bash
python main_daily.py --period week --value 5 --year 2025
```

### Аналіз тижневих даних:
```bash
python main_weekly.py --week 5 --year 2025
```

### Аналіз місячних даних:
```bash
python main_monthly.py --month 2 --year 2025
```

### Аналіз підмножини пошукових термінів:
```bash
# Для щоденних даних
python main_daily.py --period week --value 5 --year 2025 --search_range 100:120

# Для тижневих даних
python main_weekly.py --week 5 --year 2025 --search_range 100:120

# Для місячних даних
python main_monthly.py --month 2 --year 2025 --search_range 100:120
```

### Аналіз конкретного списку пошукових термінів за ID:
```bash
# Для тижневих даних
python main_weekly.py --week 5 --year 2025 --list_ids "2108,2316,2596"

# Для місячних даних
python main_monthly.py --month 2 --year 2025 --list_ids "2108,2316,2596"
```

### Запуск паралельно 3 ядер процесора :
```bash
# Для щоденних даних
python main_daily.py --period week --value 5 --year 2025 --parallel --workers 3

## Параметри командного рядка

### Для main_daily.py:
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
- `--parallel : для паралельного запуску ядер процесора 
- `--workers : кількість ядер 

### Для main_weekly.py:
- `--week`: Номер тижня для аналізу (1-53)
- `--year`: Рік для аналізу
- `--search_range`: Діапазон пошукових термінів у форматі "start:end"
- `--list_ids`: Список ID пошукових запитів для аналізу (наприклад, "2108,2316,2596")
- `--chunk_size`: Розмір чанка для запитів до бази даних
- `--update_chunk`: Розмір чанка для оновлення бази даних
- `--env`: Середовище для запуску (development, production)
- Всі параметри підключення до бази даних (як в main.py)

### Для main_monthly.py:
- `--month`: Номер місяця для аналізу (1-12)
- `--year`: Рік для аналізу
- `--search_range`: Діапазон пошукових термінів у форматі "start:end"
- `--list_ids`: Список ID пошукових запитів для аналізу (наприклад, "2108,2316,2596")
- `--chunk_size`: Розмір чанка для запитів до бази даних
- `--update_chunk`: Розмір чанка для оновлення бази даних
- `--env`: Середовище для запуску (development, production)
- Всі параметри підключення до бази даних (як в main.py)

## Структура проекту

### Головні модулі:
- `main_daily.py`: Головний скрипт для запуску щоденного аналізу
- `main_weekly.py`: Скрипт для аналізу тижневих даних
- `main_monthly.py`: Скрипт для аналізу місячних даних
- `get_id_terms.py`: Отримання унікальних пошукових термінів
- `columns.py`: Визначення колонок та їхнього мапінгу
- `logger.py`: Налаштування логування
- `config.py`: Конфігурація для різних середовищ
- `seasonal_factor.py`: Розрахунок сезонного фактора

### Модуль week/ (Тижневі дані):
- `get_weekly_data.py`: Отримання і агрегація щоденних даних за тиждень
- `get_weekly_shares.py`: Отримання часток кліків та конверсій з тижневої таблиці
- `weekly_conversions_clicks.py`: Аналіз та розрахунок тижневих конверсій та кліків
- `update_weekly_database.py`: Оновлення тижневої таблиці результатами аналізу

### Модуль month/ (Місячні дані):
- `get_monthly_data.py`: Отримання і агрегація щоденних даних за місяць
- `get_monthly_shares.py`: Отримання часток кліків та конверсій з місячної таблиці
- `monthly_conversions_clicks.py`: Аналіз та розрахунок місячних конверсій та кліків
- `update_monthly_database.py`: Оновлення місячної таблиці результатами аналізу

### Інші модулі:
- `get_data.py`: Завантаження даних з бази
- `conversions_clicks.py`: Аналіз конверсій та кліків
- `adjust_conv_click.py`: Коригування та обробка даних про кліки та замовлення
- `update_database.py`: Оновлення бази даних результатами аналізу

## Логування

Логи зберігаються у папці `logs/` з іменем файлу, що включає поточну дату.

## Робочий процес аналізу даних

1. **Щоденні дані**:
   - Використовуйте `main_daily.py` для аналізу щоденних даних

2. **Тижневі дані**:
   - Спочатку впевніться, що щоденні дані за відповідний тиждень вже оброблені
   - Запустіть `main_weekly.py` для розрахунку кліків та конверсій за тиждень
   - Результати будуть записані в таблицю `ad_amz_search_term_weekly_data`

3. **Місячні дані**:
   - Спочатку впевніться, що щоденні дані за відповідний місяць вже оброблені
   - Запустіть `main_monthly.py` для розрахунку кліків та конверсій за місяць
   - Результати будуть записані в таблицю `ad_amz_search_term_monthly_data`

## Вирішення проблем

### Проблеми з підключенням до бази даних

Якщо виникають проблеми з підключенням до БД:

1. Перевірте правильність параметрів у `.env` файлі
2. Переконайтеся, що сервер БД доступний з вашого сервера
3. Перевірте підключення вручну:
   ```python
   python3 -c "from sqlalchemy import create_engine; from config import get_config; cfg = get_config()['db']; print(f\"Підключення до: {cfg['host']}:{cfg['port']}/{cfg['database']}\"); engine = create_engine(f\"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}\"); conn = engine.connect(); print('Підключення успішне!'); conn.close()"
   ```

### Помилки при запуску скриптів

1. Переконайтеся, що віртуальне середовище активоване
2. Перевірте, чи всі залежності встановлені: `pip list`
3. Перевірте логи на наявність детальних помилок: `cat logs/*.log`

