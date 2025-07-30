import os
from dotenv import load_dotenv

# Загружаем переменные окружения для локальной разработки
load_dotenv()

# Telegram
BOT_TOKEN = os.environ.get('BOT_TOKEN')

# Yandex Cloud
FOLDER_ID = os.environ.get('FOLDER_ID')
API_KEY = os.environ.get('YANDEX_API_KEY')

# Настройки модели
MODEL = "yandexgpt-lite"
TEMPERATURE = 0.3
MAX_TOKENS = 2000
