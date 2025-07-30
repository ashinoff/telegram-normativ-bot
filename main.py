import os
import logging
import asyncio
from typing import List, Dict
from dotenv import load_dotenv
from aiohttp import web
import threading
import aiohttp

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from yandex_cloud_ml_sdk import AsyncYCloudML

# ====================== КОНФИГУРАЦИЯ ======================
load_dotenv()

BOT_TOKEN = os.environ.get('BOT_TOKEN')
FOLDER_ID = os.environ.get('FOLDER_ID')
API_KEY = os.environ.get('YANDEX_API_KEY')
MODEL_URI = f"gpt://{FOLDER_ID}/yandexgpt-lite/latest"
TEMPERATURE = 0.3
MAX_TOKENS = 2000

# Выбор метода работы с YandexGPT (SDK или HTTP)
USE_SDK = True  # Поставьте False, если SDK не работает

# ====================== БЕЗОПАСНАЯ НАСТРОЙКА ЛОГИРОВАНИЯ ======================
# Настраиваем основной логгер
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ВАЖНО: Отключаем логирование httpx, чтобы не показывать токены
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram.ext._application").setLevel(logging.WARNING)

# Альтернативный вариант - создать кастомный фильтр для логов
class TokenFilter(logging.Filter):
    """Фильтр для скрытия токенов в логах"""
    def __init__(self, token):
        super().__init__()
        self.token = token
        self.masked_token = token[:6] + "..." + token[-4:] if token else ""
    
    def filter(self, record):
        if self.token and hasattr(record, 'msg'):
            record.msg = str(record.msg).replace(self.token, self.masked_token)
        return True

# Применяем фильтр ко всем хендлерам
if BOT_TOKEN:
    token_filter = TokenFilter(BOT_TOKEN)
    for handler in logging.root.handlers:
        handler.addFilter(token_filter)

# Создаем безопасный логгер для отладки Telegram
telegram_logger = logging.getLogger('telegram')
telegram_logger.setLevel(logging.WARNING)  # Только предупреждения и ошибки

# ====================== ПРОСТОЙ ПОИСК БЕЗ EMBEDDINGS ======================
class SimpleDocumentSearch:
    def __init__(self):
        self.documents = []
        self.load_documents()
        
    def load_documents(self):
        """Загружаем все документы из папки documents"""
        docs_path = "documents"
        
        if not os.path.exists(docs_path):
            os.makedirs(docs_path)
            logger.warning(f"Создана папка {docs_path}")
            return
            
        # Загружаем все текстовые файлы целиком
        for filename in os.listdir(docs_path):
            if filename.endswith(('.txt', '.md')):
                try:
                    filepath = os.path.join(docs_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    self.documents.append({
                        'filename': filename,
                        'content': content
                    })
                    logger.info(f"Загружен файл: {filename}")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке файла {filename}: {e}")
        
        logger.info(f"Загружено документов: {len(self.documents)}")
    
    def search(self, query: str) -> List[Dict]:
        """Простой поиск по ключевым словам"""
        if not self.documents:
            return []
        
        # Ключевые слова из запроса
        keywords = query.lower().split()
        
        results = []
        for doc in self.documents:
            content_lower = doc['content'].lower()
            
            # Подсчитываем количество найденных ключевых слов
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            
            if matches > 0:
                # Находим релевантные фрагменты
                fragments = self.extract_relevant_fragments(doc['content'], keywords)
                if fragments:
                    results.append({
                        'filename': doc['filename'],
                        'content': fragments,
                        'matches': matches
                    })
        
        # Сортируем по количеству совпадений
        results.sort(key=lambda x: x['matches'], reverse=True)
        return results[:3]  # Возвращаем топ-3
    
    def extract_relevant_fragments(self, text: str, keywords: List[str], context_size: int = 300) -> str:
        """Извлекаем фрагменты текста вокруг ключевых слов"""
        sentences = text.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence.strip())
        
        # Объединяем и ограничиваем размер
        result = '. '.join(relevant_sentences[:5])
        if len(result) > 2000:
            result = result[:2000] + '...'
        
        return result

# ====================== YANDEXGPT HELPERS ======================
async def call_yandexgpt_http(messages: List[Dict], temperature: float, max_tokens: int) -> str:
    """Альтернативный метод вызова YandexGPT через HTTP API"""
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    
    # Маскируем API ключ в логах
    masked_api_key = API_KEY[:8] + "..." + API_KEY[-4:] if API_KEY else "not_set"
    logger.debug(f"Вызов YandexGPT API (ключ: {masked_api_key})")
    
    headers = {
        "Authorization": f"Api-Key {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "modelUri": MODEL_URI,
        "completionOptions": {
            "stream": False,
            "temperature": temperature,
            "maxTokens": max_tokens
        },
        "messages": messages
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"YandexGPT API error: {response.status} - {error_text}")
            
            result = await response.json()
            return result["result"]["alternatives"][0]["message"]["text"]

async def call_yandexgpt_sdk(sdk: AsyncYCloudML, messages: List[Dict], temperature: float, max_tokens: int) -> str:
    """Вызов YandexGPT через SDK"""
    try:
        # Создаем модель
        model = sdk.models.completions(MODEL_URI)
        
        # Выполняем запрос
        result = await model.run(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return result.alternatives[0].text
    except Exception as e:
        logger.error(f"SDK error: {e}")
        # Если SDK не работает, пробуем через HTTP
        logger.info("Переключаемся на HTTP метод...")
        return await call_yandexgpt_http(messages, temperature, max_tokens)

# ====================== ИНИЦИАЛИЗАЦИЯ ======================
sdk_async = AsyncYCloudML(folder_id=FOLDER_ID, auth=API_KEY) if USE_SDK else None
searcher = SimpleDocumentSearch()

# ====================== HEALTH CHECK SERVER ======================
async def health_check(request):
    """Простой health check endpoint для Render"""
    return web.Response(text="Bot is running", status=200)

def run_health_server():
    """Запускаем простой веб-сервер для health checks"""
    app = web.Application()
    app.router.add_get('/', health_check)
    app.router.add_get('/health', health_check)
    
    # Получаем порт из переменной окружения (Render устанавливает PORT)
    port = int(os.environ.get('PORT', 10000))
    
    try:
        # Отключаем access логи aiohttp чтобы не спамить
        access_logger = logging.getLogger('aiohttp.access')
        access_logger.setLevel(logging.WARNING)
        
        web.run_app(app, host='0.0.0.0', port=port, print=lambda _: None, access_log=None)
    except Exception as e:
        logger.error(f"Health server error: {e}")

# ====================== ОБРАБОТЧИКИ ======================
async def start(update: Update, context):
    """Обработчик команды /start"""
    docs_count = len(searcher.documents)
    user_name = update.effective_user.first_name or "Друг"
    
    await update.message.reply_text(
        f"👋 Привет, {user_name}! Я бот-помощник по нормативным документам ППРФ 442.\n\n"
        "📚 Могу ответить на вопросы по:\n"
        "• Коммерческому учету электроэнергии\n"
        "• Установке и замене приборов учета\n"
        "• Срокам и ответственности\n"
        "• Проверкам и контролю\n"
        "• Безучетному потреблению\n\n"
        f"📄 Загружено документов: {docs_count}\n\n"
        "❓ Просто задайте ваш вопрос!"
    )
    
    # Логируем без показа чувствительных данных
    logger.info(f"Пользователь {user_name} (ID: {update.effective_user.id}) запустил бота")

async def list_docs(update: Update, context):
    """Показать список загруженных документов"""
    if not searcher.documents:
        await update.message.reply_text("📂 Документы пока не загружены.")
        return
    
    file_list = "\n".join([f"• {doc['filename']}" for doc in searcher.documents])
    
    await update.message.reply_text(
        f"📚 Загруженные документы:\n\n{file_list}"
    )

async def reload_docs(update: Update, context):
    """Перезагрузить документы"""
    user_id = update.effective_user.id
    
    # Можно добавить проверку прав доступа
    # ADMIN_IDS = [123456789]  # Список ID администраторов
    # if user_id not in ADMIN_IDS:
    #     await update.message.reply_text("⛔ У вас нет прав для этой команды")
    #     return
    
    await update.message.reply_text("🔄 Перезагружаю документы...")
    
    searcher.documents = []
    searcher.load_documents()
    
    await update.message.reply_text(
        f"✅ Документы перезагружены!\n"
        f"📄 Загружено: {len(searcher.documents)} файлов"
    )
    
    logger.info(f"Пользователь ID:{user_id} перезагрузил документы")

async def handle_message(update: Update, context):
    """Обработчик текстовых сообщений"""
    user_message = update.message.text
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name or "Пользователь"
    
    # Логируем запрос без показа содержимого (для приватности)
    logger.info(f"Получен запрос от {user_name} (ID: {user_id}), длина: {len(user_message)} символов")
    
    # Показываем, что бот печатает
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, 
        action="typing"
    )
    
    try:
        # Простой поиск по документам
        search_results = searcher.search(user_message)
        
        # Формируем контекст
        context_text = ""
        sources = []
        
        if search_results:
            logger.info(f"Найдено {len(search_results)} документов")
            for result in search_results:
                context_text += f"\nИз файла {result['filename']}:\n{result['content']}\n"
                sources.append(result['filename'])
        else:
            logger.info("Релевантные документы не найдены")
        
        # Промпт для YandexGPT
        system_prompt = """Ты - эксперт по постановлению правительства РФ №442 (ППРФ 442).
Отвечай на основе предоставленных документов.
Если информации недостаточно, скажи об этом.
Структурируй ответы, указывай конкретные пункты ППРФ.
Будь дружелюбным и полезным."""
        
        messages = [{"role": "system", "text": system_prompt}]
        
        if context_text:
            prompt = f"Документы:\n{context_text}\n\nВопрос от {user_name}: {user_message}"
        else:
            prompt = (f"Вопрос от {user_name}: {user_message}\n\n"
                     "В загруженных документах информация не найдена. "
                     "Ответь на основе общих знаний о ППРФ 442, если возможно.")
        
        messages.append({"role": "user", "text": prompt})
        
        # Вызов YandexGPT
        if USE_SDK and sdk_async:
            response_text = await call_yandexgpt_sdk(sdk_async, messages, TEMPERATURE, MAX_TOKENS)
        else:
            response_text = await call_yandexgpt_http(messages, TEMPERATURE, MAX_TOKENS)
        
        # Добавляем источники
        if sources:
            response_text += f"\n\n📎 _Источники: {', '.join(set(sources))}_"
        
        # Отправляем ответ
        await update.message.reply_text(response_text, parse_mode='Markdown')
        
        # Логируем успешный ответ
        logger.info(f"Успешно отправлен ответ пользователю ID:{user_id}")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения от ID:{user_id}: {type(e).__name__}: {str(e)}")
        
        error_message = (
            "😔 Произошла ошибка при обработке вашего запроса.\n\n"
            "Возможные причины:\n"
            "• Проблема с подключением к YandexGPT\n"
            "• Слишком длинный запрос\n"
            "• Временные технические проблемы\n\n"
            "Попробуйте:\n"
            "• Переформулировать вопрос короче\n"
            "• Повторить попытку через минуту"
        )
        
        await update.message.reply_text(error_message)

# ====================== ГЛАВНАЯ ФУНКЦИЯ ======================
def main():
    """Запуск бота"""
    # Проверка переменных окружения
    if not all([BOT_TOKEN, FOLDER_ID, API_KEY]):
        logger.error("Не заданы необходимые переменные окружения!")
        logger.error("Требуются: BOT_TOKEN, FOLDER_ID, YANDEX_API_KEY")
        return
    
    # Показываем маскированный токен для отладки
    masked_token = BOT_TOKEN[:6] + "..." + BOT_TOKEN[-4:] if BOT_TOKEN else "not_set"
    logger.info(f"Используется бот с токеном: {masked_token}")
    
    # Запускаем health check сервер в отдельном потоке (для Render)
    if os.environ.get('PORT'):
        health_thread = threading.Thread(target=run_health_server, daemon=True)
        health_thread.start()
        logger.info("🌐 Health check сервер запущен")
    
    # Создаем приложение
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("docs", list_docs))
    application.add_handler(CommandHandler("reload", reload_docs))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем бота
    logger.info(f"🤖 Бот запущен!")
    logger.info(f"📄 Загружено документов: {len(searcher.documents)}")
    logger.info(f"🔧 Метод API: {'SDK' if USE_SDK else 'HTTP'}")
    
    # Запускаем polling
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True  # Игнорируем старые сообщения
    )

if __name__ == '__main__':
    main()
