import os
import logging
import asyncio
from typing import List, Dict
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from yandex_cloud_ml_sdk import AsyncYCloudML

# ====================== КОНФИГУРАЦИЯ ======================
load_dotenv()

BOT_TOKEN = os.environ.get('BOT_TOKEN')
FOLDER_ID = os.environ.get('FOLDER_ID')
API_KEY = os.environ.get('YANDEX_API_KEY')
MODEL = "yandexgpt-lite"
TEMPERATURE = 0.3
MAX_TOKENS = 2000

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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

# ====================== ИНИЦИАЛИЗАЦИЯ ======================
sdk_async = AsyncYCloudML(folder_id=FOLDER_ID, auth=API_KEY)
searcher = SimpleDocumentSearch()

# ====================== ОБРАБОТЧИКИ ======================
async def start(update: Update, context):
    """Обработчик команды /start"""
    docs_count = len(searcher.documents)
    
    await update.message.reply_text(
        "👋 Привет! Я бот-помощник по нормативным документам ППРФ 442.\n\n"
        "📚 Могу ответить на вопросы по:\n"
        "• Коммерческому учету электроэнергии\n"
        "• Установке и замене приборов учета\n"
        "• Срокам и ответственности\n"
        "• Проверкам и контролю\n"
        "• Безучетному потреблению\n\n"
        f"📄 Загружено документов: {docs_count}\n\n"
        "❓ Просто задайте ваш вопрос!"
    )

async def list_docs(update: Update, context):
    """Показать список загруженных документов"""
    if not searcher.documents:
        await update.message.reply_text("📂 Документы пока не загружены.")
        return
    
    file_list = "\n".join([f"• {doc['filename']}" for doc in searcher.documents])
    
    await update.message.reply_text(
        f"📚 Загруженные документы:\n\n{file_list}"
    )

async def handle_message(update: Update, context):
    """Обработчик текстовых сообщений"""
    user_message = update.message.text
    
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
            for result in search_results:
                context_text += f"\nИз файла {result['filename']}:\n{result['content']}\n"
                sources.append(result['filename'])
        
        # Промпт для YandexGPT
        system_prompt = """Ты - эксперт по ППРФ 442. Отвечай на основе предоставленных документов.
Если информации недостаточно, скажи об этом. Структурируй ответы, указывай пункты ППРФ."""
        
        messages = [{"role": "system", "text": system_prompt}]
        
        if context_text:
            prompt = f"Документы:\n{context_text}\n\nВопрос: {user_message}"
        else:
            prompt = f"Вопрос: {user_message}\n\nВ документах информация не найдена. Что можешь сказать из общих знаний?"
        
        messages.append({"role": "user", "text": prompt})
        
        # Запрос к YandexGPT
        result = await sdk_async.models.completions(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        response_text = result.alternatives[0].text
        
        # Добавляем источники
        if sources:
            response_text += f"\n\n📎 _Источники: {', '.join(set(sources))}_"
        
        await update.message.reply_text(response_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        await update.message.reply_text(
            "😔 Произошла ошибка. Попробуйте переформулировать вопрос."
        )

# ====================== ГЛАВНАЯ ФУНКЦИЯ ======================
def main():
    """Запуск бота"""
    if not all([BOT_TOKEN, FOLDER_ID, API_KEY]):
        logger.error("Не заданы переменные окружения!")
        return
    
    application = Application.builder().token(BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("docs", list_docs))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info(f"🤖 Бот запущен! Документов: {len(searcher.documents)}")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
