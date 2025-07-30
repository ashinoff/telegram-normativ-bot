import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from yandex_cloud_ml_sdk import AsyncYCloudML
import config
from document_search import searcher

# Логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Инициализация YandexGPT
sdk = AsyncYCloudML(folder_id=config.FOLDER_ID, auth=config.API_KEY)

async def start(update: Update, context):
    """Обработчик команды /start"""
    await update.message.reply_text(
        "👋 Привет! Я бот-помощник по нормативным документам.\n\n"
        "📚 Задавайте любые вопросы, и я помогу найти ответы!\n\n"
        f"📄 Загружено документов: {len(searcher.documents)}"
    )

async def handle_message(update: Update, context):
    """Обработчик текстовых сообщений"""
    user_message = update.message.text
    
    # Показываем, что бот печатает
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, 
        action="typing"
    )
    
    try:
        # Ищем релевантные документы
        search_results = searcher.search(user_message, top_k=3)
        
        # Формируем контекст из найденных документов
        context_text = ""
        if search_results:
            context_text = "Найденная информация из документов:\n\n"
            for result in search_results:
                doc = result['document']
                context_text += f"Из файла {doc['filename']}:\n{doc['content']}\n\n"
        
        # Формируем промпт для YandexGPT
        system_prompt = """Ты - эксперт по нормативным документам. 
        Отвечай на основе предоставленной информации из документов.
        Если информации недостаточно, скажи об этом.
        Всегда указывай источник информации (название документа)."""
        
        messages = [
            {"role": "system", "text": system_prompt}
        ]
        
        if context_text:
            messages.append({"role": "user", "text": f"{context_text}\n\nВопрос пользователя: {user_message}"})
        else:
            messages.append({"role": "user", "text": f"Вопрос пользователя: {user_message}\n\nВ базе документов информация не найдена."})
        
        # Запрос к YandexGPT
        result = await sdk.models.completions(
            model=config.MODEL,
            messages=messages,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
        
        response_text = result.alternatives[0].text
        
        # Добавляем информацию о поиске
        if search_results:
            response_text += f"\n\n📎 _Найдено совпадений: {len(search_results)}_"
        
        await update.message.reply_text(response_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        await update.message.reply_text(
            "😔 Извините, произошла ошибка. Попробуйте позже."
        )

async def list_docs(update: Update, context):
    """Показать список загруженных документов"""
    if not searcher.documents:
        await update.message.reply_text("📂 Документы пока не загружены.")
        return
    
    files = set(doc['filename'] for doc in searcher.documents)
    file_list = "\n".join([f"• {filename}" for filename in files])
    
    await update.message.reply_text(
        f"📚 Загруженные документы:\n\n{file_list}\n\n"
        f"Всего фрагментов: {len(searcher.documents)}"
    )

def main():
    """Запуск бота"""
    # Создаем приложение
    application = Application.builder().token(config.BOT_TOKEN).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("docs", list_docs))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем бота
    logger.info(f"🤖 Бот запущен! Загружено {len(searcher.documents)} фрагментов документов")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
