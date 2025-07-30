import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from yandex_cloud_ml_sdk import AsyncYCloudML
import config

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
        "Пример: 'Какие требования к пожарной безопасности?'"
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
        # Запрос к YandexGPT
        result = await sdk.models.completions(
            model=config.MODEL,
            messages=[
                {
                    "role": "system",
                    "text": "Ты - эксперт по нормативным документам. Давай четкие, структурированные ответы."
                },
                {
                    "role": "user",
                    "text": user_message
                }
            ],
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
        
        response_text = result.alternatives[0].text
        await update.message.reply_text(response_text)
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        await update.message.reply_text(
            "😔 Извините, произошла ошибка. Попробуйте позже."
        )

def main():
    """Запуск бота"""
    # Создаем приложение
    application = Application.builder().token(config.BOT_TOKEN).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем бота
    logger.info("🤖 Бот запущен!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
