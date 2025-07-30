import os
import logging
import asyncio
import json
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from yandex_cloud_ml_sdk import YCloudML, AsyncYCloudML

# ====================== КОНФИГУРАЦИЯ ======================
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

# ====================== ЛОГИРОВАНИЕ ======================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ====================== КЛАСС ДЛЯ ПОИСКА ПО ДОКУМЕНТАМ ======================
class DocumentSearch:
    def __init__(self):
        self.sdk = YCloudML(folder_id=FOLDER_ID, auth=API_KEY)
        self.documents = []
        self.embeddings = []
        self.load_documents()
        
    def load_documents(self):
        """Загружаем все документы из папки documents"""
        docs_path = "documents"
        
        # Проверяем есть ли папка
        if not os.path.exists(docs_path):
            os.makedirs(docs_path)
            logger.warning(f"Создана папка {docs_path}")
            return
            
        # Загружаем все текстовые файлы
        loaded_files = []
        for filename in os.listdir(docs_path):
            if filename.endswith(('.txt', '.md')):
                try:
                    filepath = os.path.join(docs_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Разбиваем на чанки по 1000 символов
                    chunks = self.split_text(content, 1000)
                    
                    for i, chunk in enumerate(chunks):
                        self.documents.append({
                            'id': f"{filename}_{i}",
                            'filename': filename,
                            'content': chunk,
                            'chunk_index': i
                        })
                    loaded_files.append(filename)
                except Exception as e:
                    logger.error(f"Ошибка при загрузке файла {filename}: {e}")
        
        logger.info(f"Загружено файлов: {len(loaded_files)}")
        logger.info(f"Всего фрагментов: {len(self.documents)}")
        
        # Создаем embeddings для всех документов
        if self.documents:
            self.create_embeddings()
    
    def split_text(self, text: str, chunk_size: int) -> List[str]:
        """Разбиваем текст на части с учетом слов"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1
            if current_size > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def create_embeddings(self):
        """Создаем векторные представления документов"""
        logger.info(f"Создаем embeddings для {len(self.documents)} фрагментов...")
        
        for i, doc in enumerate(self.documents):
            try:
                # Используем YandexGPT для создания embeddings
                response = self.sdk.models.text_embeddings(
                    model="text-search-doc/latest",
                    text=doc['content']
                )
                self.embeddings.append(response.embedding)
                
                if i % 10 == 0:
                    logger.info(f"Обработано {i}/{len(self.documents)} фрагментов")
                    
            except Exception as e:
                logger.error(f"Ошибка при создании embedding для фрагмента {i}: {e}")
                # Заглушка на случай ошибки
                self.embeddings.append([0.0] * 256)
        
        self.embeddings = np.array(self.embeddings)
        logger.info("Embeddings созданы!")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Поиск релевантных документов по запросу"""
        if not self.documents:
            logger.warning("Нет загруженных документов для поиска")
            return []
        
        try:
            # Создаем embedding для запроса
            query_response = self.sdk.models.text_embeddings(
                model="text-search-query/latest",
                text=query
            )
            query_embedding = np.array(query_response.embedding)
            
            # Вычисляем косинусное сходство
            similarities = np.dot(self.embeddings, query_embedding)
            norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            # Избегаем деления на ноль
            norms[norms == 0] = 1
            similarities = similarities / norms
            
            # Находим top_k самых похожих
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Понизил порог релевантности
                    results.append({
                        'document': self.documents[idx],
                        'score': float(similarities[idx])
                    })
            
            logger.info(f"Найдено {len(results)} релевантных фрагментов для запроса: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            return []

# ====================== ИНИЦИАЛИЗАЦИЯ ======================
# Инициализация YandexGPT
sdk_async = AsyncYCloudML(folder_id=FOLDER_ID, auth=API_KEY)

# Глобальный экземпляр для поиска
searcher = DocumentSearch()

# ====================== ОБРАБОТЧИКИ КОМАНД ======================
async def start(update: Update, context):
    """Обработчик команды /start"""
    docs_count = len(searcher.documents)
    files_count = len(set(doc['filename'] for doc in searcher.documents))
    
    await update.message.reply_text(
        "👋 Привет! Я бот-помощник по нормативным документам ППРФ 442.\n\n"
        "📚 Могу ответить на вопросы по:\n"
        "• Коммерческому учету электроэнергии\n"
        "• Установке и замене приборов учета\n"
        "• Срокам и ответственности\n"
        "• Проверкам и контролю\n"
        "• Безучетному потреблению\n\n"
        f"📄 Загружено: {files_count} файлов ({docs_count} фрагментов)\n\n"
        "❓ Просто задайте ваш вопрос!"
    )

async def list_docs(update: Update, context):
    """Показать список загруженных документов"""
    if not searcher.documents:
        await update.message.reply_text("📂 Документы пока не загружены.")
        return
    
    files = set(doc['filename'] for doc in searcher.documents)
    file_list = "\n".join([f"• {filename}" for filename in sorted(files)])
    
    await update.message.reply_text(
        f"📚 Загруженные документы:\n\n{file_list}\n\n"
        f"Всего фрагментов: {len(searcher.documents)}"
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
        search_results = searcher.search(user_message, top_k=5)  # Увеличил количество результатов
        
        # Формируем контекст из найденных документов
        context_text = ""
        sources = []
        
        if search_results:
            logger.info(f"Используем {len(search_results)} фрагментов для ответа")
            context_parts = []
            
            for result in search_results:
                doc = result['document']
                score = result['score']
                context_parts.append(f"[Релевантность: {score:.2f}] Из файла {doc['filename']}:\n{doc['content']}")
                sources.append(doc['filename'])
            
            context_text = "\n\n---\n\n".join(context_parts)
            sources = list(set(sources))  # Уникальные источники
        
        # Формируем промпт для YandexGPT
        system_prompt = """Ты - эксперт по Постановлению Правительства РФ №442 "О функционировании розничных рынков электрической энергии".
        
Твоя задача - давать точные и полезные ответы на основе предоставленных фрагментов документов.

Правила ответа:
1. Используй ТОЛЬКО информацию из предоставленных документов
2. Если информации недостаточно, честно скажи об этом
3. Всегда указывай конкретные пункты ППРФ 442, если они упоминаются в документах
4. Структурируй ответ: используй списки, выделение важного
5. Будь конкретен и практичен в рекомендациях"""
        
        messages = [{"role": "system", "text": system_prompt}]
        
        if context_text:
            user_prompt = f"""Найденная информация из документов:

{context_text}

Вопрос пользователя: {user_message}

Дай структурированный ответ на основе найденной информации."""
            messages.append({"role": "user", "text": user_prompt})
        else:
            messages.append({
                "role": "user", 
                "text": f"Вопрос: {user_message}\n\nК сожалению, в загруженных документах не найдена релевантная информация. Что ты можешь сказать по этому вопросу на основе общих знаний о ППРФ 442?"
            })
        
        # Запрос к YandexGPT
        result = await sdk_async.models.completions(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        response_text = result.alternatives[0].text
        
        # Добавляем информацию об источниках
        if sources:
            sources_text = "\n\n📎 _Источники: " + ", ".join(sources) + "_"
            response_text += sources_text
        elif not search_results:
            response_text += "\n\n⚠️ _Ответ основан на общих знаниях, так как в документах информация не найдена_"
        
        await update.message.reply_text(response_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}", exc_info=True)
        await update.message.reply_text(
            "😔 Извините, произошла ошибка при обработке вашего запроса.\n"
            "Попробуйте переформулировать вопрос или обратитесь позже."
        )

async def reload_docs(update: Update, context):
    """Перезагрузить документы"""
    await update.message.reply_text("🔄 Перезагружаю документы...")
    
    try:
        global searcher
        searcher = DocumentSearch()
        
        docs_count = len(searcher.documents)
        files_count = len(set(doc['filename'] for doc in searcher.documents))
        
        await update.message.reply_text(
            f"✅ Документы перезагружены!\n"
            f"Загружено: {files_count} файлов ({docs_count} фрагментов)"
        )
    except Exception as e:
        logger.error(f"Ошибка при перезагрузке документов: {e}")
        await update.message.reply_text("❌ Ошибка при перезагрузке документов")

# ====================== ГЛАВНАЯ ФУНКЦИЯ ======================
def main():
    """Запуск бота"""
    # Проверяем наличие необходимых переменных
    if not all([BOT_TOKEN, FOLDER_ID, API_KEY]):
        logger.error("Не заданы необходимые переменные окружения!")
        logger.error("Проверьте BOT_TOKEN, FOLDER_ID, YANDEX_API_KEY")
        return
    
    # Создаем приложение
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Регистрируем обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("docs", list_docs))
    application.add_handler(CommandHandler("reload", reload_docs))
    
    # Регистрируем обработчик сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем бота
    logger.info(f"🤖 Бот запущен! Загружено {len(searcher.documents)} фрагментов документов")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
