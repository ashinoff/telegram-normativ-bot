import os
import json
import numpy as np
from typing import List, Dict
from yandex_cloud_ml_sdk import YCloudML
import config

class DocumentSearch:
    def __init__(self):
        self.sdk = YCloudML(folder_id=config.FOLDER_ID, auth=config.API_KEY)
        self.documents = []
        self.embeddings = []
        self.load_documents()
        
    def load_documents(self):
        """Загружаем все документы из папки documents"""
        docs_path = "documents"
        
        # Проверяем есть ли папка
        if not os.path.exists(docs_path):
            os.makedirs(docs_path)
            return
            
        for filename in os.listdir(docs_path):
            if filename.endswith(('.txt', '.md')):
                with open(os.path.join(docs_path, filename), 'r', encoding='utf-8') as f:
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
        
        # Создаем embeddings для всех документов
        if self.documents:
            self.create_embeddings()
    
    def split_text(self, text: str, chunk_size: int) -> List[str]:
        """Разбиваем текст на части"""
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
        print(f"Создаем embeddings для {len(self.documents)} фрагментов...")
        
        for doc in self.documents:
            try:
                # Используем YandexGPT для создания embeddings
                response = self.sdk.models.text_embeddings(
                    model="text-search-doc/latest",
                    text=doc['content']
                )
                self.embeddings.append(response.embedding)
            except Exception as e:
                print(f"Ошибка при создании embedding: {e}")
                # Заглушка на случай ошибки
                self.embeddings.append([0.0] * 256)
        
        self.embeddings = np.array(self.embeddings)
        print("Embeddings созданы!")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Поиск релевантных документов"""
        if not self.documents:
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
            similarities = similarities / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding))
            
            # Находим top_k самых похожих
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.5:  # Порог релевантности
                    results.append({
                        'document': self.documents[idx],
                        'score': float(similarities[idx])
                    })
            
            return results
            
        except Exception as e:
            print(f"Ошибка при поиске: {e}")
            return []

# Глобальный экземпляр для использования в боте
searcher = DocumentSearch()
