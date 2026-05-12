import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearch:
    def __init__(self, dataset_path, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        print(f"🚀 Loading Search Engine (Model: {model_name})...")
        self.model = SentenceTransformer(model_name)
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        print(f"📊 Indexing {len(self.data)} transcript segments...")
        self.texts = [item['text'] for item in self.data]
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True)
        print("✅ Indexing complete!")

    def search(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top K indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.data[idx],
                "score": float(similarities[idx])
            })
        return results

def main():
    searcher = SemanticSearch("youtube_transcript.json")
    
    while True:
        print("\n" + "="*50)
        query = input("🔍 Enter your Arabic search query (or 'exit' to quit): ")
        if query.lower() in ['exit', 'quit', 'q', 'خروج']:
            break
            
        results = searcher.search(query)
        
        print(f"\n🎯 Top Results for: '{query}'")
        for i, res in enumerate(results):
            chunk = res['chunk']
            print(f"\n[{i+1}] Score: {res['score']:.4f}")
            print(f"⏱️ Time: {chunk['start_time']}s -> {chunk['end_time']}s")
            print(f"📝 Text: {chunk['text']}")
            print(f"🔗 Link: https://www.youtube.com/watch?v={chunk['video_id']}&t={int(chunk['start_time'])}s")

if __name__ == "__main__":
    main()
