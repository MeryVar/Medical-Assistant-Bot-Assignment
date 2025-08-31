import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    question: str
    answer: str
    similarity_score: float
    urgency_level: str
    medical_category: str
    confidence: float

class DatasetRetrievalAssistant:
    
    def __init__(self, dataset_path: str = "mle_screening_dataset.csv"):
        self.dataset_path = dataset_path
        self.medical_data = None
        self.processed_data = None
        self._load_dataset()
        self._preprocess_dataset()
        
    def _load_dataset(self):
        try:
            print(f" Loading medical dataset from {self.dataset_path}...")
            self.medical_data = pd.read_csv(self.dataset_path)
            print(f" Loaded {len(self.medical_data)} medical Q&A pairs")
            
            print(" Sample entries:")
            for i, row in self.medical_data.head(3).iterrows():
                print(f"   Q: {row['question'][:60]}...")
                print(f"   A: {row['answer'][:100]}...")
                print()
                
        except Exception as e:
            print(f" Error loading dataset: {e}")
            raise
    
    def _preprocess_dataset(self):
        print(" Preprocessing dataset for retrieval...")
        
        self.processed_data = self.medical_data.copy()
        
        self.processed_data['question_clean'] = self.processed_data['question'].str.lower().str.strip()
        self.processed_data['answer_clean'] = self.processed_data['answer'].str.lower()
        
        self.processed_data['medical_category'] = self.processed_data.apply(
            self._extract_medical_category, axis=1
        )
        
        self.processed_data['urgency_level'] = self.processed_data.apply(
            self._extract_urgency_level, axis=1
        )
        
        self.processed_data['keywords'] = self.processed_data.apply(
            self._extract_keywords, axis=1
        )
        
        print(f" Preprocessed {len(self.processed_data)} entries for retrieval")
        
    def _extract_medical_category(self, row) -> str:
        text = (str(row['question']) + " " + str(row['answer'])).lower()
        
        categories = {
            'cardiovascular': ['heart', 'blood pressure', 'cardiac', 'hypertension', 'chest pain'],
            'respiratory': ['lung', 'breathing', 'asthma', 'cough', 'respiratory'],
            'neurological': ['brain', 'nerve', 'headache', 'seizure', 'stroke'],
            'gastrointestinal': ['stomach', 'digestive', 'liver', 'intestine', 'bowel'],
            'endocrine': ['diabetes', 'thyroid', 'hormone', 'insulin'],
            'musculoskeletal': ['bone', 'joint', 'muscle', 'arthritis'],
            'dermatological': ['skin', 'rash', 'dermatitis'],
            'urological': ['urinary', 'kidney', 'bladder', 'incontinence'],
            'oncological': ['cancer', 'tumor', 'malignant'],
            'ophthalmological': ['eye', 'vision', 'glaucoma', 'blind'],
            'infectious': ['infection', 'virus', 'bacteria'],
            'mental_health': ['depression', 'anxiety', 'mental'],
            'pediatric': ['child', 'infant', 'pediatric'],
            'general': ['symptom', 'disease', 'condition', 'treatment']
        }
        
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return 'general'
    
    def _extract_urgency_level(self, row) -> str:
        text = (str(row['question']) + " " + str(row['answer'])).lower()
        
        if any(word in text for word in ['emergency', 'immediate', 'urgent', 'severe', 'critical']):
            return 'high'
        elif any(word in text for word in ['treatment', 'therapy', 'consult', 'see doctor']):
            return 'medium'
        else:
            return 'low'
    
    def _extract_keywords(self, row) -> List[str]:
        text = str(row['question']).lower()
        
        text = re.sub(r'\b(what|is|are|how|why|when|where|who|can|should|do|does|will)\b', '', text)
        
        words = re.findall(r'\b[a-z]{3,}\b', text)
        
        return list(set(words))
    
    def _calculate_similarity(self, user_query: str, dataset_entry: pd.Series) -> float:
        user_query_lower = user_query.lower()
        question_lower = str(dataset_entry['question']).lower()
        answer_lower = str(dataset_entry['answer']).lower()
        
        if user_query_lower in question_lower or question_lower in user_query_lower:
            return 0.95
        
        user_words = set(re.findall(r'\b[a-z]{3,}\b', user_query_lower))
        question_words = set(re.findall(r'\b[a-z]{3,}\b', question_lower))
        answer_words = set(re.findall(r'\b[a-z]{3,}\b', answer_lower))
        
        if not user_words:
            return 0.0
        
        question_overlap = len(user_words & question_words) / len(user_words)
        answer_overlap = len(user_words & answer_words) / len(user_words | answer_words)
        
        similarity = 0.7 * question_overlap + 0.3 * answer_overlap
        
        return similarity
    
    def search_dataset(self, user_query: str, top_k: int = 3) -> List[RetrievalResult]:
        print(f" Searching dataset for: '{user_query}'")
        
        similarities = []
        for idx, row in self.processed_data.iterrows():
            similarity = self._calculate_similarity(user_query, row)
            
            if similarity > 0.1:
                similarities.append({
                    'index': idx,
                    'similarity': similarity,
                    'question': row['question'],
                    'answer': row['answer'],
                    'medical_category': row['medical_category'],
                    'urgency_level': row['urgency_level']
                })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        results = []
        for i, match in enumerate(similarities[:top_k]):
            result = RetrievalResult(
                question=match['question'],
                answer=match['answer'],
                similarity_score=match['similarity'],
                urgency_level=match['urgency_level'],
                medical_category=match['medical_category'],
                confidence=match['similarity']
            )
            results.append(result)
        
        print(f" Found {len(results)} relevant matches")
        return results
    
    def get_best_answer(self, user_query: str) -> RetrievalResult:
        results = self.search_dataset(user_query, top_k=1)
        
        if results:
            return results[0]
        else:
            return RetrievalResult(
                question="No exact match found",
                answer="I couldn't find a specific answer to your question in the medical database. Please consult with a healthcare provider for personalized medical advice.",
                similarity_score=0.0,
                urgency_level='medium',
                medical_category='general',
                confidence=0.0
            )
    
    def interactive_search(self):
        print("\n MEDICAL DATASET SEARCH INTERFACE")
        print("=" * 60)
        print("Ask medical questions and get answers from the real medical dataset!")
        print("Type 'quit' to exit")
        print()
        
        while True:
            user_query = input(" Your medical question: ").strip()
            
            if not user_query:
                continue
                
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            print("\n Searching medical database...")
            results = self.search_dataset(user_query, top_k=3)
            
            if results:
                print(f"\n Found {len(results)} relevant answers:")
                print("=" * 80)
                
                for i, result in enumerate(results, 1):
                    print(f"\n RESULT {i} (Similarity: {result.similarity_score:.2f})")
                    print("─" * 60)
                    print(f" Original Question: {result.question}")
                    print(f" Medical Answer: {result.answer}")
                    print(f" Urgency: {result.urgency_level.upper()}")
                    print(f" Category: {result.medical_category}")
                    print("─" * 60)
                
                best_result = results[0]
                print(f"\n BEST ANSWER (Confidence: {best_result.confidence:.2f})")
                print("=" * 60)
                print(f"{best_result.answer}")
                print("=" * 60)
                
            else:
                print(" No relevant answers found in the medical database.")
            
            print("\nAsk another question or type 'quit' to exit.")

def main():
    try:
        assistant = DatasetRetrievalAssistant("mle_screening_dataset.csv")
        
        assistant.interactive_search()
        
    except Exception as e:
        print(f" Error: {e}")
        print("Make sure the mle_screening_dataset.csv file is in the current directory")

if __name__ == "__main__":
    main()
