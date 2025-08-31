import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MedicalAnswer:
    original_question: str
    medical_answer: str
    similarity_score: float
    urgency_level: str
    medical_category: str
    confidence: float
    dataset_index: int

class BERTMedicalAssistant:

    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 dataset_path: str = "mle_screening_dataset.csv",
                 cache_embeddings: bool = True):

        self.model_name = model_name
        self.dataset_path = dataset_path
        self.cache_path = "medical_embeddings_cache.pkl"
        self.cache_embeddings = cache_embeddings

        print(f" Loading BERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        print(" Loading medical dataset...")
        self.medical_data = self._load_medical_dataset()

        self.question_embeddings = self._get_or_create_embeddings()

        self.medical_categories = self._load_medical_categories()

        print(" BERT Medical Assistant initialized successfully!")
        print(f" Dataset: {len(self.medical_data)} medical Q&A pairs")
        print(f" Model: {model_name}")
        print(f" Embeddings: {self.question_embeddings.shape}")

    def _load_medical_dataset(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.dataset_path)

            df = df.dropna(subset=['question', 'answer'])
            df['question_clean'] = df['question'].str.strip()
            df['answer_clean'] = df['answer'].str.strip()

            df['medical_category'] = df.apply(self._classify_medical_category, axis=1)
            df['urgency_level'] = df.apply(self._classify_urgency, axis=1)

            print(f" Loaded and preprocessed {len(df)} medical Q&A pairs")
            return df

        except Exception as e:
            print(f" Error loading dataset: {e}")
            raise

    def _classify_medical_category(self, row) -> str:
        text = (str(row['question']) + " " + str(row['answer'])).lower()

        category_keywords = {
            'cardiovascular': ['heart', 'blood pressure', 'cardiac', 'hypertension', 'chest pain', 'cholesterol'],
            'respiratory': ['lung', 'breathing', 'asthma', 'cough', 'respiratory', 'pneumonia'],
            'neurological': ['brain', 'nerve', 'headache', 'seizure', 'stroke', 'migraine'],
            'gastrointestinal': ['stomach', 'digestive', 'liver', 'intestine', 'bowel', 'gastric'],
            'endocrine': ['diabetes', 'thyroid', 'hormone', 'insulin', 'glucose'],
            'musculoskeletal': ['bone', 'joint', 'muscle', 'arthritis', 'fracture'],
            'dermatological': ['skin', 'rash', 'dermatitis', 'eczema'],
            'urological': ['urinary', 'kidney', 'bladder', 'incontinence', 'prostate'],
            'oncological': ['cancer', 'tumor', 'malignant', 'chemotherapy'],
            'ophthalmological': ['eye', 'vision', 'glaucoma', 'blind', 'cataract'],
            'infectious': ['infection', 'virus', 'bacteria', 'flu', 'fever'],
            'mental_health': ['depression', 'anxiety', 'mental', 'stress'],
            'pediatric': ['child', 'infant', 'pediatric', 'baby'],
            'gynecological': ['pregnancy', 'menstrual', 'ovarian', 'cervical']
        }

        scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[category] = score

        return max(scores, key=scores.get) if scores else 'general'

    def _classify_urgency(self, row) -> str:
        text = (str(row['question']) + " " + str(row['answer'])).lower()

        emergency_keywords = ['emergency', 'urgent', 'severe', 'critical', 'immediate', 'life-threatening']
        high_keywords = ['serious', 'important', 'significant', 'concerning', 'worrisome']
        medium_keywords = ['treatment', 'therapy', 'consult', 'doctor', 'medical attention']

        if any(keyword in text for keyword in emergency_keywords):
            return 'emergency'
        elif any(keyword in text for keyword in high_keywords):
            return 'high'
        elif any(keyword in text for keyword in medium_keywords):
            return 'medium'
        else:
            return 'low'

    def _load_medical_categories(self) -> Dict:
        return {
            'urgency_indicators': {
                'emergency': ['severe', 'urgent', 'emergency', 'critical', 'life-threatening'],
                'high': ['serious', 'concerning', 'significant', 'worrisome', 'important'],
                'medium': ['treatment', 'consult', 'see doctor', 'medical attention'],
                'low': ['general', 'information', 'prevention', 'routine']
            }
        }

    def _encode_text(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text,
                               return_tensors='pt',
                               truncation=True,
                               padding=True,
                               max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].numpy()

        return embedding.flatten()

    def _get_or_create_embeddings(self) -> np.ndarray:

        if self.cache_embeddings and Path(self.cache_path).exists():
            print(" Loading cached BERT embeddings...")
            try:
                with open(self.cache_path, 'rb') as f:
                    embeddings = pickle.load(f)
                print(f" Loaded cached embeddings: {embeddings.shape}")
                return embeddings
            except Exception as e:
                print(f"  Error loading cache: {e}")

        print(" Creating BERT embeddings for medical questions...")
        embeddings = []

        total = len(self.medical_data)
        for i, row in self.medical_data.iterrows():
            if i % 100 == 0:
                print(f"   Processing: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

            question_embedding = self._encode_text(str(row['question']))
            embeddings.append(question_embedding)

        embeddings = np.array(embeddings)

        if self.cache_embeddings:
            try:
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(embeddings, f)
                print(f" Cached embeddings to {self.cache_path}")
            except Exception as e:
                print(f"  Could not cache embeddings: {e}")

        print(f" Created embeddings: {embeddings.shape}")
        return embeddings

    def find_similar_questions(self, user_query: str, top_k: int = 5) -> List[MedicalAnswer]:

        print(f" Searching for: '{user_query}'")

        query_embedding = self._encode_text(user_query).reshape(1, -1)

        similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:
                row = self.medical_data.iloc[idx]

                result = MedicalAnswer(
                    original_question=str(row['question']),
                    medical_answer=str(row['answer']),
                    similarity_score=float(similarities[idx]),
                    urgency_level=str(row['urgency_level']),
                    medical_category=str(row['medical_category']),
                    confidence=float(similarities[idx]),
                    dataset_index=int(idx)
                )
                results.append(result)

        print(f" Found {len(results)} relevant matches")
        return results

    def get_medical_answer(self, user_query: str) -> MedicalAnswer:
        results = self.find_similar_questions(user_query, top_k=1)

        if results:
            return results[0]
        else:
            return MedicalAnswer(
                original_question="No similar question found",
                medical_answer="I couldn't find a closely matching question in the medical database. Please consult with a healthcare provider for personalized medical advice regarding your specific concern.",
                similarity_score=0.0,
                urgency_level='medium',
                medical_category='general',
                confidence=0.0,
                dataset_index=-1
            )

    def interactive_consultation(self):
        print("\n BERT-BASED MEDICAL CONSULTATION")
        print("=" * 70)
        print("Ask medical questions and get answers from real medical professionals!")
        print("Using BERT embeddings for accurate question matching.")
        print("Type 'quit' to exit, 'help' for commands")
        print()

        while True:
            try:
                user_input = input(" Your medical question: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(" Thank you for using BERT Medical Assistant!")
                    break

                if user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  help - Show this help")
                    print("  quit - Exit the system")
                    print("  Just type your medical question for answers!")
                    continue

                print(f"\n BERT is analyzing: '{user_input}'")

                results = self.find_similar_questions(user_input, top_k=3)

                if results:
                    best = results[0]
                    print("\n" + "="*80)
                    print(" BEST MEDICAL ANSWER")
                    print("="*80)
                    print(f" Similar Question: {best.original_question}")
                    print(f" Medical Answer: {best.medical_answer}")
                    print(f" Similarity: {best.similarity_score:.3f}")
                    print(f" Urgency: {best.urgency_level.upper()}")
                    print(f" Category: {best.medical_category.title()}")
                    print("="*80)

                    if len(results) > 1:
                        print(f"\n Additional {len(results)-1} similar matches:")
                        for i, result in enumerate(results[1:], 2):
                            print(f"\n{i}. Question: {result.original_question[:100]}...")
                            print(f"   Similarity: {result.similarity_score:.3f}")
                            print(f"   Category: {result.medical_category}")

                else:
                    print("\n No sufficiently similar questions found in the medical database.")
                    print(" Try rephrasing your question or consulting a healthcare provider.")

                print("\nAsk another question or type 'quit' to exit.")

            except KeyboardInterrupt:
                print("\n\n Goodbye!")
                break
            except Exception as e:
                print(f" Error: {e}")
                continue

def main():
    try:
        assistant = BERTMedicalAssistant(
            model_name="bert-base-uncased",
            dataset_path="mle_screening_dataset.csv"
        )

        assistant.interactive_consultation()

    except FileNotFoundError:
        print(" Dataset file 'mle_screening_dataset.csv' not found!")
        print("Make sure the dataset is in the current directory.")
    except Exception as e:
        print(f" Error initializing BERT Medical Assistant: {e}")

if __name__ == "__main__":
    main()
