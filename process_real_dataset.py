import json
import pandas as pd
import re
from typing import Dict, List
from pathlib import Path
from datetime import datetime

from src import data_processor

def extract_medical_category(question: str, answer: str) -> str:
    text = (question + " " + answer).lower()
    
    categories = {
        'cardiovascular': ['heart', 'cardiac', 'blood pressure', 'hypertension', 'chest pain', 'angina', 'arrhythmia'],
        'respiratory': ['lung', 'breathing', 'asthma', 'cough', 'pneumonia', 'bronchitis', 'respiratory'],
        'neurological': ['brain', 'nerve', 'headache', 'migraine', 'seizure', 'stroke', 'neurological'],
        'gastrointestinal': ['stomach', 'digestive', 'liver', 'intestine', 'bowel', 'gastric', 'abdominal'],
        'endocrine': ['diabetes', 'thyroid', 'hormone', 'insulin', 'glucose', 'endocrine'],
        'musculoskeletal': ['bone', 'joint', 'muscle', 'arthritis', 'fracture', 'spine', 'orthopedic'],
        'dermatological': ['skin', 'rash', 'dermatitis', 'eczema', 'acne', 'dermatology'],
        'urological': ['urinary', 'kidney', 'bladder', 'incontinence', 'prostate', 'urology'],
        'oncological': ['cancer', 'tumor', 'malignant', 'chemotherapy', 'oncology', 'carcinoma'],
        'ophthalmological': ['eye', 'vision', 'glaucoma', 'blind', 'cataract', 'ophthalmology'],
        'infectious': ['infection', 'virus', 'bacteria', 'flu', 'fever', 'infectious'],
        'mental_health': ['depression', 'anxiety', 'mental', 'stress', 'psychiatric'],
        'pediatric': ['child', 'infant', 'pediatric', 'baby', 'children'],
        'gynecological': ['pregnancy', 'menstrual', 'ovarian', 'cervical', 'gynecology']
    }
    
    category_scores = {}
    for category, keywords in categories.items():
        score = sum(text.count(keyword) for keyword in keywords)
        if score > 0:
            category_scores[category] = score

    if category_scores:
        return max(category_scores, key=category_scores.get)
    return 'general'

def extract_urgency_level(question: str, answer: str) -> str:
    text = (question + " " + answer).lower()

    emergency_words = ['emergency', 'immediately', 'urgent', 'call 911', 'life-threatening', 'seek immediate']
    if any(word in text for word in emergency_words):
        return 'emergency'

    high_urgency_words = ['serious', 'severe', 'dangerous', 'critical', 'important to treat', 'see doctor soon']
    if any(word in text for word in high_urgency_words):
        return 'high'

    medium_words = ['treatment', 'therapy', 'medication', 'consult', 'doctor', 'physician']
    if any(word in text for word in medium_words):
        return 'medium'

    return 'low'

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    
    return text

def process_medical_qa_dataset(input_file: str, output_file: str = None) -> pd.DataFrame:
    print(f" Processing medical Q&A dataset: {input_file}")
    
    try:
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.json'):
            df = pd.read_json(input_file)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        print(f" Loaded {len(df)} entries from dataset")
        
        original_columns = df.columns.tolist()
        print(f" Original columns: {original_columns}")
        
        question_col = None
        answer_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'question' in col_lower or 'query' in col_lower or 'input' in col_lower:
                question_col = col
            elif 'answer' in col_lower or 'response' in col_lower or 'output' in col_lower:
                answer_col = col
        
        if not question_col or not answer_col:
            print(" Warning: Could not automatically detect question/answer columns")
            print(" Available columns:", df.columns.tolist())
            question_col = df.columns[0] if len(df.columns) > 0 else 'question'
            answer_col = df.columns[1] if len(df.columns) > 1 else 'answer'
        
        print(f" Using '{question_col}' as question column")
        print(f" Using '{answer_col}' as answer column")
        
        processed_data = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"   Processing: {idx+1}/{len(df)}")
            
            question = clean_text(row[question_col])
            answer = clean_text(row[answer_col])
            
            if len(question.strip()) < 5 or len(answer.strip()) < 10:
                continue
            
            medical_category = extract_medical_category(question, answer)
            urgency_level = extract_urgency_level(question, answer)
            
            processed_entry = {
                'question': question,
                'answer': answer,
                'medical_category': medical_category,
                'urgency_level': urgency_level,
                'original_index': idx
            }
            
            processed_data.append(processed_entry)
        
        processed_df = pd.DataFrame(processed_data)
        
        print(f" Successfully processed {len(processed_df)} valid Q&A pairs")
        
        print("\nCategory distribution:")
        category_counts = processed_df['medical_category'].value_counts()
        for category, count in category_counts.items():
            print(f"   {category}: {count}")
        
        print("\nUrgency distribution:")
        urgency_counts = processed_df['urgency_level'].value_counts()
        for urgency, count in urgency_counts.items():
            print(f"   {urgency}: {count}")
        
        if output_file:
            processed_df.to_csv(output_file, index=False)
            print(f" Saved processed dataset to: {output_file}")
        
        return processed_df
        
    except Exception as e:
        print(f" Error processing dataset: {e}")
        raise

def merge_datasets(datasets: List[pd.DataFrame], output_file: str = None) -> pd.DataFrame:
    print(f" Merging {len(datasets)} datasets...")
    
    merged_df = pd.concat(datasets, ignore_index=True)
    
    print(f" Before deduplication: {len(merged_df)} entries")
    
    merged_df = merged_df.drop_duplicates(subset=['question', 'answer'])
    
    print(f" After deduplication: {len(merged_df)} entries")
    
    if output_file:
        merged_df.to_csv(output_file, index=False)
        print(f" Saved merged dataset to: {output_file}")
    
    return merged_df

def analyze_dataset(df: pd.DataFrame) -> Dict:
    analysis = {
        'total_entries': len(df),
        'avg_question_length': df['question'].str.len().mean(),
        'avg_answer_length': df['answer'].str.len().mean(),
        'category_distribution': df['medical_category'].value_counts().to_dict(),
        'urgency_distribution': df['urgency_level'].value_counts().to_dict(),
        'empty_questions': df['question'].isna().sum(),
        'empty_answers': df['answer'].isna().sum(),
    }
    
    return analysis

def save_analysis(analysis: Dict, filepath: str):
    with open(filepath, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f" Analysis saved to: {filepath}")

def main():
    print("REAL MEDICAL DATASET PROCESSOR")
    print("=" * 50)
    
    try:
        input_file = "data/raw/real_medical_dataset.json"
        
        if not Path(input_file).exists():
            print(f" Error: Input file '{input_file}' not found")
            print(" Please ensure the real medical dataset is available")
            return
        
        processed_df = process_medical_qa_dataset(
            input_file=input_file,
            output_file="data/processed/real_medical_qa_processed.csv"
        )
        
        analysis = analyze_dataset(processed_df)
        
        print("\nDATASET ANALYSIS:")
        print("=" * 30)
        print(f"Total entries: {analysis['total_entries']}")
        print(f"Average question length: {analysis['avg_question_length']:.1f} characters")
        print(f"Average answer length: {analysis['avg_answer_length']:.1f} characters")
        
        save_analysis(analysis, "data/real_dataset_analysis.json")
        
        print("\nProcessing complete!")
        print(f"Processed dataset saved with {len(processed_df)} medical Q&A pairs")
        
    except Exception as e:
        print(f"Error in main processing: {e}")
        raise

if __name__ == "__main__":
    main()
