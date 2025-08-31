import pandas as pd
import numpy as np
from pathlib import Path
import re
import logging
from typing import Dict, List, Tuple, Union, Optional
import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class MedicalDataProcessor:

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.medical_categories = {
            'cardiovascular': ['heart', 'cardiac', 'blood pressure', 'hypertension'],
            'respiratory': ['lung', 'breathing', 'asthma', 'cough'],
            'neurological': ['brain', 'nerve', 'headache', 'seizure'],
            'gastrointestinal': ['stomach', 'digestive', 'liver', 'bowel'],
            'endocrine': ['diabetes', 'thyroid', 'hormone', 'insulin'],
            'musculoskeletal': ['bone', 'joint', 'muscle', 'arthritis'],
            'dermatological': ['skin', 'rash', 'dermatitis', 'eczema'],
            'urological': ['urinary', 'kidney', 'bladder', 'prostate'],
            'oncological': ['cancer', 'tumor', 'malignant', 'chemotherapy'],
            'general': ['symptom', 'disease', 'condition', 'treatment']
        }

    def load_data(self, data_path: Union[str, Path], data_format: str = 'auto') -> pd.DataFrame:
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if data_format == 'auto':
            data_format = data_path.suffix.lower()

        logger.info(f"Loading data from {data_path} (format: {data_format})")

        try:
            if data_format in ['.csv']:
                df = pd.read_csv(data_path)
            elif data_format in ['.json']:
                df = pd.read_json(data_path)
            elif data_format in ['.jsonl']:
                df = pd.read_json(data_path, lines=True)
            else:
                raise ValueError(f"Unsupported format: {data_format}")

            logger.info(f"Loaded {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        return text

    def preprocess_medical_text(self, df: pd.DataFrame, 
                                question_col: str = 'question',
                                answer_col: str = 'answer') -> pd.DataFrame:
        
        processed_df = df.copy()
        
        processed_df[f'{question_col}_clean'] = processed_df[question_col].apply(self.clean_text)
        processed_df[f'{answer_col}_clean'] = processed_df[answer_col].apply(self.clean_text)
        
        processed_df = processed_df[
            (processed_df[f'{question_col}_clean'].str.len() > 5) &
            (processed_df[f'{answer_col}_clean'].str.len() > 10)
        ]
        
        processed_df['medical_category'] = processed_df.apply(
            lambda row: self.classify_medical_category(
                str(row[question_col]) + " " + str(row[answer_col])
            ), axis=1
        )
        
        processed_df['urgency_level'] = processed_df.apply(
            lambda row: self.classify_urgency(
                str(row[question_col]) + " " + str(row[answer_col])
            ), axis=1
        )
        
        logger.info(f"Preprocessed {len(processed_df)} medical text pairs")
        return processed_df

    def classify_medical_category(self, text: str) -> str:
        text_lower = text.lower()
        
        category_scores = {}
        for category, keywords in self.medical_categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return 'general'

    def classify_urgency(self, text: str) -> str:
        text_lower = text.lower()
        
        emergency_keywords = ['emergency', 'urgent', 'immediately', 'critical', 'severe']
        high_keywords = ['serious', 'concerning', 'important', 'significant']
        medium_keywords = ['treatment', 'consult', 'doctor', 'physician']
        
        if any(keyword in text_lower for keyword in emergency_keywords):
            return 'emergency'
        elif any(keyword in text_lower for keyword in high_keywords):
            return 'high'
        elif any(keyword in text_lower for keyword in medium_keywords):
            return 'medium'
        else:
            return 'low'

    def tokenize_text(self, text: str, remove_stopwords: bool = True) -> List[str]:
        tokens = word_tokenize(text.lower())
        
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        tokens = [token for token in tokens if token.isalnum()]
        
        return tokens

    def create_train_test_split(self, df: pd.DataFrame, 
                               test_size: float = 0.2,
                               validation_size: float = 0.1,
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        train_df, temp_df = train_test_split(
            df, test_size=(test_size + validation_size), 
            random_state=random_state,
            stratify=df['medical_category'] if 'medical_category' in df.columns else None
        )
        
        val_size_adjusted = validation_size / (test_size + validation_size)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_size_adjusted),
            random_state=random_state,
            stratify=temp_df['medical_category'] if 'medical_category' in temp_df.columns else None
        )
        
        logger.info(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df

    def save_processed_data(self, data: Union[pd.DataFrame, Dict], 
                           output_path: Union[str, Path],
                           format: str = 'csv'):
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            if format == 'csv':
                data.to_csv(output_path, index=False)
            elif format == 'json':
                data.to_json(output_path, orient='records', indent=2)
        elif isinstance(data, dict):
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved processed data to {output_path}")

    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        analysis = {
            'total_samples': len(df),
            'categories': {},
            'urgency_levels': {},
            'text_statistics': {},
            'data_quality': {}
        }
        
        if 'medical_category' in df.columns:
            analysis['categories'] = df['medical_category'].value_counts().to_dict()
        
        if 'urgency_level' in df.columns:
            analysis['urgency_levels'] = df['urgency_level'].value_counts().to_dict()
        
        question_col = None
        answer_col = None
        for col in df.columns:
            if 'question' in col.lower():
                question_col = col
            elif 'answer' in col.lower():
                answer_col = col
        
        if question_col and answer_col:
            analysis['text_statistics'] = {
                'avg_question_length': df[question_col].str.len().mean(),
                'avg_answer_length': df[answer_col].str.len().mean(),
                'max_question_length': df[question_col].str.len().max(),
                'max_answer_length': df[answer_col].str.len().max()
            }
            
            analysis['data_quality'] = {
                'empty_questions': df[question_col].isna().sum(),
                'empty_answers': df[answer_col].isna().sum(),
                'duplicate_pairs': df.duplicated(subset=[question_col, answer_col]).sum()
            }
        
        return analysis

    def generate_processing_report(self, analysis: Dict) -> str:
        report = []
        report.append("MEDICAL DATA PROCESSING REPORT")
        report.append("=" * 50)
        
        report.append(f"\nTotal Samples: {analysis.get('total_samples', 'N/A')}")
        
        if 'categories' in analysis and analysis['categories']:
            report.append("\nMedical Categories:")
            for category, count in analysis['categories'].items():
                percentage = (count / analysis['total_samples']) * 100
                report.append(f"  {category}: {count} ({percentage:.1f}%)")
        
        if 'urgency_levels' in analysis and analysis['urgency_levels']:
            report.append("\nUrgency Levels:")
            for urgency, count in analysis['urgency_levels'].items():
                percentage = (count / analysis['total_samples']) * 100
                report.append(f"  {urgency}: {count} ({percentage:.1f}%)")
        
        if 'text_statistics' in analysis:
            stats = analysis['text_statistics']
            report.append("\nText Statistics:")
            report.append(f"  Average Question Length: {stats.get('avg_question_length', 0):.1f} chars")
            report.append(f"  Average Answer Length: {stats.get('avg_answer_length', 0):.1f} chars")
            report.append(f"  Max Question Length: {stats.get('max_question_length', 0)} chars")
            report.append(f"  Max Answer Length: {stats.get('max_answer_length', 0)} chars")
        
        if 'data_quality' in analysis:
            quality = analysis['data_quality']
            report.append("\nData Quality:")
            report.append(f"  Empty Questions: {quality.get('empty_questions', 0)}")
            report.append(f"  Empty Answers: {quality.get('empty_answers', 0)}")
            report.append(f"  Duplicate Pairs: {quality.get('duplicate_pairs', 0)}")
        
        return "\n".join(report)

def process_medical_dataset(input_path: str, output_dir: str = "data/processed"):
    processor = MedicalDataProcessor()
    
    print("Loading medical dataset...")
    df = processor.load_data(input_path)
    
    print("Preprocessing medical text...")
    processed_df = processor.preprocess_medical_text(df)
    
    print("Creating train/validation/test splits...")
    train_df, val_df, test_df = processor.create_train_test_split(processed_df)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processor.save_processed_data(train_df, output_path / "train.csv")
    processor.save_processed_data(val_df, output_path / "validation.csv")
    processor.save_processed_data(test_df, output_path / "test.csv")
    
    analysis = processor.analyze_dataset(processed_df)
    processor.save_processed_data(analysis, output_path / "processing_analysis.json", format='json')
    
    report = processor.generate_processing_report(analysis)
    print(report)
    
    with open(output_path / "processing_report.txt", 'w') as f:
        f.write(report)
    
    print(f"\nProcessing complete! Files saved to {output_path}")
    return processed_df, analysis

if __name__ == "__main__":
    input_file = "mle_screening_dataset.csv"
    if Path(input_file).exists():
        process_medical_dataset(input_file)
    else:
        print(f"Dataset file '{input_file}' not found!")
