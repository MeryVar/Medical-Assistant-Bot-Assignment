import sys
sys.path.append('src')

import pandas as pd
import json
import re
from pathlib import Path
from data_processor import MedicalDataProcessor
from config import data_config

def extract_medical_category(question: str, answer: str) -> str:
    text = (question + " " + answer).lower()

    categories = {
    'cardiovascular': ['heart', 'blood pressure', 'cardiac', 'cardiovascular', 'cholesterol', 'hypertension', 'blood vessel'],
    'respiratory': ['lung', 'breathing', 'respiratory', 'asthma', 'copd', 'pneumonia', 'cough'],
    'neurological': ['brain', 'nerve', 'neurological', 'stroke', 'headache', 'seizure', 'migraine'],
    'endocrine': ['diabetes', 'thyroid', 'hormone', 'endocrine', 'insulin', 'glucose'],
    'gastrointestinal': ['stomach', 'digestive', 'intestine', 'bowel', 'liver', 'gastro'],
    'musculoskeletal': ['bone', 'joint', 'muscle', 'arthritis', 'osteo', 'fracture'],
    'dermatological': ['skin', 'dermatology', 'rash', 'acne', 'eczema', 'psoriasis'],
    'ophthalmological': ['eye', 'vision', 'glaucoma', 'cataract', 'retina', 'blindness'],
    'oncological': ['cancer', 'tumor', 'malignant', 'chemotherapy', 'oncology'],
    'infectious': ['infection', 'virus', 'bacteria', 'antibiotic', 'fever', 'flu'],
    'mental_health': ['depression', 'anxiety', 'mental health', 'psychiatric', 'stress'],
    'pediatric': ['child', 'infant', 'pediatric', 'baby', 'adolescent'],
    'geriatric': ['elderly', 'aging', 'geriatric', 'senior']
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

def extract_condition(question: str) -> str:
    question = question.lower().strip()

    question = re.sub(r'what is \(are\)\s*', '', question)
    question = re.sub(r'what are the (symptoms|treatments|causes) of\s*', '', question)
    question = re.sub(r'who is at risk for\s*', '', question)
    question = re.sub(r'how to prevent\s*', '', question)
    question = re.sub(r'\s*\?+\s*$', '', question)

    condition = question.strip()
    if not condition:
    condition = 'general medical information'

    return condition

def main():
    print(" Processing Real Medical Dataset (MLE Screening)")
    print("=" * 60)

    print("\n Loading MLE screening dataset...")
    df = pd.read_csv('mle_screening_dataset.csv')
    print(f" Loaded {len(df)} medical Q&A pairs")

    print(f"\n Dataset Overview:")
    print(f" Shape: {df.shape}")
    print(f" Columns: {list(df.columns)}")
    print(f" Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    sample_size = 1000
    print(f"\n Sampling {sample_size} entries for processing...")
    df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print("\n Processing medical conversations...")
    processed_data = []

    for idx, row in df_sample.iterrows():
    if idx % 100 == 0:
    print(f" Processing entry {idx+1}/{len(df_sample)}")

    question = str(row['question']).strip()
    answer = str(row['answer']).strip()

    if not question or not answer or question == 'nan' or answer == 'nan':
    continue

    medical_category = extract_medical_category(question, answer)
    urgency = extract_urgency_level(question, answer)
    condition = extract_condition(question)

    entry = {
    'input': question,
    'response': answer,
    'medical_category': medical_category,
    'condition': condition,
    'urgency': urgency
    }

    processed_data.append(entry)

    print(f" Processed {len(processed_data)} medical conversations")

    print("\n DATASET ANALYSIS:")
    df_processed = pd.DataFrame(processed_data)

    print(f" Total entries: {len(processed_data)}")
    print(f" Medical categories: {df_processed['medical_category'].nunique()}")
    print(f" Unique conditions: {df_processed['condition'].nunique()}")

    print(f"\n MEDICAL CATEGORY DISTRIBUTION:")
    category_counts = df_processed['medical_category'].value_counts()
    for category, count in category_counts.head(10).items():
    percentage = (count / len(processed_data)) * 100
    print(f" {category}: {count} ({percentage:.1f}%)")

    print(f"\n URGENCY DISTRIBUTION:")
    urgency_counts = df_processed['urgency'].value_counts()
    for urgency, count in urgency_counts.items():
    percentage = (count / len(processed_data)) * 100
    print(f" {urgency}: {count} ({percentage:.1f}%)")

    df_processed['input_length'] = df_processed['input'].str.len()
    df_processed['response_length'] = df_processed['response'].str.len()

    print(f"\n TEXT LENGTH ANALYSIS:")
    print(f" Avg question length: {df_processed['input_length'].mean():.0f} characters")
    print(f" Avg answer length: {df_processed['response_length'].mean():.0f} characters")
    print(f" Max answer length: {df_processed['response_length'].max():,} characters")

    print("\n Saving processed dataset...")

    output_path = Path("data/raw/real_medical_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
    json.dump(processed_data, f, indent=2)

    print(f" Saved processed dataset to: {output_path}")

    analysis_results = {
    'original_size': len(df),
    'processed_size': len(processed_data),
    'sample_size': sample_size,
    'category_distribution': category_counts.to_dict(),
    'urgency_distribution': urgency_counts.to_dict(),
    'text_stats': {
    'avg_question_length': float(df_processed['input_length'].mean()),
    'avg_answer_length': float(df_processed['response_length'].mean()),
    'max_answer_length': int(df_processed['response_length'].max())
    }
    }

    stats_path = Path("data/real_dataset_analysis.json")
    with open(stats_path, 'w') as f:
    json.dump(analysis_results, f, indent=2)

    print(f" Analysis results saved to: {stats_path}")

    print("\n Running through standard preprocessing pipeline...")

    processor = MedicalDataProcessor()

    conversations = processor.create_conversational_format(df_processed)

    train_data, val_data, test_data = processor.split_data(
    conversations,
    train_ratio=data_config.train_split,
    val_ratio=data_config.val_split,
    test_ratio=data_config.test_split
    )

    final_data = {
    'train': train_data,
    'validation': val_data,
    'test': test_data
    }

    processor.save_processed_data(final_data, Path(data_config.processed_data_dir))

    print("\n FINAL PROCESSING RESULTS:")
    print(f" Training samples: {len(train_data)} ({(len(train_data)/len(conversations))*100:.1f}%)")
    print(f" Validation samples: {len(val_data)} ({(len(val_data)/len(conversations))*100:.1f}%)")
    print(f" Test samples: {len(test_data)} ({(len(test_data)/len(conversations))*100:.1f}%)")

    print("\n Real dataset processing completed successfully!")
    print(" Ready for model architecture design and training")

    print("\n SAMPLE PROCESSED ENTRIES:")
    for i, entry in enumerate(processed_data[:3]):
    print(f"\n--- Sample {i+1} ---")
    print(f"Category: {entry['medical_category']}")
    print(f"Urgency: {entry['urgency']}")
    print(f"Condition: {entry['condition'][:50]}...")
    print(f"Question: {entry['input'][:100]}...")
    print(f"Answer: {entry['response'][:200]}...")

    if __name__ == "__main__":
    main()
