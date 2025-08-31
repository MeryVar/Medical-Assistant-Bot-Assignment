import sys
sys.path.append('src')

from data_processor import MedicalDataProcessor
from config import data_config
import json
import pandas as pd
from pathlib import Path

def main():
    print(" Starting Medical Data Preprocessing")
    print("=" * 50)

    processor = MedicalDataProcessor()

    print("\n Loading augmented dataset...")
    augmented_data_path = Path("data/raw/augmented_medical_conversations.json")

    if not augmented_data_path.exists():
        print(" Augmented dataset not found. Please run augment_dataset.py first.")
        return

    with open(augmented_data_path, 'r') as f:
        augmented_data = json.load(f)

    print(f" Loaded {len(augmented_data)} entries for preprocessing")

    df = pd.DataFrame(augmented_data)

    print("\n Dataset Overview:")
    print(f" Shape: {df.shape}")
    print(f" Columns: {list(df.columns)}")
    print(f" Medical Categories: {df['medical_category'].nunique()}")
    print(f" Unique Conditions: {df['condition'].nunique()}")

    print("\n Starting preprocessing pipeline...")

    print(" Standardizing column names...")
    df = processor.standardize_columns(df)

    print(" Cleaning text data...")
    df['input'] = df['input'].apply(processor.clean_text)
    df['response'] = df['response'].apply(processor.clean_text)

    print(" Filtering for data quality...")
    original_count = len(df)
    df = processor.filter_data_quality(df)
    filtered_count = len(df)
    print(f" Filtered: {original_count} -> {filtered_count} entries ({(filtered_count/original_count)*100:.1f}% retained)")

    print(" Converting to conversational format...")
    conversations = processor.create_conversational_format(df)
    print(f" Created {len(conversations)} conversations")

    print(f"\n Splitting data (Train: {data_config.train_split}, Val: {data_config.val_split}, Test: {data_config.test_split})...")
    train_data, val_data, test_data = processor.split_data(
    conversations,
    train_ratio=data_config.train_split,
    val_ratio=data_config.val_split,
    test_ratio=data_config.test_split
    )

    processed_data = {
    'train': train_data,
    'validation': val_data,
    'test': test_data
    }

    print("\n Saving processed data...")
    output_dir = Path(data_config.processed_data_dir)
    processor.save_processed_data(processed_data, output_dir)

    print("\n PREPROCESSING RESULTS:")
    print(f" Training samples: {len(train_data)} ({(len(train_data)/len(conversations))*100:.1f}%)")
    print(f" Validation samples: {len(val_data)} ({(len(val_data)/len(conversations))*100:.1f}%)")
    print(f" Test samples: {len(test_data)} ({(len(test_data)/len(conversations))*100:.1f}%)")
    print(f" Total processed: {len(conversations)} conversations")

    print("\n TEXT STATISTICS (post-preprocessing):")
    all_inputs = [item['input'] for item in conversations]
    all_responses = [item['response'] for item in conversations]

    avg_input_len = sum(len(text) for text in all_inputs) / len(all_inputs)
    avg_response_len = sum(len(text) for text in all_responses) / len(all_responses)
    avg_input_words = sum(len(text.split()) for text in all_inputs) / len(all_inputs)
    avg_response_words = sum(len(text.split()) for text in all_responses) / len(all_responses)

    print(f" Average input length: {avg_input_len:.0f} characters ({avg_input_words:.0f} words)")
    print(f" Average response length: {avg_response_len:.0f} characters ({avg_response_words:.0f} words)")

    print("\n CATEGORY DISTRIBUTION BY SPLIT:")

def analyze_split(split_data, split_name):
    categories = {}
    urgencies = {}

    for item in split_data:
        cat = item['metadata']['medical_category']
        urg = item.get('metadata', {}).get('urgency', 'unknown')

        categories[cat] = categories.get(cat, 0) + 1
        urgencies[urg] = urgencies.get(urg, 0) + 1

    print(f"\n {split_name.upper()} SET:")
    print(f" Categories: {dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))}")
    print(f" Urgencies: {dict(sorted(urgencies.items(), key=lambda x: x[1], reverse=True))}")

    analyze_split(train_data, "training")
    analyze_split(val_data, "validation")
    analyze_split(test_data, "test")

    preprocessing_stats = {
    'original_entries': original_count,
    'filtered_entries': filtered_count,
    'final_conversations': len(conversations),
    'train_size': len(train_data),
    'val_size': len(val_data),
    'test_size': len(test_data),
    'text_stats': {
    'avg_input_length': avg_input_len,
    'avg_response_length': avg_response_len,
    'avg_input_words': avg_input_words,
    'avg_response_words': avg_response_words
    }
    }

    stats_path = Path("data/preprocessing_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(preprocessing_stats, f, indent=2)

    print(f"\n Preprocessing statistics saved to: {stats_path}")
    print("\n Data preprocessing completed successfully!")
    print(" Next step: Model architecture design")

    print("\n Testing tokenization...")
    try:
        sample_conversations = conversations[:3]
        tokenized_sample = processor.tokenize_conversations(sample_conversations)
        print(f" Successfully tokenized {len(sample_conversations)} sample conversations")
        print(f" Token statistics: input_ids shape example: {tokenized_sample['input_ids'][0].shape}")
    except Exception as e:
        print(f" Tokenization test failed: {e}")

if __name__ == "__main__":
    main()
