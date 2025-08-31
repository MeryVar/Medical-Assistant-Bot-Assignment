import sys
sys.path.append('src')

from dataset_loader import MedicalDatasetLoader
from data_augmentation import MedicalDataAugmentor
import json
from pathlib import Path

def main():
    print(" Starting Medical Dataset Augmentation")
    print("=" * 50)

    loader = MedicalDatasetLoader()
    augmentor = MedicalDataAugmentor()

    print("\n Loading original dataset...")
    original_data = loader.combine_datasets(['sample', 'synthetic'])
    print(f" Loaded {len(original_data)} original entries")

    target_size = 150
    print(f"\n Augmenting dataset to {target_size} entries...")
    augmented_data = augmentor.augment_dataset(original_data, target_size)
    print(f" Dataset augmented to {len(augmented_data)} entries")

    stats = augmentor.get_augmentation_stats(original_data, augmented_data)

    print("\n AUGMENTATION RESULTS:")
    print(f" Original size: {stats['original_size']}")
    print(f" Final size: {stats['augmented_size']}")
    print(f" Added entries: {stats['added_entries']}")

    print("\n CATEGORY DISTRIBUTION:")
    for category, count in stats['augmented_categories'].items():
        percentage = (count / stats['augmented_size']) * 100
    print(f" {category}: {count} ({percentage:.1f}%)")

    print("\n URGENCY DISTRIBUTION:")
    for urgency, count in stats['augmented_urgency'].items():
        percentage = (count / stats['augmented_size']) * 100
    print(f" {urgency}: {count} ({percentage:.1f}%)")

    output_path = Path("data/raw/augmented_medical_conversations.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    augmentor.save_augmented_data(augmented_data, output_path)
    print(f"\n Augmented dataset saved to: {output_path}")

    stats_path = Path("data/augmentation_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f" Augmentation stats saved to: {stats_path}")

    print("\n Dataset augmentation completed successfully!")
    print(" Next step: Data preprocessing")

if __name__ == "__main__":
    main()
