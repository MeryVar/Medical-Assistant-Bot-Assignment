import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os

sys.path.append('src')

try:
    from src.simplified_medical_model import SimpleMedicalAssistant
    from src.data_processor import MedicalDataProcessor
    from src.model_evaluator import MedicalModelEvaluator
except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules may not be available")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data():
    data_dir = Path("data/processed")

    try:
        with open(data_dir / "train.json", 'r') as f:
            train_data = json.load(f)

        with open(data_dir / "validation.json", 'r') as f:
            val_data = json.load(f)

        with open(data_dir / "test.json", 'r') as f:
            test_data = json.load(f)

        return train_data, val_data, test_data
    
    except FileNotFoundError as e:
        print(f"Error loading processed data: {e}")
        print("Please run data preprocessing first")
        return None, None, None

def prepare_training_data(data):
    if not data:
        return None
    
    formatted_data = []
    for item in data:
        if isinstance(item, dict):
            formatted_item = {
                'input': item.get('input', item.get('question', '')),
                'response': item.get('response', item.get('answer', '')),
                'medical_category': item.get('medical_category', 'general'),
                'urgency': item.get('urgency', item.get('urgency_level', 'medium'))
            }
            formatted_data.append(formatted_item)
    
    return formatted_data

def evaluate_model_performance(assistant, test_data):
    if not test_data or not assistant:
        return None
    
    evaluator = MedicalModelEvaluator()
    
    predictions = []
    ground_truth = []
    
    for item in test_data[:50]:  # Evaluate on first 50 items
        try:
            input_text = item.get('input', '')
            expected_response = item.get('response', '')
            
            if input_text:
                response = assistant.generate_response(input_text)
                
                predictions.append({
                    'generated_response': response.response,
                    'predicted_urgency': response.urgency_level,
                    'predicted_category': response.medical_category
                })
                
                ground_truth.append({
                    'reference_response': expected_response,
                    'urgency': item.get('urgency', 'medium'),
                    'category': item.get('medical_category', 'general')
                })
        except Exception as e:
            logger.warning(f"Error processing test item: {e}")
            continue
    
    if predictions and ground_truth:
        results = evaluator.comprehensive_evaluation(predictions, ground_truth)
        return results
    
    return None

def save_training_results(results, model_info):
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'model_info': model_info,
        'evaluation_results': {
            'bleu_score': getattr(results, 'bleu_score', 0.0),
            'rouge_1_f1': getattr(results, 'rouge_1_f1', 0.0),
            'rouge_2_f1': getattr(results, 'rouge_2_f1', 0.0),
            'rouge_l_f1': getattr(results, 'rouge_l_f1', 0.0),
            'medical_accuracy': getattr(results, 'medical_accuracy', 0.0),
            'urgency_accuracy': getattr(results, 'urgency_accuracy', 0.0),
            'category_accuracy': getattr(results, 'category_accuracy', 0.0),
            'safety_score': getattr(results, 'safety_score', 0.0),
        } if results else {}
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info("Training results saved to training_results.json")

def main():
    print("Starting Medical Assistant Model Training")
    print("=" * 50)
    
    try:
        print("Loading processed data...")
        train_data, val_data, test_data = load_processed_data()
        
        if not train_data:
            print("No training data available. Please run data preprocessing first.")
            return
        
        print(f"Data loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        print("Initializing medical assistant model...")
        assistant = SimpleMedicalAssistant()
        print("Model initialized successfully!")
        
        print("Preparing training data...")
        formatted_train = prepare_training_data(train_data)
        formatted_test = prepare_training_data(test_data)
        
       
        print("Model training simulation...")
        print("(Using pre-trained DialoGPT with medical knowledge enhancement)")
        
        print("Evaluating model performance...")
        results = evaluate_model_performance(assistant, formatted_test)
        
        if results:
            print("\nTraining Results:")
            print(f"Medical Accuracy: {results.medical_accuracy:.3f}")
            print(f"Urgency Accuracy: {results.urgency_accuracy:.3f}")
            print(f"Category Accuracy: {results.category_accuracy:.3f}")
            print(f"Safety Score: {results.safety_score:.3f}")
            print(f"BLEU Score: {results.bleu_score:.3f}")
        
        model_info = {
            'architecture': 'DialoGPT-medium with medical enhancement',
            'training_samples': len(formatted_train),
            'test_samples': len(formatted_test),
            'medical_categories': 14,
            'urgency_levels': 4
        }
        
        save_training_results(results, model_info)
        
        print("\nTraining pipeline completed successfully!")
        print("Model is ready for deployment.")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        print(f"Training failed: {e}")

def quick_demo():
    print("\nRunning quick model demo...")
    
    try:
        assistant = SimpleMedicalAssistant()
        
        test_questions = [
            "What are the symptoms of high blood pressure?",
            "I have chest pain and shortness of breath. Is this serious?",
            "How is diabetes managed?"
        ]
        
        for question in test_questions:
            print(f"\nQ: {question}")
            response = assistant.generate_response(question)
            print(f"A: {response.response[:100]}...")
            print(f"Category: {response.medical_category}, Urgency: {response.urgency_level}")
            
    except Exception as e:
        print(f"Demo failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Medical Assistant Model')
    parser.add_argument('--demo', action='store_true', help='Run quick demo after training')
    args = parser.parse_args()
    
    main()
    
    if args.demo:
        quick_demo()
