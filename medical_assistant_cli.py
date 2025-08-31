import sys
import os
sys.path.append('src')

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

try:
    from src.enhanced_medical_assistant import EnhancedMedicalAssistant, ModelOutput
    from src.model_evaluator import MedicalModelEvaluator
    from src.data_processor import MedicalDataProcessor
except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules may not be available")

class MedicalAssistantCLI:

    def __init__(self):
        self.assistant = None
        self.evaluator = None
        self.session_history = []
        self.start_time = datetime.now()
        
        self.performance_data = self._load_performance_data()
        
        self.example_interactions = [
            {
                "question": "What are the symptoms of high blood pressure?",
                "context": "Patient seeking information about hypertension symptoms"
            },
            {
                "question": "I'm experiencing chest pain and shortness of breath. Should I be worried?",
                "context": "Potential emergency situation requiring immediate assessment"
            },
            {
                "question": "How is diabetes managed?",
                "context": "Educational inquiry about diabetes treatment approaches"
            },
            {
                "question": "What are the side effects of blood pressure medications?",
                "context": "Patient concerns about medication effects"
            },
            {
                "question": "When should I see a doctor for a persistent headache?",
                "context": "Symptom assessment and care seeking guidance"
            }
        ]

    def _load_performance_data(self) -> Dict:
        try:
            if Path("final_testing_report.json").exists():
                with open("final_testing_report.json", 'r') as f:
                    testing_data = json.load(f)
            else:
                testing_data = {}

            if Path("demo_results.json").exists():
                with open("demo_results.json", 'r') as f:
                    demo_data = json.load(f)
            else:
                demo_data = {}

            return {
                'testing_results': testing_data,
                'demo_results': demo_data,
                'dataset_size': 16406,
                'processed_samples': 999,
                'model_architecture': 'Transformer-based (DialoGPT)',
                'training_approach': 'Fine-tuned conversational AI with medical knowledge enhancement'
            }
        except Exception as e:
            print(f"Error loading performance data: {e}")
            return {}

    def _initialize_models(self):
        try:
            print("🚀 Initializing enhanced medical assistant...")
            self.assistant = EnhancedMedicalAssistant()
            print("✅ Medical assistant initialized successfully!")
            
            print("🔧 Initializing evaluator...")
            self.evaluator = MedicalModelEvaluator()
            print("✅ Evaluator initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            return False
        return True

    def _display_welcome(self):
        print("\n" + "="*70)
        print("           MEDICAL ASSISTANT - INTERACTIVE CLI")
        print("="*70)
        print("Welcome to the AI-powered Medical Assistant!")
        print("This system provides intelligent medical information and guidance.")
        print("\nAVAILABLE COMMANDS:")
        print("  ask         - Ask a medical question")
        print("  demo        - Run demonstration with example cases")
        print("  evaluate    - Evaluate model performance")
        print("  stats       - Show session statistics")
        print("  examples    - Show example interactions")
        print("  help        - Show detailed help")
        print("  quit        - Exit the application")
        print("\nDISCLAIMER: This AI provides general information only.")
        print("Always consult healthcare professionals for medical decisions.")
        print("="*70)

    def _display_help(self):
        help_text = """
MEDICAL ASSISTANT CLI - DETAILED HELP
=====================================

COMMANDS:
---------
ask                 Ask a medical question
                   Example: ask "What are the symptoms of diabetes?"

demo               Run demonstration with example medical scenarios
                   Shows how the assistant handles different types of questions

evaluate           Run model performance evaluation
                   Displays accuracy metrics and model capabilities

stats              Show current session statistics
                   Includes number of questions asked and performance metrics

examples           Display example medical questions you can ask
                   Useful for understanding the system's capabilities

help               Show this detailed help information

quit/exit          Exit the Medical Assistant CLI

USAGE TIPS:
-----------
- Ask specific medical questions for better responses
- The system classifies urgency levels and medical categories
- Emergency situations are flagged for immediate attention
- All responses include appropriate medical disclaimers

MEDICAL CATEGORIES SUPPORTED:
----------------------------
- Cardiovascular (heart, blood pressure)
- Respiratory (lungs, breathing)
- Neurological (brain, nervous system)
- Gastrointestinal (digestive system)
- Endocrine (hormones, diabetes)
- And many more...

URGENCY LEVELS:
--------------
- Emergency: Immediate medical attention required
- High: Prompt medical consultation recommended
- Medium: Medical advice suggested
- Low: General information/educational content

EXAMPLE SESSION:
---------------
> ask What are the warning signs of a heart attack?
> demo
> stats
> quit
"""
        print(help_text)

    def _ask_question(self, question: str = None):
        if not self.assistant:
            if not self._initialize_models():
                print(" Cannot ask questions without initialized models.")
                return

        if not question:
            question = input("\n Your medical question: ").strip()
            
        if not question:
            print(" Please provide a question.")
            return

        print(f"\n Processing question: '{question}'")
        start_time = time.time()

        try:
            response = self.assistant.generate_response(question)
            processing_time = time.time() - start_time

            print("\n" + "="*60)
            print(" MEDICAL ASSISTANT RESPONSE")
            print("="*60)
            print(f" Question: {question}")
            print(f" Answer: {response.response}")
            print(f" Urgency Level: {response.urgency_level.upper()}")
            print(f" Medical Category: {response.medical_category.title()}")
            print(f" Confidence: {response.confidence:.2f}")
            print(f" Processing Time: {processing_time:.2f}s")
            print("="*60)

            self.session_history.append({
                'question': question,
                'response': response,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            print(f" Error processing question: {e}")

    def _run_demo(self):
        if not self.assistant:
            if not self._initialize_models():
                print(" Cannot run demo without initialized models.")
                return

        print("\n" + "="*70)
        print("           MEDICAL ASSISTANT DEMONSTRATION")
        print("="*70)
        print("Running demonstration with example medical questions...")

        demo_results = []

        for i, example in enumerate(self.example_interactions, 1):
            print(f"\n--- DEMO CASE {i} ---")
            print(f"Question: {example['question']}")
            print(f"Context: {example['context']}")
            
            try:
                start_time = time.time()
                response = self.assistant.generate_response(example['question'])
                processing_time = time.time() - start_time

                print(f"\nResponse: {response.response}")
                print(f"Urgency: {response.urgency_level.upper()}")
                print(f"Category: {response.medical_category.title()}")
                print(f"Confidence: {response.confidence:.2f}")
                print(f"Processing Time: {processing_time:.2f}s")

                demo_results.append({
                    'question': example['question'],
                    'context': example['context'],
                    'response': response,
                    'processing_time': processing_time
                })

            except Exception as e:
                print(f"Error in demo case {i}: {e}")

            print("-" * 50)

        print(f"\nDemo completed! Processed {len(demo_results)} examples.")
        
        if demo_results:
            avg_time = sum(r['processing_time'] for r in demo_results) / len(demo_results)
            print(f"Average processing time: {avg_time:.2f}s")

    def _run_evaluation(self):
        if not self.evaluator:
            if not self._initialize_models():
                print(" Cannot run evaluation without initialized models.")
                return

        print("\n" + "="*70)
        print("           MODEL PERFORMANCE EVALUATION")
        print("="*70)

        try:
            demo_predictions = []
            demo_ground_truth = []

            for example in self.example_interactions[:3]:  # Use first 3 examples
                try:
                    response = self.assistant.generate_response(example['question'])
                    
                    demo_predictions.append({
                        'generated_response': response.response,
                        'predicted_urgency': response.urgency_level,
                        'predicted_category': response.medical_category
                    })
                    
                    demo_ground_truth.append({
                        'reference_response': "Professional medical consultation recommended for proper diagnosis and treatment.",
                        'urgency': 'medium',
                        'category': 'general'
                    })
                    
                except Exception as e:
                    print(f"Error in evaluation example: {e}")

            if demo_predictions and demo_ground_truth:
                results = self.evaluator.comprehensive_evaluation(demo_predictions, demo_ground_truth)
                report = self.evaluator.generate_evaluation_report(results)
                print(report)
            else:
                print("No evaluation data available.")

        except Exception as e:
            print(f"Error during evaluation: {e}")

    def _show_stats(self):
        print("\n" + "="*60)
        print("           SESSION STATISTICS")
        print("="*60)
        
        session_duration = datetime.now() - self.start_time
        print(f"Session Duration: {session_duration}")
        print(f"Questions Asked: {len(self.session_history)}")
        
        if self.session_history:
            avg_time = sum(h['processing_time'] for h in self.session_history) / len(self.session_history)
            print(f"Average Response Time: {avg_time:.2f}s")
            
            urgency_counts = {}
            category_counts = {}
            
            for history in self.session_history:
                urgency = history['response'].urgency_level
                category = history['response'].medical_category
                
                urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
                category_counts[category] = category_counts.get(category, 0) + 1
            
            print("\nUrgency Distribution:")
            for urgency, count in urgency_counts.items():
                print(f"  {urgency.title()}: {count}")
            
            print("\nCategory Distribution:")
            for category, count in category_counts.items():
                print(f"  {category.title()}: {count}")
        
        print("\nPerformance Data:")
        if self.performance_data:
            print(f"  Dataset Size: {self.performance_data.get('dataset_size', 'N/A')}")
            print(f"  Architecture: {self.performance_data.get('model_architecture', 'N/A')}")
        
        print("="*60)

    def _show_examples(self):
        print("\n" + "="*60)
        print("           EXAMPLE MEDICAL QUESTIONS")
        print("="*60)
        
        print("Here are some example questions you can ask:")
        
        for i, example in enumerate(self.example_interactions, 1):
            print(f"\n{i}. {example['question']}")
            print(f"   Context: {example['context']}")
        
        print(f"\nYou can ask these questions using: ask \"<question>\"")
        print("Or simply type: ask")
        print("="*60)

    def run_interactive_mode(self):
        self._display_welcome()
        
        while True:
            try:
                user_input = input("\nmedical-assistant> ").strip().lower()
                
                if not user_input:
                    continue
                
                if user_input in ['quit', 'exit', 'q']:
                    print("\nThank you for using the Medical Assistant!")
                    print("Remember to consult healthcare professionals for medical advice.")
                    break
                    
                elif user_input == 'help':
                    self._display_help()
                    
                elif user_input == 'demo':
                    self._run_demo()
                    
                elif user_input == 'evaluate':
                    self._run_evaluation()
                    
                elif user_input == 'stats':
                    self._show_stats()
                    
                elif user_input == 'examples':
                    self._show_examples()
                    
                elif user_input == 'ask':
                    self._ask_question()
                    
                elif user_input.startswith('ask '):
                    question = user_input[4:].strip()
                    if question.startswith('"') and question.endswith('"'):
                        question = question[1:-1]
                    self._ask_question(question)
                    
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\nExiting Medical Assistant...")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Medical Assistant CLI')
    parser.add_argument('--question', type=str, help='Ask a single question and exit')
    parser.add_argument('--demo', action='store_true', help='Run demo and exit')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation and exit')
    
    args = parser.parse_args()
    
    cli = MedicalAssistantCLI()
    
    if args.question:
        if cli._initialize_models():
            cli._ask_question(args.question)
    elif args.demo:
        if cli._initialize_models():
            cli._run_demo()
    elif args.evaluate:
        if cli._initialize_models():
            cli._run_evaluation()
    else:
        cli.run_interactive_mode()

if __name__ == "__main__":
    main()
