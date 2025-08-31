import sys
import os
sys.path.append('src')

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

try:
    from src.simplified_medical_model import SimpleMedicalAssistant, ModelOutput
except ImportError as e:
    print(f"Import error: {e}")
    print("SimpleMedicalAssistant may not be available")

class MedicalAssistantDemo:
    
    def __init__(self):
        self.assistant = None
        self.demo_results = []
        
    def initialize_assistant(self):
        try:
            print("Initializing Medical Assistant...")
            self.assistant = SimpleMedicalAssistant()
            print("Medical Assistant initialized successfully!")
            return True
        except Exception as e:
            print(f"Failed to initialize Medical Assistant: {e}")
            return False
    
    def get_demo_scenarios(self) -> List[Dict]:
        return [
            {
                "id": 1,
                "question": "What are the symptoms of high blood pressure?",
                "expected_category": "cardiovascular",
                "expected_urgency": "medium",
                "context": "Educational inquiry about hypertension symptoms"
            },
            {
                "id": 2,
                "question": "I'm having severe chest pain and shortness of breath. What should I do?",
                "expected_category": "cardiovascular",
                "expected_urgency": "emergency",
                "context": "Potential medical emergency requiring immediate attention"
            },
            {
                "id": 3,
                "question": "How is type 2 diabetes typically managed?",
                "expected_category": "endocrine",
                "expected_urgency": "low",
                "context": "General information about diabetes management"
            },
            {
                "id": 4,
                "question": "I've been having persistent headaches for the past week. Should I be concerned?",
                "expected_category": "neurological",
                "expected_urgency": "medium",
                "context": "Symptom assessment and medical advice"
            },
            {
                "id": 5,
                "question": "What are the side effects of blood pressure medication?",
                "expected_category": "cardiovascular",
                "expected_urgency": "low",
                "context": "Medication information inquiry"
            },
            {
                "id": 6,
                "question": "I'm experiencing severe abdominal pain and vomiting. Is this serious?",
                "expected_category": "gastrointestinal",
                "expected_urgency": "high",
                "context": "Acute symptoms requiring medical evaluation"
            }
        ]
    
    def validate_response_appropriateness(self, response: ModelOutput, expected: Dict) -> Dict:
        validation = {
            'urgency_appropriate': False,
            'category_reasonable': False,
            'safety_guidance_present': False,
            'response_coherent': True
        }
        
        urgency_level = response.urgency_level.lower()
        expected_urgency = expected['expected_urgency'].lower()
        
        if urgency_level == expected_urgency:
            validation['urgency_appropriate'] = True
        elif urgency_level == 'emergency' and expected_urgency in ['high', 'emergency']:
            validation['urgency_appropriate'] = True
        elif urgency_level == 'high' and expected_urgency in ['medium', 'high']:
            validation['urgency_appropriate'] = True
        
        if response.medical_category.lower() == expected['expected_category'].lower():
            validation['category_reasonable'] = True
        
        response_lower = response.response.lower()
        safety_terms = [
            'healthcare provider', 'doctor', 'physician', 'medical professional',
            'consult', 'seek medical attention', 'emergency', 'call', '911'
        ]
        
        if any(term in response_lower for term in safety_terms):
            validation['safety_guidance_present'] = True
        
        if len(response.response.strip()) < 20:
            validation['response_coherent'] = False
        
        return validation
    
    def run_single_demo(self, scenario: Dict) -> Dict:
        """Run a single demo scenario"""
        print(f"\nDemo {scenario['id']}: {scenario['context']}")
        print(f"Question: {scenario['question']}")
        
        start_time = time.time()
        
        try:
            response = self.assistant.generate_response(scenario['question'])
            processing_time = time.time() - start_time
            
            validation = self.validate_response_appropriateness(response, scenario)
            
            print("\nResponse:")
            print(f"  Answer: {response.response}")
            print(f"  Category: {response.medical_category}")
            print(f"  Urgency: {response.urgency_level}")
            print(f"  Confidence: {response.confidence:.2f}")
            print(f"  Processing Time: {processing_time:.2f}s")
            
            print("\nValidation:")
            print(f"  Urgency Appropriate: {'✓' if validation['urgency_appropriate'] else '✗'}")
            print(f"  Category Reasonable: {'✓' if validation['category_reasonable'] else '✗'}")
            print(f"  Safety Guidance: {'✓' if validation['safety_guidance_present'] else '✗'}")
            print(f"  Response Coherent: {'✓' if validation['response_coherent'] else '✗'}")
            
            result = {
                'scenario_id': scenario['id'],
                'question': scenario['question'],
                'context': scenario['context'],
                'response': {
                    'answer': response.response,
                    'category': response.medical_category,
                    'urgency': response.urgency_level,
                    'confidence': response.confidence
                },
                'expected': {
                    'category': scenario['expected_category'],
                    'urgency': scenario['expected_urgency']
                },
                'validation': validation,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"Error in demo {scenario['id']}: {e}")
            return {
                'scenario_id': scenario['id'],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_full_demo(self) -> List[Dict]:
        print("MEDICAL ASSISTANT DEMONSTRATION")
        print("=" * 60)
        print("Running comprehensive demonstration of medical assistant capabilities")
        
        if not self.assistant:
            if not self.initialize_assistant():
                return []
        
        scenarios = self.get_demo_scenarios()
        results = []
        
        for scenario in scenarios:
            result = self.run_single_demo(scenario)
            results.append(result)
            self.demo_results.append(result)
            print("-" * 60)
        
        self.print_demo_summary(results)
        
        return results
    
    def print_demo_summary(self, results: List[Dict]):
        print("\nDEMO SUMMARY")
        print("=" * 60)
        
        total_scenarios = len(results)
        successful_scenarios = len([r for r in results if 'error' not in r])
        
        if successful_scenarios == 0:
            print("No successful scenarios to analyze.")
            return
        
        validations = [r['validation'] for r in results if 'validation' in r]
        
        urgency_correct = sum(1 for v in validations if v['urgency_appropriate'])
        category_correct = sum(1 for v in validations if v['category_reasonable'])
        safety_present = sum(1 for v in validations if v['safety_guidance_present'])
        coherent_responses = sum(1 for v in validations if v['response_coherent'])
        
        processing_times = [r['processing_time'] for r in results if 'processing_time' in r]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        print(f"Total Scenarios: {total_scenarios}")
        print(f"Successful: {successful_scenarios}")
        print(f"Failed: {total_scenarios - successful_scenarios}")
        
        if validations:
            print(f"\nValidation Results:")
            print(f"  Urgency Classification: {urgency_correct}/{len(validations)} ({urgency_correct/len(validations)*100:.1f}%)")
            print(f"  Category Classification: {category_correct}/{len(validations)} ({category_correct/len(validations)*100:.1f}%)")
            print(f"  Safety Guidance Present: {safety_present}/{len(validations)} ({safety_present/len(validations)*100:.1f}%)")
            print(f"  Response Coherence: {coherent_responses}/{len(validations)} ({coherent_responses/len(validations)*100:.1f}%)")
            print(f"  Average Processing Time: {avg_processing_time:.2f}s")
    
    def save_results(self, filepath: str = "demo_results.json"):
        try:
            with open(filepath, 'w') as f:
                json.dump(self.demo_results, f, indent=2, default=str)
            print(f"\nDemo results saved to {filepath}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def interactive_demo(self):
        print("\nINTERACTIVE DEMO MODE")
        print("=" * 40)
        print("Ask your own medical questions or type 'scenarios' to run predefined demos")
        print("Type 'quit' to exit")
        
        if not self.assistant:
            if not self.initialize_assistant():
                return
        
        while True:
            try:
                user_input = input("\nYour question (or 'scenarios'/'quit'): ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                elif user_input.lower() == 'scenarios':
                    self.run_full_demo()
                elif user_input:
                    print("\nProcessing your question...")
                    response = self.assistant.generate_response(user_input)
                    
                    print("\nResponse:")
                    print(f"  Answer: {response.response}")
                    print(f"  Category: {response.medical_category}")
                    print(f"  Urgency: {response.urgency_level}")
                    print(f"  Confidence: {response.confidence:.2f}")
                
            except KeyboardInterrupt:
                print("\n\nExiting interactive demo...")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    demo = MedicalAssistantDemo()
    results = demo.run_full_demo()
    
    if results:
        demo.save_results()
    
    print("\nWould you like to try the interactive demo? (y/n)")
    try:
        response = input().strip().lower()
        if response in ['y', 'yes']:
            demo.interactive_demo()
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
