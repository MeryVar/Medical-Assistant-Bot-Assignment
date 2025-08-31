import random
import json
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MedicalDataAugmentor:

    def __init__(self):
        self.symptom_templates = self._load_symptom_templates()
        self.medical_knowledge = self._load_medical_knowledge()
        self.response_templates = self._load_response_templates()

    def _load_symptom_templates(self) -> Dict:
        return {
            'duration_inquiry': [
                "I have been experiencing {} for {} {}. What could this be?",
                "I've had {} for the past {} {}. Should I be concerned?",
                "For {} {} I've been dealing with {}. Is this serious?",
                "I've noticed {} lasting {} {}. What should I do?",
            ],
            'severity_inquiry': [
                "I'm having {} {} pain. When should I see a doctor?",
                "My {} is {} and it's getting worse. What could this mean?",
                "I have {} {} in my {}. Is this normal?",
                "The {} in my {} is quite {}. Should I worry?",
            ],
            'general_inquiry': [
                "What are the symptoms of {}?",
                "How is {} treated?",
                "What causes {}?",
                "Can you tell me about {}?",
                "Is {} serious?",
                "How do I know if I have {}?",
            ],
            'comparison_inquiry': [
                "What's the difference between {} and {}?",
                "How can I tell {} from {}?",
                "Is {} worse than {}?",
                "Which is more serious: {} or {}?",
            ],
            'prevention_inquiry': [
                "How can I prevent {}?",
                "What can I do to avoid {}?",
                "Are there ways to reduce my risk of {}?",
                "What lifestyle changes help with {}?",
            ]
        }

    def _load_medical_knowledge(self) -> Dict:
        return {
            'symptoms': [
                'headache', 'fever', 'cough', 'chest pain', 'shortness of breath',
                'nausea', 'vomiting', 'diarrhea', 'fatigue', 'dizziness',
                'back pain', 'joint pain', 'muscle pain', 'sore throat',
                'runny nose', 'rash', 'abdominal pain', 'constipation'
            ],
            'conditions': [
                'diabetes', 'hypertension', 'asthma', 'pneumonia', 'bronchitis',
                'migraine', 'arthritis', 'flu', 'common cold', 'allergies',
                'depression', 'anxiety', 'insomnia', 'heartburn', 'ulcer'
            ],
            'body_parts': [
                'head', 'chest', 'back', 'stomach', 'throat', 'leg', 'arm',
                'shoulder', 'neck', 'knee', 'foot', 'hand', 'eye', 'ear'
            ],
            'durations': [
                ('2', 'days'), ('3', 'days'), ('1', 'week'), ('2', 'weeks'),
                ('1', 'month'), ('few', 'days'), ('several', 'hours')
            ],
            'severity_levels': [
                'mild', 'moderate', 'severe', 'intense', 'sharp', 'dull',
                'throbbing', 'burning', 'stabbing', 'aching'
            ]
        }

    def _load_response_templates(self) -> Dict:
        return {
            'general_advice': [
                "It's important to consult with a healthcare provider for proper diagnosis and treatment.",
                "I recommend speaking with your doctor about these symptoms.",
                "Please seek medical attention to get a proper evaluation.",
                "A healthcare professional can provide the best guidance for your situation.",
                "Consider scheduling an appointment with your physician.",
            ],
            'emergency_advice': [
                "These symptoms may require immediate medical attention. Please seek emergency care.",
                "This could be serious. Go to the emergency room or call emergency services.",
                "Don't delay - seek immediate medical help for these symptoms.",
                "This warrants urgent medical evaluation. Please get help right away.",
            ],
            'routine_advice': [
                "These symptoms are often manageable with proper care and monitoring.",
                "Many people experience this, and there are effective treatments available.",
                "This is a common condition that can be well-managed with medical guidance.",
                "With proper treatment, this condition can be effectively controlled.",
            ],
            'lifestyle_advice': [
                "Lifestyle modifications may help improve your symptoms.",
                "Diet and exercise changes can often make a significant difference.",
                "Stress management and healthy habits are important for this condition.",
                "Preventive measures can help reduce the risk of complications.",
            ]
        }

    def generate_medical_conversation(self, template_type: str = None) -> Dict:
        if template_type is None:
            template_type = random.choice(list(self.symptom_templates.keys()))
        
        template = random.choice(self.symptom_templates[template_type])
        
        if template_type == 'duration_inquiry':
            return self._generate_duration_conversation(template)
        elif template_type == 'severity_inquiry':
            return self._generate_severity_conversation(template)
        elif template_type == 'general_inquiry':
            return self._generate_general_conversation(template)
        elif template_type == 'comparison_inquiry':
            return self._generate_comparison_conversation(template)
        elif template_type == 'prevention_inquiry':
            return self._generate_prevention_conversation(template)
        else:
            return self._generate_general_conversation(template)

    def _generate_duration_conversation(self, template: str) -> Dict:
        symptom = random.choice(self.medical_knowledge['symptoms'])
        duration, unit = random.choice(self.medical_knowledge['durations'])
        
        question = template.format(symptom, duration, unit)
        
        response = self._generate_appropriate_response(symptom, urgency='medium')
        
        return {
            'input': question,
            'response': response,
            'medical_category': self._categorize_symptom(symptom),
            'urgency': 'medium',
            'template_type': 'duration_inquiry'
        }

    def _generate_severity_conversation(self, template: str) -> Dict:
        body_part = random.choice(self.medical_knowledge['body_parts'])
        severity = random.choice(self.medical_knowledge['severity_levels'])
        
        if len(template.split('{}')) == 3:
            question = template.format(body_part, severity, body_part)
        elif len(template.split('{}')) == 4:
            symptom = random.choice(self.medical_knowledge['symptoms'])
            question = template.format(symptom, severity, body_part, body_part)
        else:
            question = template.format(body_part, severity)
        
        urgency = 'high' if severity in ['severe', 'intense', 'sharp'] else 'medium'
        response = self._generate_appropriate_response(body_part, urgency=urgency)
        
        return {
            'input': question,
            'response': response,
            'medical_category': self._categorize_symptom(body_part),
            'urgency': urgency,
            'template_type': 'severity_inquiry'
        }

    def _generate_general_conversation(self, template: str) -> Dict:
        condition = random.choice(self.medical_knowledge['conditions'])
        question = template.format(condition)
        
        response = self._generate_appropriate_response(condition, urgency='low')
        
        return {
            'input': question,
            'response': response,
            'medical_category': self._categorize_symptom(condition),
            'urgency': 'low',
            'template_type': 'general_inquiry'
        }

    def _generate_comparison_conversation(self, template: str) -> Dict:
        conditions = random.sample(self.medical_knowledge['conditions'], 2)
        question = template.format(conditions[0], conditions[1])
        
        response = self._generate_appropriate_response(conditions[0], urgency='low')
        
        return {
            'input': question,
            'response': response,
            'medical_category': 'general',
            'urgency': 'low',
            'template_type': 'comparison_inquiry'
        }

    def _generate_prevention_conversation(self, template: str) -> Dict:
        condition = random.choice(self.medical_knowledge['conditions'])
        question = template.format(condition)
        
        response = random.choice(self.response_templates['lifestyle_advice'])
        response += " " + random.choice(self.response_templates['general_advice'])
        
        return {
            'input': question,
            'response': response,
            'medical_category': self._categorize_symptom(condition),
            'urgency': 'low',
            'template_type': 'prevention_inquiry'
        }

    def _generate_appropriate_response(self, medical_term: str, urgency: str = 'medium') -> str:
        if urgency == 'high':
            base_response = random.choice(self.response_templates['emergency_advice'])
        elif urgency == 'low':
            base_response = random.choice(self.response_templates['routine_advice'])
        else:
            base_response = random.choice(self.response_templates['general_advice'])
        
        return base_response

    def _categorize_symptom(self, term: str) -> str:
        cardiovascular_terms = ['chest pain', 'heart', 'cardiac', 'blood pressure']
        respiratory_terms = ['cough', 'shortness of breath', 'breathing', 'lung']
        neurological_terms = ['headache', 'dizziness', 'migraine', 'head']
        gastrointestinal_terms = ['nausea', 'vomiting', 'diarrhea', 'stomach', 'abdominal']
        musculoskeletal_terms = ['back pain', 'joint pain', 'muscle pain', 'arthritis']
        
        term_lower = term.lower()
        
        if any(t in term_lower for t in cardiovascular_terms):
            return 'cardiovascular'
        elif any(t in term_lower for t in respiratory_terms):
            return 'respiratory'
        elif any(t in term_lower for t in neurological_terms):
            return 'neurological'
        elif any(t in term_lower for t in gastrointestinal_terms):
            return 'gastrointestinal'
        elif any(t in term_lower for t in musculoskeletal_terms):
            return 'musculoskeletal'
        else:
            return 'general'

    def augment_dataset(self, original_data: List[Dict], target_size: int = 1000) -> List[Dict]:
        augmented_data = original_data.copy()
        
        while len(augmented_data) < target_size:
            new_conversation = self.generate_medical_conversation()
            augmented_data.append(new_conversation)
        
        print(f"Augmented dataset from {len(original_data)} to {len(augmented_data)} examples")
        return augmented_data

    def create_balanced_dataset(self, target_size: int = 500) -> List[Dict]:
        balanced_data = []
        
        categories = ['cardiovascular', 'respiratory', 'neurological', 'gastrointestinal', 'musculoskeletal', 'general']
        examples_per_category = target_size // len(categories)
        
        for category in categories:
            category_count = 0
            while category_count < examples_per_category:
                conversation = self.generate_medical_conversation()
                if conversation['medical_category'] == category:
                    balanced_data.append(conversation)
                    category_count += 1
        
        random.shuffle(balanced_data)
        print(f"Created balanced dataset with {len(balanced_data)} examples across {len(categories)} categories")
        return balanced_data

    def save_augmented_data(self, data: List[Dict], filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} augmented examples to {filepath}")

    def load_existing_data(self, filepath: str) -> List[Dict]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} existing examples from {filepath}")
            return data
        except FileNotFoundError:
            print(f"File {filepath} not found. Starting with empty dataset.")
            return []
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            return []

    def analyze_dataset(self, data: List[Dict]) -> Dict:
        if not data:
            return {}
        
        analysis = {
            'total_examples': len(data),
            'category_distribution': {},
            'urgency_distribution': {},
            'template_distribution': {},
            'average_input_length': 0,
            'average_response_length': 0
        }
        
        input_lengths = []
        response_lengths = []
        
        for item in data:
            category = item.get('medical_category', 'unknown')
            urgency = item.get('urgency', 'unknown')
            template = item.get('template_type', 'unknown')
            
            analysis['category_distribution'][category] = analysis['category_distribution'].get(category, 0) + 1
            analysis['urgency_distribution'][urgency] = analysis['urgency_distribution'].get(urgency, 0) + 1
            analysis['template_distribution'][template] = analysis['template_distribution'].get(template, 0) + 1
            
            input_lengths.append(len(str(item.get('input', ''))))
            response_lengths.append(len(str(item.get('response', ''))))
        
        analysis['average_input_length'] = sum(input_lengths) / len(input_lengths) if input_lengths else 0
        analysis['average_response_length'] = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        
        return analysis

def create_augmented_medical_dataset():
    augmentor = MedicalDataAugmentor()
    
    print("Creating augmented medical conversation dataset...")
    
    existing_data = augmentor.load_existing_data('data/raw/medical_conversations.json')
    
    if len(existing_data) < 150:
        target_size = 150
        augmented_data = augmentor.augment_dataset(existing_data, target_size)
    else:
        augmented_data = existing_data
        print(f"Dataset already has {len(existing_data)} examples")
    
    analysis = augmentor.analyze_dataset(augmented_data)
    
    print("\nDataset Analysis:")
    print(f"Total examples: {analysis['total_examples']}")
    print("\nCategory distribution:")
    for category, count in analysis['category_distribution'].items():
        print(f"  {category}: {count}")
    
    print("\nUrgency distribution:")
    for urgency, count in analysis['urgency_distribution'].items():
        print(f"  {urgency}: {count}")
    
    augmentor.save_augmented_data(augmented_data, 'data/raw/augmented_medical_conversations.json')
    
    return augmented_data, analysis

if __name__ == "__main__":
    create_augmented_medical_dataset()
