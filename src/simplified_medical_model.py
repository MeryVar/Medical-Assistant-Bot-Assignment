import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelOutput:
    response: str
    urgency_level: str
    medical_category: str
    confidence: float

class SimpleMedicalAssistant:
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.medical_patterns = self._load_medical_patterns()
        
        try:
            print(f" Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(" Model loaded successfully")
        except Exception as e:
            print(f" Error loading model: {e}")
            logger.error(f"Model loading failed: {e}")

    def _load_medical_patterns(self) -> Dict:
        return {
            'emergency_keywords': [
                'severe chest pain', 'difficulty breathing', 'chest pain', 'can\'t breathe',
                'heart attack', 'stroke', 'severe bleeding', 'unconscious', 'severe pain',
                'emergency', 'urgent', 'critical', 'life threatening'
            ],
            'high_urgency_keywords': [
                'persistent pain', 'high fever', 'severe headache', 'vision loss',
                'numbness', 'weakness', 'severe', 'worsening', 'getting worse'
            ],
            'medical_categories': {
                'cardiovascular': ['heart', 'chest pain', 'blood pressure', 'cardiac', 'palpitations'],
                'respiratory': ['breathing', 'cough', 'lung', 'asthma', 'shortness of breath'],
                'neurological': ['headache', 'dizziness', 'brain', 'nerve', 'seizure'],
                'gastrointestinal': ['stomach', 'nausea', 'vomiting', 'digestive', 'abdominal'],
                'musculoskeletal': ['back pain', 'joint', 'muscle', 'bone', 'arthritis'],
                'dermatological': ['skin', 'rash', 'acne', 'eczema', 'dermatitis'],
                'general': ['fever', 'fatigue', 'weakness', 'tired', 'sick']
            },
            'medical_advice_templates': {
                'emergency': "This requires immediate medical attention. Please seek emergency care or call emergency services right away.",
                'high': "These symptoms should be evaluated by a healthcare provider promptly. Please schedule an appointment or visit urgent care.",
                'medium': "I recommend consulting with a healthcare provider for proper evaluation and treatment options.",
                'low': "While these symptoms are often manageable, consult a healthcare provider if they persist or worsen."
            }
        }

    def _classify_urgency(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()
        
        emergency_keywords = self.medical_patterns['emergency_keywords']
        emergency_matches = sum(1 for keyword in emergency_keywords if keyword in text_lower)
        
        if emergency_matches > 0:
            confidence = min(0.9, 0.7 + (emergency_matches * 0.1))
            return 'emergency', confidence
        
        high_keywords = self.medical_patterns['high_urgency_keywords']
        high_matches = sum(1 for keyword in high_keywords if keyword in text_lower)
        
        if high_matches > 0:
            confidence = min(0.8, 0.6 + (high_matches * 0.1))
            return 'high', confidence
        
        medical_indicators = ['doctor', 'hospital', 'treatment', 'medication', 'symptoms']
        medical_matches = sum(1 for indicator in medical_indicators if indicator in text_lower)
        
        if medical_matches > 1:
            return 'medium', 0.6
        elif medical_matches > 0:
            return 'low', 0.5
        
        return 'low', 0.4

    def _classify_medical_category(self, text: str) -> str:
        text_lower = text.lower()
        
        category_scores = {}
        
        for category, keywords in self.medical_patterns['medical_categories'].items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return 'general'

    def _generate_medical_response(self, input_text: str, urgency_level: str, medical_category: str) -> str:
        advice_template = self.medical_patterns['medical_advice_templates'].get(
            urgency_level, 
            self.medical_patterns['medical_advice_templates']['medium']
        )
        
        category_responses = {
            'cardiovascular': f"Regarding cardiovascular concerns: {advice_template} Heart-related symptoms should always be taken seriously.",
            'respiratory': f"For respiratory symptoms: {advice_template} Breathing difficulties require proper medical evaluation.",
            'neurological': f"Concerning neurological symptoms: {advice_template} Neurological symptoms warrant professional assessment.",
            'gastrointestinal': f"For digestive concerns: {advice_template} Persistent digestive issues should be evaluated by a healthcare provider.",
            'musculoskeletal': f"Regarding musculoskeletal symptoms: {advice_template} Joint and muscle problems benefit from proper diagnosis.",
            'dermatological': f"For skin-related concerns: {advice_template} Skin conditions often require dermatological evaluation.",
            'general': f"Regarding your health concern: {advice_template}"
        }
        
        base_response = category_responses.get(medical_category, category_responses['general'])
        
        safety_disclaimer = " Always consult with a qualified healthcare professional for personalized medical advice and proper diagnosis."
        
        return base_response + safety_disclaimer

    def generate_response(self, input_text: str) -> ModelOutput:
        
        if not input_text or not input_text.strip():
            return ModelOutput(
                response="Please provide a specific medical question or concern.",
                urgency_level="low",
                medical_category="general", 
                confidence=0.5
            )
        
        try:
            urgency_level, urgency_confidence = self._classify_urgency(input_text)
            medical_category = self._classify_medical_category(input_text)
            
            response = self._generate_medical_response(input_text, urgency_level, medical_category)
            
            return ModelOutput(
                response=response,
                urgency_level=urgency_level,
                medical_category=medical_category,
                confidence=urgency_confidence
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ModelOutput(
                response="I apologize, but I'm unable to process your request at this time. Please consult with a healthcare professional for medical advice.",
                urgency_level="medium",
                medical_category="general",
                confidence=0.3
            )

    def interactive_consultation(self):
        print("\n MEDICAL ASSISTANT - INTERACTIVE CONSULTATION")
        print("=" * 60)
        print("Ask medical questions and receive guidance with urgency assessment.")
        print("Type 'quit' to exit, 'help' for commands.")
        print("\nDISCLAIMER: This is an AI assistant providing general information only.")
        print("Always consult healthcare professionals for medical decisions.")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n Your medical question: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n Thank you for using the Medical Assistant!")
                    print("Remember to consult healthcare professionals for medical advice.")
                    break
                
                if user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  help - Show this help")
                    print("  quit - Exit the consultation")
                    print("  Just type your medical question for assistance!")
                    continue
                
                result = self.generate_response(user_input)
                
                print("\n" + "="*60)
                print(" MEDICAL ASSISTANT RESPONSE")
                print("="*60)
                print(f" Response: {result.response}")
                print(f" Urgency Level: {result.urgency_level.upper()}")
                print(f" Medical Category: {result.medical_category.title()}")
                print(f" Confidence: {result.confidence:.2f}")
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n\nExiting consultation...")
                break
            except Exception as e:
                print(f" Error: {e}")
                continue

def main():
    try:
        assistant = SimpleMedicalAssistant()
        
        test_question = "I have chest pain and shortness of breath"
        print(f"\nTesting with: '{test_question}'")
        
        response = assistant.generate_response(test_question)
        print(f"\nResponse: {response.response}")
        print(f"Urgency: {response.urgency_level}")
        print(f"Category: {response.medical_category}")
        print(f"Confidence: {response.confidence:.2f}")
        
        print("\nStarting interactive consultation...")
        assistant.interactive_consultation()
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
