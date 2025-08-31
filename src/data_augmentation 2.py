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
        'combination_inquiry': [
        "I have {} and {} together. Could these be related?",
        "Along with {}, I'm also experiencing {}. What could this indicate?",
        "I've developed {} and {} at the same time. Is this concerning?",
        "Both {} and {} started recently. What might be causing this?",
        ]
        }

    def _load_medical_knowledge(self) -> Dict:
        return {
        'symptoms': {
        'headaches': {
        'causes': ['tension', 'migraines', 'dehydration', 'stress', 'eye strain'],
        'urgency': 'medium',
        'category': 'neurological'
        },
        'chest pain': {
        'causes': ['heart conditions', 'lung problems', 'muscle strain', 'anxiety'],
        'urgency': 'high',
        'category': 'cardiovascular'
        },
        'shortness of breath': {
        'causes': ['asthma', 'heart problems', 'lung conditions', 'anxiety'],
        'urgency': 'high',
        'category': 'respiratory'
        },
        'stomach pain': {
        'causes': ['gastritis', 'food poisoning', 'ulcers', 'stress'],
        'urgency': 'medium',
        'category': 'gastrointestinal'
        },
        'back pain': {
        'causes': ['muscle strain', 'poor posture', 'injury', 'disc problems'],
        'urgency': 'low',
        'category': 'musculoskeletal'
        },
        'fever': {
        'causes': ['viral infection', 'bacterial infection', 'inflammatory conditions'],
        'urgency': 'medium',
        'category': 'general'
        },
        'fatigue': {
        'causes': ['poor sleep', 'anemia', 'thyroid problems', 'depression'],
        'urgency': 'low',
        'category': 'general'
        },
        'dizziness': {
        'causes': ['inner ear problems', 'low blood pressure', 'dehydration', 'medications'],
        'urgency': 'medium',
        'category': 'neurological'
        },
        'nausea': {
        'causes': ['gastroenteritis', 'pregnancy', 'medication side effects', 'motion sickness'],
        'urgency': 'low',
        'category': 'gastrointestinal'
        },
        'skin rash': {
        'causes': ['allergic reactions', 'infections', 'autoimmune conditions', 'irritants'],
        'urgency': 'low',
        'category': 'dermatological'
        }
        },
        'duration_modifiers': ['sudden', 'gradual', 'intermittent', 'constant', 'recurring'],
        'severity_levels': ['mild', 'moderate', 'severe', 'intense', 'unbearable'],
        'time_periods': {
        'hours': ['few hours', '6 hours', '12 hours', '24 hours'],
        'days': ['2 days', '3 days', 'few days', 'several days', 'a week'],
        'weeks': ['2 weeks', '3 weeks', 'few weeks', 'several weeks', 'a month'],
        'months': ['2 months', '3 months', 'few months', 'several months']
        }
        }

    def _load_response_templates(self) -> Dict:
        return {
        'emergency': [
        "These symptoms require immediate medical attention. Please go to the emergency room or call emergency services right away. {}",
        "This combination of symptoms is concerning and needs urgent evaluation. Seek emergency medical care immediately. {}",
        "These are potentially serious symptoms that warrant immediate medical attention. Do not delay seeking emergency care. {}"
        ],
        'high': [
        "These symptoms should be evaluated by a healthcare provider promptly. {} Please schedule an appointment or visit urgent care if symptoms worsen.",
        "{} I recommend contacting your healthcare provider today or visiting urgent care if you cannot get an appointment quickly.",
        "While not immediately life-threatening, these symptoms need medical evaluation. {} Please see a healthcare provider within 24-48 hours."
        ],
        'medium': [
        "{} These symptoms should be discussed with your healthcare provider. Schedule an appointment if symptoms persist or worsen.",
        "These symptoms can have various causes. {} I recommend consulting with your healthcare provider for proper evaluation.",
        "{} While often not serious, it's best to have these symptoms evaluated by a medical professional if they continue."
        ],
        'low': [
        "These symptoms are often manageable with self-care, but {} If symptoms persist or worsen, consult your healthcare provider.",
        "{} These symptoms typically improve with rest and self-care. Contact your healthcare provider if they don't resolve.",
        "While usually not serious, {} Monitor your symptoms and consult a healthcare provider if they persist or worsen."
        ]
        }

    def generate_symptom_variations(self, base_symptoms: List[str], count: int = 50) -> List[Dict]:
        variations = []

        for _ in range(count):
        template_type = random.choice(list(self.symptom_templates.keys()))
        template = random.choice(self.symptom_templates[template_type])

        if template_type == 'duration_inquiry':
        symptom = random.choice(list(self.medical_knowledge['symptoms'].keys()))
        duration_num = random.choice(['2', '3', 'few', 'several'])
        duration_unit = random.choice(['hours', 'days', 'weeks'])

        input_text = template.format(symptom, duration_num, duration_unit)
        response_text = self._generate_medical_response(symptom, 'duration')

        elif template_type == 'severity_inquiry':
        symptom = random.choice(list(self.medical_knowledge['symptoms'].keys()))
        severity = random.choice(self.medical_knowledge['severity_levels'])
        body_part = random.choice(['chest', 'head', 'abdomen', 'back', 'leg', 'arm'])

        input_text = template.format(symptom, severity, body_part, severity, body_part, severity)
        response_text = self._generate_medical_response(symptom, 'severity')

        elif template_type == 'combination_inquiry':
        symptoms = random.sample(list(self.medical_knowledge['symptoms'].keys()), 2)
        input_text = template.format(symptoms[0], symptoms[1])
        response_text = self._generate_medical_response(symptoms[0], 'combination', symptoms[1])

        symptom_info = self.medical_knowledge['symptoms'][symptoms[0] if 'symptoms' in locals() else symptom]

        variation = {
        'input': input_text,
        'response': response_text,
        'medical_category': symptom_info['category'],
        'condition': symptoms[0] if 'symptoms' in locals() else symptom,
        'urgency': symptom_info['urgency']
        }
        variations.append(variation)

        return variations

    def _generate_medical_response(self, primary_symptom: str, inquiry_type: str, secondary_symptom: str = None) -> str:
        symptom_info = self.medical_knowledge['symptoms'][primary_symptom]
        urgency = symptom_info['urgency']

        template = random.choice(self.response_templates[urgency])

        if inquiry_type == 'duration':
        advice = f"{primary_symptom.capitalize()} can be caused by {', '.join(symptom_info['causes'][:3])}."
        elif inquiry_type == 'severity':
        advice = f"Severe {primary_symptom} may indicate {random.choice(symptom_info['causes'])} or other conditions."
        elif inquiry_type == 'combination':
        advice = f"The combination of {primary_symptom} and {secondary_symptom} can occur together in various conditions."
        else:
        advice = f"{primary_symptom.capitalize()} has multiple potential causes including {', '.join(symptom_info['causes'][:2])}."

        if urgency in ['emergency', 'high']:
        disclaimer = "This is not a substitute for professional medical advice."
        else:
        disclaimer = "Consider lifestyle factors and monitor symptoms carefully."

        return template.format(advice) + f" {disclaimer}"

    def generate_preventive_care_data(self, count: int = 20) -> List[Dict]:
        preventive_topics = [
        {
        'topic': 'vaccination',
        'questions': [
        "What vaccines do I need as a {age}-year-old adult?",
        "When should I get my {vaccine_type} vaccine updated?",
        "Are there any vaccines recommended for travel to {location}?"
        ],
        'age_groups': ['25', '35', '45', '65'],
        'vaccine_types': ['flu', 'tetanus', 'COVID-19', 'pneumonia'],
        'locations': ['Europe', 'Asia', 'Africa', 'South America']
        },
        {
        'topic': 'screening',
        'questions': [
        "When should I get screened for {condition}?",
        "How often should I have {test_type} done?",
        "At what age should I start {screening_type} screening?"
        ],
        'conditions': ['diabetes', 'high cholesterol', 'heart disease', 'cancer'],
        'test_types': ['blood pressure checks', 'cholesterol tests', 'mammograms', 'colonoscopies'],
        'screening_types': ['colorectal', 'breast', 'cervical', 'prostate']
        },
        {
        'topic': 'lifestyle',
        'questions': [
        "What lifestyle changes can help prevent {condition}?",
        "How much {activity} is recommended for someone my age?",
        "What dietary changes can improve {health_aspect}?"
        ],
        'conditions': ['heart disease', 'diabetes', 'osteoporosis', 'high blood pressure'],
        'activities': ['exercise', 'walking', 'strength training', 'cardio'],
        'health_aspects': ['heart health', 'bone health', 'brain health', 'digestive health']
        }
        ]

        preventive_data = []

        for _ in range(count):
        topic_data = random.choice(preventive_topics)
        question_template = random.choice(topic_data['questions'])

        if '{age}' in question_template:
        question = question_template.format(age=random.choice(topic_data['age_groups']))
        elif '{vaccine_type}' in question_template:
        question = question_template.format(vaccine_type=random.choice(topic_data['vaccine_types']))
        elif '{location}' in question_template:
        question = question_template.format(location=random.choice(topic_data['locations']))
        elif '{condition}' in question_template:
        question = question_template.format(condition=random.choice(topic_data['conditions']))
        elif '{test_type}' in question_template:
        question = question_template.format(test_type=random.choice(topic_data['test_types']))
        elif '{screening_type}' in question_template:
        question = question_template.format(screening_type=random.choice(topic_data['screening_types']))
        elif '{activity}' in question_template:
        question = question_template.format(activity=random.choice(topic_data['activities']))
        elif '{health_aspect}' in question_template:
        question = question_template.format(health_aspect=random.choice(topic_data['health_aspects']))
        else:
        question = question_template

        response = self._generate_preventive_response(topic_data['topic'], question)

        entry = {
        'input': question,
        'response': response,
        'medical_category': 'preventive',
        'condition': topic_data['topic'],
        'urgency': 'low'
        }
        preventive_data.append(entry)

        return preventive_data

    def _generate_preventive_response(self, topic: str, question: str) -> str:
        responses = {
        'vaccination': "Vaccination schedules vary based on age, health conditions, and risk factors. For personalized recommendations, consult with your healthcare provider or local health department. They can review your immunization history and recommend appropriate vaccines based on current guidelines.",
        'screening': "Screening recommendations depend on age, family history, and individual risk factors. Your healthcare provider can determine the appropriate screening schedule for your specific situation. Early detection through regular screening is important for many conditions.",
        'lifestyle': "Lifestyle modifications play a crucial role in preventing many health conditions. A combination of regular physical activity, balanced nutrition, adequate sleep, and stress management is generally recommended. Consult with your healthcare provider for personalized recommendations based on your health profile."
        }

        return responses.get(topic, "For personalized medical advice regarding preventive care, please consult with your healthcare provider who can assess your individual risk factors and health needs.")

    def generate_pediatric_data(self, count: int = 15) -> List[Dict]:
        pediatric_scenarios = [
        "My {age} has had a fever of {temp}°F for {duration}. When should I call the pediatrician?",
        "My {age} is {symptom}. Is this normal for their age?",
        "How do I know if my {age}'s {symptom} needs medical attention?",
        "My {age} has been {symptom} for {duration}. Should I be concerned?",
        ]

        ages = ['2-year-old', '5-year-old', '8-year-old', '12-year-old', 'toddler', 'infant']
        temps = ['100.5', '101', '102', '103']
        durations = ['2 days', '3 days', 'since yesterday', 'all week']
        symptoms = ['not eating well', 'very fussy', 'not sleeping well', 'coughing', 'complaining of stomach pain']

        pediatric_data = []

        for _ in range(count):
        template = random.choice(pediatric_scenarios)

        if '{temp}' in template:
        question = template.format(
        age=random.choice(ages),
        temp=random.choice(temps),
        duration=random.choice(durations)
        )
        else:
        question = template.format(
        age=random.choice(ages),
        symptom=random.choice(symptoms),
        duration=random.choice(durations) if '{duration}' in template else ''
        ).strip()

        response = ("Pediatric symptoms should always be evaluated with extra caution. For infants under 3 months, any fever warrants immediate medical attention. For older children, contact your pediatrician if fever is high, persistent, or accompanied by other concerning symptoms. Trust your parental instincts - if you're worried, it's always appropriate to contact your child's healthcare provider.")

        entry = {
        'input': question,
        'response': response,
        'medical_category': 'pediatric',
        'condition': 'pediatric concerns',
        'urgency': 'medium'
        }
        pediatric_data.append(entry)

        return pediatric_data

    def augment_dataset(self, original_data: List[Dict], target_size: int = 200) -> List[Dict]:
        current_size = len(original_data)
        needed_entries = max(0, target_size - current_size)

        if needed_entries == 0:
        logger.info("Dataset already meets target size")
        return original_data

        logger.info(f"Augmenting dataset from {current_size} to {target_size} entries")

        symptom_variations = min(needed_entries // 2, 80)
        preventive_care = min(needed_entries // 4, 30)
        pediatric = min(needed_entries - symptom_variations - preventive_care, 20)

        augmented_data = original_data.copy()

        if symptom_variations > 0:
        augmented_data.extend(self.generate_symptom_variations([], symptom_variations))

        if preventive_care > 0:
        augmented_data.extend(self.generate_preventive_care_data(preventive_care))

        if pediatric > 0:
        augmented_data.extend(self.generate_pediatric_data(pediatric))

        remaining = target_size - len(augmented_data)
        if remaining > 0:
        augmented_data.extend(self.generate_symptom_variations([], remaining))

        logger.info(f"Dataset augmented to {len(augmented_data)} entries")
        return augmented_data

    def save_augmented_data(self, data: List[Dict], output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

        logger.info(f"Saved augmented dataset to {output_path}")

    def get_augmentation_stats(self, original_data: List[Dict], augmented_data: List[Dict]) -> Dict:
        original_df = pd.DataFrame(original_data) if original_data else pd.DataFrame()
        augmented_df = pd.DataFrame(augmented_data)

        stats = {
        'original_size': len(original_data),
        'augmented_size': len(augmented_data),
        'added_entries': len(augmented_data) - len(original_data),
        'original_categories': original_df['medical_category'].value_counts().to_dict() if not original_df.empty else {},
        'augmented_categories': augmented_df['medical_category'].value_counts().to_dict(),
        'original_urgency': original_df['urgency'].value_counts().to_dict() if not original_df.empty else {},
        'augmented_urgency': augmented_df['urgency'].value_counts().to_dict()
        }

        return stats
