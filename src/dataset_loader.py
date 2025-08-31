import pandas as pd
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MedicalDatasetLoader:

    def __init__(self):
        self.available_datasets = {
        'sample': self._create_sample_dataset,
        'synthetic': self._generate_synthetic_medical_data
        }

    def _create_sample_dataset(self) -> List[Dict[str, str]]:
        sample_data = [
        {
        "input": "I've been experiencing chest pain and shortness of breath for the past few days. Should I be concerned?",
        "response": "Chest pain combined with shortness of breath can be serious symptoms that require immediate medical attention. These could indicate various conditions including heart problems, lung issues, or other medical conditions. I strongly recommend you contact your healthcare provider immediately or visit an emergency room, especially if the symptoms are severe or worsening.",
        "medical_category": "cardiovascular",
        "condition": "chest pain",
        "urgency": "high"
        },
        {
        "input": "My blood pressure reading was 150/95. Is this high blood pressure?",
        "response": "A blood pressure reading of 150/95 mmHg is considered high blood pressure (hypertension). Normal blood pressure is typically below 120/80 mmHg. High blood pressure can lead to serious health complications if left untreated. I recommend consulting with your healthcare provider to discuss treatment options, which may include lifestyle changes and possibly medication.",
        "medical_category": "cardiovascular",
        "condition": "hypertension",
        "urgency": "medium"
        },

        {
        "input": "I have a persistent cough that's lasted for 3 weeks. What could this be?",
        "response": "A persistent cough lasting 3 weeks warrants medical evaluation. While it could be due to various causes like a lingering viral infection, allergies, acid reflux, or other conditions, any cough persisting beyond 2-3 weeks should be examined by a healthcare provider. They may recommend tests such as a chest X-ray to determine the underlying cause and appropriate treatment.",
        "medical_category": "respiratory",
        "condition": "persistent cough",
        "urgency": "medium"
        },
        {
        "input": "I'm having trouble breathing and wheezing. I think it might be asthma.",
        "response": "Difficulty breathing and wheezing are concerning symptoms that need prompt medical attention. While these can be signs of asthma, they could also indicate other respiratory conditions. If you're experiencing severe breathing difficulties, seek emergency care immediately. For ongoing breathing issues, please consult a healthcare provider who can perform proper tests to diagnose asthma or other conditions and provide appropriate treatment.",
        "medical_category": "respiratory",
        "condition": "breathing difficulties",
        "urgency": "high"
        },

        {
        "input": "I've had stomach pain and nausea for 2 days after eating. What should I do?",
        "response": "Stomach pain and nausea lasting 2 days could indicate food poisoning, gastritis, or other gastrointestinal issues. Stay hydrated with clear fluids and consider a bland diet (BRAT - bananas, rice, applesauce, toast). If symptoms worsen, you develop fever, severe dehydration, or persistent vomiting, seek medical care. Contact your healthcare provider if symptoms don't improve within 24-48 hours.",
        "medical_category": "gastrointestinal",
        "condition": "stomach pain",
        "urgency": "low"
        },
        {
        "input": "I have severe abdominal pain in my lower right side. It started this morning.",
        "response": "Severe abdominal pain in the lower right side that started suddenly could be a sign of appendicitis, which is a medical emergency. This type of pain, especially if it's worsening or accompanied by fever, nausea, or vomiting, requires immediate medical evaluation. Please go to the emergency room or contact emergency services right away. Do not eat or drink anything until you've been evaluated by medical professionals.",
        "medical_category": "gastrointestinal",
        "condition": "acute abdominal pain",
        "urgency": "emergency"
        },

        {
        "input": "I've been having frequent headaches and feeling dizzy. Could this be serious?",
        "response": "Frequent headaches combined with dizziness can have various causes, ranging from stress and dehydration to more serious conditions. While often not serious, the combination of symptoms warrants medical evaluation, especially if they're new, worsening, or interfering with daily activities. Keep track of when symptoms occur, potential triggers, and any associated symptoms to discuss with your healthcare provider.",
        "medical_category": "neurological",
        "condition": "headaches",
        "urgency": "medium"
        },
        {
        "input": "I suddenly developed a severe headache, the worst I've ever had, with neck stiffness.",
        "response": "A sudden, severe headache described as 'the worst ever' combined with neck stiffness is a medical emergency. These symptoms could indicate a serious condition such as meningitis or a brain hemorrhage. Please seek emergency medical care immediately by calling emergency services or going to the nearest emergency room. Do not delay seeking treatment.",
        "medical_category": "neurological",
        "condition": "severe headache",
        "urgency": "emergency"
        },

        {
        "input": "I hurt my back lifting heavy boxes yesterday. The pain is getting worse.",
        "response": "Back injuries from lifting can range from muscle strains to more serious issues. If pain is worsening, apply ice for the first 24-48 hours, avoid heavy lifting, and consider over-the-counter pain relievers as directed. If you experience numbness, tingling in legs, loss of bladder/bowel control, or severe pain that doesn't improve with rest, seek medical attention immediately. Otherwise, consult your healthcare provider if pain persists beyond a few days.",
        "medical_category": "musculoskeletal",
        "condition": "back injury",
        "urgency": "low"
        },

        {
        "input": "I found a mole on my arm that has changed color and size recently.",
        "response": "Changes in moles, particularly in color and size, should be evaluated by a dermatologist or healthcare provider promptly. These changes could be signs of skin cancer, though many mole changes are benign. Look for the ABCDE signs: Asymmetry, Border irregularity, Color changes, Diameter larger than 6mm, or Evolving characteristics. Schedule an appointment with a dermatologist for proper evaluation and possible biopsy if needed.",
        "medical_category": "dermatological",
        "condition": "changing mole",
        "urgency": "medium"
        },

        {
        "input": "I've been feeling very sad and have no energy for weeks. I can't seem to enjoy anything.",
        "response": "Persistent sadness, lack of energy, and inability to enjoy activities for weeks are symptoms of depression. These feelings are treatable, and you don't have to go through this alone. I strongly encourage you to speak with a healthcare provider, mental health professional, or counselor. If you're having thoughts of self-harm, please contact a crisis helpline or emergency services immediately. Treatment options including therapy and medication can be very effective.",
        "medical_category": "mental health",
        "condition": "depression symptoms",
        "urgency": "medium"
        },

        {
        "input": "When should I get my cholesterol checked? I'm 35 years old.",
        "response": "For adults age 35 and older, cholesterol screening is generally recommended every 5 years if previous levels were normal. However, if you have risk factors such as family history of heart disease, diabetes, high blood pressure, smoking, or obesity, more frequent screening may be recommended. Consult with your healthcare provider about the appropriate screening schedule for your individual risk profile and health history.",
        "medical_category": "preventive",
        "condition": "cholesterol screening",
        "urgency": "low"
        },
        {
        "input": "What vaccines do I need as an adult?",
        "response": "Adult vaccination needs vary based on age, health conditions, travel, and previous immunizations. Generally recommended vaccines include annual flu shots, Td/Tdap (tetanus/diphtheria) every 10 years, and COVID-19 vaccines as recommended. Adults may also need vaccines for shingles, pneumonia, hepatitis, HPV, or others based on individual risk factors. Consult your healthcare provider or check with your local health department for personalized vaccination recommendations.",
        "medical_category": "preventive",
        "condition": "adult vaccines",
        "urgency": "low"
        },

        {
        "input": "My 2-year-old has had a fever of 102°F for 2 days and isn't eating much.",
        "response": "A fever of 102°F (38.9°C) in a 2-year-old that persists for 2 days, especially with decreased appetite, warrants medical evaluation. For children this age, contact your pediatrician. Signs that require immediate medical attention include difficulty breathing, lethargy, persistent crying, signs of dehydration, or fever above 104°F. Keep your child hydrated and comfortable, and monitor for any worsening symptoms.",
        "medical_category": "pediatric",
        "condition": "pediatric fever",
        "urgency": "medium"
        }
        ]

        logger.info(f"Created sample medical dataset with {len(sample_data)} entries")
        return sample_data

    def _generate_synthetic_medical_data(self) -> List[Dict[str, str]]:
        templates = [
        {
        "input_template": "I have been experiencing {} for {} {}. What could this mean?",
        "response_template": "{} lasting {} {} can have several causes. {} I recommend consulting with a healthcare provider for proper evaluation and diagnosis.",
        "symptoms": ["headaches", "fatigue", "joint pain", "muscle aches", "dizziness"],
        "durations": ["several", "a few", "the past few", "about"],
        "time_units": ["days", "weeks", "hours"],
        "advice": [
        "This could be related to various conditions including viral infections, stress, or other medical issues.",
        "These symptoms may indicate different underlying conditions that require medical assessment.",
        "While often not serious, persistent symptoms should be evaluated by a medical professional."
        ]
        }
        ]

        synthetic_data = []

        base_entries = [
        {
        "input": "I have been experiencing frequent urination and increased thirst. Should I be concerned?",
        "response": "Frequent urination and increased thirst can be symptoms of diabetes or other conditions. These symptoms, especially when occurring together, should be evaluated by a healthcare provider. Simple blood tests can help determine if diabetes or other conditions are present. Early detection and treatment are important for managing these conditions effectively.",
        "medical_category": "endocrine",
        "condition": "diabetes symptoms",
        "urgency": "medium"
        },
        {
        "input": "I've noticed unusual fatigue and weakness for the past month. What should I do?",
        "response": "Persistent fatigue and weakness lasting a month can have many causes including anemia, thyroid problems, sleep disorders, depression, or other medical conditions. Since this has been ongoing for an extended period, I recommend scheduling an appointment with your healthcare provider for evaluation. They may recommend blood tests or other examinations to identify the underlying cause.",
        "medical_category": "general",
        "condition": "chronic fatigue",
        "urgency": "low"
        }
        ]

        synthetic_data.extend(base_entries)
        logger.info(f"Generated {len(synthetic_data)} synthetic medical conversation entries")
        return synthetic_data

    def load_dataset(self, dataset_name: str = 'sample') -> List[Dict[str, str]]:
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset '{dataset_name}' not available. Choose from: {list(self.available_datasets.keys())}")

        logger.info(f"Loading dataset: {dataset_name}")
        data = self.available_datasets[dataset_name]()

        return data

    def combine_datasets(self, dataset_names: List[str]) -> List[Dict[str, str]]:
        combined_data = []

        for name in dataset_names:
            data = self.load_dataset(name)
            combined_data.extend(data)

        logger.info(f"Combined {len(dataset_names)} datasets into {len(combined_data)} total entries")
        return combined_data

    def save_dataset(self, data: List[Dict[str, str]], output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved dataset with {len(data)} entries to {output_path}")

    def get_dataset_info(self, data: List[Dict[str, str]]) -> Dict:
        if not data:
            return {"total_entries": 0}

        df = pd.DataFrame(data)

        info = {
        "total_entries": len(data),
        "columns": list(df.columns),
        "medical_categories": df.get('medical_category', pd.Series([])).value_counts().to_dict(),
        "urgency_levels": df.get('urgency', pd.Series([])).value_counts().to_dict(),
        "avg_input_length": df.get('input', pd.Series([])).str.len().mean(),
        "avg_response_length": df.get('response', pd.Series([])).str.len().mean(),
        }

        return info
