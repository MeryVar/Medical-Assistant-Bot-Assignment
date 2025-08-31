import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
AutoTokenizer, AutoModelForCausalLM, AutoModel,
TrainingArguments, Trainer, get_linear_schedule_with_warmup
)
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelOutput:
    response: str
    urgency_level: str
    confidence_score: float
    medical_category: str
    retrieved_contexts: List[str]

class MedicalKnowledgeBase:

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.knowledge_base = []
        self.embeddings = None
        self.index = None

    def build_knowledge_base(self, medical_data: List[Dict]):
        logger.info("Building medical knowledge base...")

        self.knowledge_base = []
        texts_to_encode = []

        for item in medical_data:
            knowledge_entry = {
                'text': f"Q: {item['input']} A: {item['response']}",
                'category': item.get('medical_category', 'general'),
                'urgency': item.get('urgency', 'low'),
                'condition': item.get('condition', 'general')
            }

            self.knowledge_base.append(knowledge_entry)
            texts_to_encode.append(knowledge_entry['text'])

        logger.info(f"Encoding {len(texts_to_encode)} knowledge entries...")
        self.embeddings = self.encoder.encode(texts_to_encode, show_progress_bar=True)

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))

        logger.info(f"Knowledge base built with {len(self.knowledge_base)} entries")

    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.index is None:
            return []

        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)

        similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)

        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.knowledge_base):
                context = self.knowledge_base[idx].copy()
                context['similarity_score'] = float(similarity)
                context['rank'] = i + 1
                results.append(context)

        return results

    def save_knowledge_base(self, path: str):
        save_data = {
        'knowledge_base': self.knowledge_base,
        'embeddings': self.embeddings.tolist() if self.embeddings is not None else None
        }

        with open(path, 'w') as f:
            json.dump(save_data, f)

        if self.index is not None:
            faiss.write_index(self.index, path.replace('.json', '.faiss'))  

        logger.info(f"Knowledge base saved to {path}")

class MedicalConversationDataset(Dataset):

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.urgency_to_id = {'low': 0, 'medium': 1, 'high': 2, 'emergency': 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        input_text = f"Patient: {item['input']}"
        target_text = f"Doctor: {item['response']}"
        full_text = f"{input_text} {target_text}"

        encoding = self.tokenizer(
        full_text,
        truncation=True,
        padding='max_length',
        max_length=self.max_length,
        return_tensors='pt'
        )

        urgency_label = self.urgency_to_id.get(item.get('urgency', 'low'), 0)

        return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'labels': encoding['input_ids'].flatten().clone(),
        'urgency_label': torch.tensor(urgency_label, dtype=torch.long)
        }

class MedicalAssistantModel(nn.Module):

    def __init__(self,
        model_name: str = "microsoft/DialoGPT-medium",
        num_urgency_classes: int = 4,
        knowledge_base: Optional[MedicalKnowledgeBase] = None):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.language_model = AutoModelForCausalLM.from_pretrained(model_name)

        self.urgency_classifier = nn.Sequential(
        nn.Linear(self.language_model.config.hidden_size, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_urgency_classes)
        )

        self.knowledge_base = knowledge_base

        self.id_to_urgency = {0: 'low', 1: 'medium', 2: 'high', 3: 'emergency'}

        self.training_step = 0

    def forward(self, input_ids, attention_mask, labels=None, urgency_label=None):
        outputs = self.language_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True
        )

        last_hidden_state = outputs.hidden_states[-1]

        pooled_output = torch.mean(last_hidden_state, dim=1)

        urgency_logits = self.urgency_classifier(pooled_output)

        total_loss = None
        if labels is not None and urgency_label is not None:
            lm_loss = outputs.loss
            urgency_loss = F.cross_entropy(urgency_logits, urgency_label)

            total_loss = lm_loss + 0.3 * urgency_loss

        return {
        'loss': total_loss,
        'logits': outputs.logits,
        'urgency_logits': urgency_logits,
        'hidden_states': outputs.hidden_states
        }

    def generate_response(self,
        patient_input: str,
        max_length: int = 150,
        temperature: float = 0.7,
        use_rag: bool = True) -> ModelOutput:

        retrieved_contexts = []
        if use_rag and self.knowledge_base is not None:
            contexts = self.knowledge_base.retrieve_relevant_context(patient_input, top_k=3)
            retrieved_contexts = [ctx['text'] for ctx in contexts]

        if retrieved_contexts and use_rag:
            context_text = " ".join(retrieved_contexts[:2]) # Use top 2 contexts
            prompt = f"Context: {context_text}\n\nPatient: {patient_input}\nDoctor:"
        else:
            prompt = f"Patient: {patient_input}\nDoctor:"

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            model_outputs = self.forward(input_ids, attention_mask)
            urgency_logits = model_outputs['urgency_logits']

            urgency_probs = F.softmax(urgency_logits, dim=-1)
            urgency_id = torch.argmax(urgency_probs, dim=-1).item()
            urgency_level = self.id_to_urgency[urgency_id]
            confidence_score = float(torch.max(urgency_probs).item())

        generated_ids = self.language_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_length,
        temperature=temperature,
        do_sample=True,
        pad_token_id=self.tokenizer.eos_token_id,
        num_return_sequences=1,
        repetition_penalty=1.1
        )

        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        if "Doctor:" in generated_text:
            response = generated_text.split("Doctor:")[-1].strip()
        else:
            response = generated_text[len(prompt):].strip()

        medical_category = self._infer_medical_category(patient_input)

        return ModelOutput(
        response=response,
        urgency_level=urgency_level,
        confidence_score=confidence_score,
        medical_category=medical_category,
        retrieved_contexts=retrieved_contexts
        )

    def _infer_medical_category(self, text: str) -> str:
        text_lower = text.lower()

        categories = {
        'cardiovascular': ['heart', 'blood pressure', 'chest pain', 'cardiac'],
        'respiratory': ['breathing', 'cough', 'lung', 'asthma'],
        'neurological': ['headache', 'brain', 'dizzy', 'nerve'],
        'gastrointestinal': ['stomach', 'nausea', 'digestive', 'bowel'],
        'musculoskeletal': ['back pain', 'joint', 'muscle', 'bone'],
        'dermatological': ['skin', 'rash', 'acne', 'eczema']
        }

        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category

        return 'general'

class MedicalAssistantTrainer:

    def __init__(self, model: MedicalAssistantModel, output_dir: str = "models/medical_assistant"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self,
        train_data: List[Dict],
        val_data: List[Dict],
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5):

        logger.info("Starting medical assistant model training...")

        train_dataset = MedicalConversationDataset(train_data, self.model.tokenizer)
        val_dataset = MedicalConversationDataset(val_data, self.model.tokenizer)

        training_args = TrainingArguments(
        output_dir=str(self.output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=str(self.output_dir / 'logs'),
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_accumulation_steps=2,
        dataloader_num_workers=2,
        )

        trainer = Trainer(
        model=self.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=self.model.tokenizer,
        )

        trainer.train()

        trainer.save_model(str(self.output_dir / "final"))
        self.model.tokenizer.save_pretrained(str(self.output_dir / "final"))

        logger.info(f"Training completed. Model saved to {self.output_dir}")

        return trainer

    def save_model(self, path: str):
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), save_path / "model.pt")

        self.model.tokenizer.save_pretrained(str(save_path))

        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: str):
        load_path = Path(path)

        self.model.load_state_dict(torch.load(load_path / "model.pt"))

        logger.info(f"Model loaded from {load_path}")

    def create_medical_assistant(model_name: str = "microsoft/DialoGPT-medium",
        knowledge_base_data: Optional[List[Dict]] = None) -> MedicalAssistantModel:

        knowledge_base = None
        if knowledge_base_data:
            logger.info("Initializing medical knowledge base...")
            knowledge_base = MedicalKnowledgeBase()
            knowledge_base.build_knowledge_base(knowledge_base_data)

        model = MedicalAssistantModel(
        model_name=model_name,
        knowledge_base=knowledge_base
        )

        logger.info(f"Medical assistant model initialized with {model_name}")
        return model
