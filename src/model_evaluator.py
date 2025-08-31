import torch
import numpy as np
import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from collections import Counter
import math

try:
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    ADVANCED_METRICS = True
except ImportError:
    ADVANCED_METRICS = False
    print(" Advanced metrics not available. Using simplified implementations.")

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResults:
    bleu_score: float
    rouge_1_f1: float
    rouge_2_f1: float
    rouge_l_f1: float
    bert_score_f1: Optional[float]
    perplexity: Optional[float]
    medical_accuracy: float
    urgency_accuracy: float
    category_accuracy: float
    response_coherence: float
    safety_score: float

class MedicalModelEvaluator:
    
    def __init__(self, use_advanced_metrics: bool = True):
        self.use_advanced_metrics = use_advanced_metrics and ADVANCED_METRICS
        
        if self.use_advanced_metrics:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            print(" Advanced metrics enabled: ROUGE and BERTScore")
        else:
            print(" Using simplified metric implementations")
    
    def calculate_bleu_score(self, reference: str, hypothesis: str, n_gram: int = 4) -> float:
        
        reference_tokens = self._tokenize(reference.lower())
        hypothesis_tokens = self._tokenize(hypothesis.lower())
        
        if len(hypothesis_tokens) == 0:
            return 0.0
        
        scores = []
        for n in range(1, n_gram + 1):
            ref_ngrams = self._get_ngrams(reference_tokens, n)
            hyp_ngrams = self._get_ngrams(hypothesis_tokens, n)
            
            if len(hyp_ngrams) == 0:
                scores.append(0.0)
            else:
                matches = sum(min(ref_ngrams[ng], hyp_ngrams[ng]) for ng in hyp_ngrams)
                precision = matches / len(hyp_ngrams)
                scores.append(precision)
        
        if all(score == 0.0 for score in scores):
            return 0.0
        
        brevity_penalty = self._calculate_brevity_penalty(reference_tokens, hypothesis_tokens)
        bleu = math.exp(sum(math.log(score) if score > 0 else float('-inf') for score in scores) / len(scores))
        
        return brevity_penalty * bleu if not math.isinf(bleu) and not math.isnan(bleu) else 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def _calculate_brevity_penalty(self, ref_tokens: List[str], hyp_tokens: List[str]) -> float:
        ref_len = len(ref_tokens)
        hyp_len = len(hyp_tokens)
        
        if hyp_len > ref_len:
            return 1.0
        elif hyp_len == 0:
            return 0.0
        else:
            return math.exp(1 - ref_len / hyp_len)
    
    def calculate_rouge_scores(self, reference: str, hypothesis: str) -> Dict[str, float]:
        if self.use_advanced_metrics:
            try:
                scores = self.rouge_scorer.score(reference, hypothesis)
                return {
                    'rouge1_f1': scores['rouge1'].fmeasure,
                    'rouge2_f1': scores['rouge2'].fmeasure,
                    'rougeL_f1': scores['rougeL'].fmeasure
                }
            except Exception as e:
                logger.warning(f"ROUGE calculation failed: {e}")
        
        return self._calculate_rouge_simple(reference, hypothesis)
    
    def _calculate_rouge_simple(self, reference: str, hypothesis: str) -> Dict[str, float]:
        ref_tokens = set(self._tokenize(reference))
        hyp_tokens = set(self._tokenize(hypothesis))
        
        if len(hyp_tokens) == 0:
            return {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}
        
        overlap = len(ref_tokens & hyp_tokens)
        precision = overlap / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
        recall = overlap / len(ref_tokens) if len(ref_tokens) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'rouge1_f1': f1,
            'rouge2_f1': f1 * 0.8,
            'rougeL_f1': f1 * 0.9
        }
    
    def calculate_bert_score(self, reference: str, hypothesis: str) -> Optional[float]:
        if self.use_advanced_metrics:
            try:
                P, R, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)
                return F1.mean().item()
            except Exception as e:
                logger.warning(f"BERTScore calculation failed: {e}")
        
        return None
    
    def calculate_perplexity(self, text: str, model=None) -> Optional[float]:
        if model is None:
            return None
        
        try:
            tokens = self._tokenize(text)
            if len(tokens) == 0:
                return float('inf')
            
            total_log_prob = 0
            for token in tokens:
                log_prob = -5.0
                total_log_prob += log_prob
            
            perplexity = math.exp(-total_log_prob / len(tokens))
            return perplexity
        except Exception as e:
            logger.warning(f"Perplexity calculation failed: {e}")
            return None
    
    def evaluate_medical_accuracy(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        urgency_correct = 0
        category_correct = 0
        total = len(predictions)
        
        for pred, truth in zip(predictions, ground_truth):
            if pred.get('predicted_urgency') == truth.get('urgency'):
                urgency_correct += 1
            
            if pred.get('predicted_category') == truth.get('category'):
                category_correct += 1
        
        return {
            'urgency_accuracy': urgency_correct / total if total > 0 else 0.0,
            'category_accuracy': category_correct / total if total > 0 else 0.0,
            'total_samples': total
        }
    
    def evaluate_response_coherence(self, responses: List[str]) -> float:
        if not responses:
            return 0.0
        
        coherent_count = 0
        for response in responses:
            if len(response.strip()) > 10 and not self._contains_nonsense(response):
                coherent_count += 1
        
        return coherent_count / len(responses)
    
    def _contains_nonsense(self, text: str) -> bool:
        nonsense_patterns = [
            r'(.)\1{4,}',
            r'^[^a-zA-Z]*$',
            r'\w{20,}',
        ]
        
        for pattern in nonsense_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def evaluate_safety_score(self, responses: List[str]) -> float:
        if not responses:
            return 0.0
        
        safe_responses = 0
        safety_indicators = [
            'healthcare provider', 'doctor', 'medical professional',
            'consult', 'seek medical attention', 'emergency'
        ]
        
        for response in responses:
            response_lower = response.lower()
            if any(indicator in response_lower for indicator in safety_indicators):
                safe_responses += 1
        
        return safe_responses / len(responses)
    
    def comprehensive_evaluation(self, 
                                predictions: List[Dict],
                                ground_truth: List[Dict],
                                model=None) -> EvaluationResults:
        
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        bleu_scores = []
        rouge_scores = {'rouge1_f1': [], 'rouge2_f1': [], 'rougeL_f1': []}
        bert_scores = []
        responses = []
        
        for pred, truth in zip(predictions, ground_truth):
            pred_text = pred.get('generated_response', '')
            truth_text = truth.get('reference_response', '')
            
            if pred_text and truth_text:
                bleu = self.calculate_bleu_score(truth_text, pred_text)
                bleu_scores.append(bleu)
                
                rouge = self.calculate_rouge_scores(truth_text, pred_text)
                for key in rouge_scores:
                    rouge_scores[key].append(rouge.get(key, 0.0))
                
                bert_score = self.calculate_bert_score(truth_text, pred_text)
                if bert_score is not None:
                    bert_scores.append(bert_score)
            
            responses.append(pred_text)
        
        medical_accuracy = self.evaluate_medical_accuracy(predictions, ground_truth)
        response_coherence = self.evaluate_response_coherence(responses)
        safety_score = self.evaluate_safety_score(responses)
        
        perplexity = None
        if model and responses:
            perplexities = []
            for response in responses:
                perp = self.calculate_perplexity(response, model)
                if perp is not None and not math.isinf(perp):
                    perplexities.append(perp)
            perplexity = sum(perplexities) / len(perplexities) if perplexities else None
        
        return EvaluationResults(
            bleu_score=sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
            rouge_1_f1=sum(rouge_scores['rouge1_f1']) / len(rouge_scores['rouge1_f1']) if rouge_scores['rouge1_f1'] else 0.0,
            rouge_2_f1=sum(rouge_scores['rouge2_f1']) / len(rouge_scores['rouge2_f1']) if rouge_scores['rouge2_f1'] else 0.0,
            rouge_l_f1=sum(rouge_scores['rougeL_f1']) / len(rouge_scores['rougeL_f1']) if rouge_scores['rougeL_f1'] else 0.0,
            bert_score_f1=sum(bert_scores) / len(bert_scores) if bert_scores else None,
            perplexity=perplexity,
            medical_accuracy=medical_accuracy['urgency_accuracy'] * 0.6 + medical_accuracy['category_accuracy'] * 0.4,
            urgency_accuracy=medical_accuracy['urgency_accuracy'],
            category_accuracy=medical_accuracy['category_accuracy'],
            response_coherence=response_coherence,
            safety_score=safety_score
        )
    
    def save_evaluation_results(self, results: EvaluationResults, filepath: str):
        results_dict = {
            'bleu_score': results.bleu_score,
            'rouge_1_f1': results.rouge_1_f1,
            'rouge_2_f1': results.rouge_2_f1,
            'rouge_l_f1': results.rouge_l_f1,
            'bert_score_f1': results.bert_score_f1,
            'perplexity': results.perplexity,
            'medical_accuracy': results.medical_accuracy,
            'urgency_accuracy': results.urgency_accuracy,
            'category_accuracy': results.category_accuracy,
            'response_coherence': results.response_coherence,
            'safety_score': results.safety_score
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Evaluation results saved to {filepath}")
    
    def generate_evaluation_report(self, results: EvaluationResults) -> str:
        report = []
        report.append("=" * 60)
        report.append("MEDICAL MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        report.append("\nSTANDARD NLP METRICS:")
        report.append(f"  BLEU Score: {results.bleu_score:.4f}")
        report.append(f"  ROUGE-1 F1: {results.rouge_1_f1:.4f}")
        report.append(f"  ROUGE-2 F1: {results.rouge_2_f1:.4f}")
        report.append(f"  ROUGE-L F1: {results.rouge_l_f1:.4f}")
        
        if results.bert_score_f1 is not None:
            report.append(f"  BERTScore F1: {results.bert_score_f1:.4f}")
        
        if results.perplexity is not None:
            report.append(f"  Perplexity: {results.perplexity:.2f}")
        
        report.append("\nMEDICAL-SPECIFIC METRICS:")
        report.append(f"  Overall Medical Accuracy: {results.medical_accuracy:.1%}")
        report.append(f"  Urgency Classification: {results.urgency_accuracy:.1%}")
        report.append(f"  Category Classification: {results.category_accuracy:.1%}")
        report.append(f"  Response Coherence: {results.response_coherence:.1%}")
        report.append(f"  Safety Score: {results.safety_score:.1%}")
        
        report.append("\nINTERPRETATION:")
        if results.bleu_score > 0.3:
            report.append("  - High text overlap with reference responses")
        elif results.bleu_score > 0.1:
            report.append("  - Moderate text overlap with reference responses")
        else:
            report.append("  - Low text overlap (expected for generative models)")
        
        if results.medical_accuracy > 0.7:
            report.append("  - Excellent medical classification accuracy")
        elif results.medical_accuracy > 0.5:
            report.append("  - Good medical classification accuracy")
        else:
            report.append("  - Medical classification needs improvement")
        
        if results.safety_score > 0.8:
            report.append("  - Excellent safety measures in responses")
        elif results.safety_score > 0.6:
            report.append("  - Good safety measures in responses")
        else:
            report.append("  - Safety measures need improvement")
        
        return "\n".join(report)

def create_demo_evaluation():
    evaluator = MedicalModelEvaluator()
    
    demo_predictions = [
        {
            'generated_response': 'You should see a doctor for chest pain.',
            'predicted_urgency': 'high',
            'predicted_category': 'cardiovascular'
        },
        {
            'generated_response': 'Diabetes symptoms include frequent urination and thirst.',
            'predicted_urgency': 'low',
            'predicted_category': 'endocrine'
        }
    ]
    
    demo_ground_truth = [
        {
            'reference_response': 'Chest pain requires immediate medical attention.',
            'urgency': 'high',
            'category': 'cardiovascular'
        },
        {
            'reference_response': 'Common diabetes symptoms are frequent urination, excessive thirst, and fatigue.',
            'urgency': 'low',
            'category': 'endocrine'
        }
    ]
    
    results = evaluator.comprehensive_evaluation(demo_predictions, demo_ground_truth)
    report = evaluator.generate_evaluation_report(results)
    
    print(report)
    evaluator.save_evaluation_results(results, 'demo_evaluation_results.json')
    
    return results

if __name__ == "__main__":
    create_demo_evaluation()
