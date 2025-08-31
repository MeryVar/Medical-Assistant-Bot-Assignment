import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str = "microsoft/DialoGPT-medium" 
    max_length: int = 512
    max_new_tokens: int = 150
    temperature: float = 0.7
    do_sample: bool = True
    pad_token_id: int = 50256

    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4

@dataclass
class DataConfig:
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_context_length: int = 400
    min_response_length: int = 10
    max_response_length: int = 200

    data_dir: str = "data"
    processed_data_dir: str = "data/processed"
    raw_data_dir: str = "data/raw"

@dataclass
class TrainingConfig:
    device: str = "cuda" if os.getenv("CUDA_AVAILABLE") == "true" else "cpu"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    save_strategy: str = "epoch"
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100

    model_output_dir: str = "models"
    logs_dir: str = "logs"
    checkpoints_dir: str = "checkpoints"

@dataclass
class EvaluationConfig:
    metrics: Optional[list] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["bleu", "rouge", "bert_score", "perplexity"]
