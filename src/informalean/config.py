from pydantic import BaseModel
from pathlib import Path
import yaml

class DataConfig(BaseModel):
    minhash_shingle_length: int
    minhash_num_perm: int
    minhash_lsh_threshold: float
    max_tokenized_length: int

class TrainConfig(BaseModel):
    model_name: str
    # SFT config
    max_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    max_steps: int
    fp16: bool 
    gradient_checkpointing: bool
    logging_steps: int 
    eval_steps: int
    save_steps: int
    save_total_limit: int
    # LoRA config
    lora_r: int 
    lora_alpha: int 
    lora_dropout: float 
    lora_target_modules: list[str]
    # Quantization
    load_in_4bit: bool
    bnb_4bit_use_double_quant: bool
    # Logging
    example_eval_freq: int

class Config(BaseModel):
    data: DataConfig
    train: TrainConfig

def config_path(config_name: str): return Path(f"config/{config_name}.yaml")


def load_config(config_name) -> Config:
    return Config.model_validate(yaml.safe_load(config_path(config_name).read_text()))
