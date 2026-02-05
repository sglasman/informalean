from functools import lru_cache
from typing import Optional
from peft import LoraConfig
from transformers import BitsAndBytesConfig, Trainer
from trl import SFTConfig
import torch
from informalean.files import statements_models_path

from informalean.config import Config, DataConfig, TrainConfig, load_config
from informalean.data.statements.process import load_processed_statements
from informalean.train.statements.informalean_trainer import InformaleanStatementTrainer
from informalean.train.statements.log_example_completions import LogExampleCompletions
import argparse


def model_init_kwargs(train_config: TrainConfig):
    result = {}
    if train_config.load_in_4bit:
        bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=train_config.bnb_4bit_use_double_quant,
        )
        if train_config.fp16:
            bnb_config.bnb_4bit_compute_dtype = torch.float16
        result["quantization_config"] = bnb_config
    return result


def sft_config(train_config: TrainConfig):
    return SFTConfig(
        eos_token="<|im_end|>",
        max_length=train_config.max_length,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        warmup_ratio=train_config.warmup_ratio,
        max_steps=train_config.max_steps,
        fp16=train_config.fp16,
        gradient_checkpointing=train_config.gradient_checkpointing,
        logging_steps=train_config.logging_steps,
        eval_steps=train_config.eval_steps,
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        report_to="wandb",
        model_init_kwargs=model_init_kwargs(train_config),
    )


def peft_config(train_config: TrainConfig):
    return LoraConfig(
        r=train_config.lora_r,
        task_type="CAUSAL_LM",
        target_modules=train_config.lora_target_modules,
        lora_alpha=train_config.lora_alpha,
        lora_dropout=train_config.lora_dropout,
        bias="none",
    )


@lru_cache(maxsize=1)
def get_dataset(data_config: DataConfig):
    return load_processed_statements(data_config=data_config)


def create_trainer(config: Config):
    train_config = config.train
    dataset = get_dataset(config.data)
    trainer = InformaleanStatementTrainer(
        model=config.train.model_name,
        args=sft_config(train_config),
        weights=dataset["train"]["weight"],
        peft_config=peft_config(train_config),
        train_dataset=dataset["train"],
    )
    trainer.add_callback(
        LogExampleCompletions(
            example_dataset=dataset["val"].select(_sample_val_indices),
            eval_freq=train_config.example_eval_freq,
            train_config=train_config,
        )
    )
    return trainer


def train(
    train_config: TrainConfig, model_subdir: str, trainer: Optional[Trainer] = None
):
    if trainer is None:
        trainer = create_trainer(train_config)
    trainer.train()
    trainer.save_model(statements_models_path / model_subdir)


_sample_val_indices = [41905, 7296, 1639, 48598, 18024, 16049, 14628, 9144, 48265, 6717]


def model_subdir() -> str:
    from datetime import datetime

    date_prefix = datetime.now().strftime("%m%d%y")
    existing = list(statements_models_path.glob(f"{date_prefix}_*"))
    run_numbers = [
        int(p.name.split("_")[-1]) for p in existing if p.name.split("_")[-1].isdigit()
    ]
    next_run = max(run_numbers, default=-1) + 1
    return f"{date_prefix}_{next_run}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="cloud")
    config = load_config(parser.parse_args().config_name)
    train(config, model_subdir())

if __name__ == "__main__":
    main()