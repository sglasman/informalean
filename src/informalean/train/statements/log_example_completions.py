from tokenizers import Tokenizer
from transformers import TrainerCallback
import wandb
from datasets import Dataset
import torch
import logging
from informalean.common import dependencies
from informalean.config import TrainConfig

logger = logging.getLogger(__name__)


class LogExampleCompletions(TrainerCallback):
    def __init__(
        self, example_dataset: Dataset, eval_freq: int, train_config: TrainConfig
    ):
        self.example_dataset = example_dataset
        self.eval_freq = eval_freq
        self.tokenizer: Tokenizer = dependencies.tokenizer(train_config.model_name)
        self.train_config = train_config

    def on_step_end(self, args, state, control, **kwargs):
        logger.info(f"In callback. Step: {state.global_step}")
        if not (state.global_step == 1 or state.global_step % self.eval_freq == 0):
            return

        model = kwargs["model"]
        model.eval()

        def _process_example(example):
            logger.info(f"Generating example for prompt {example['prompt']}")
            with torch.no_grad():
                tokenized = self.tokenizer.apply_chat_template(
                    example["prompt"], add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)
                generated = model.generate(
                    tokenized, max_new_tokens=256, do_sample=False
                )
                input_len = tokenized.shape[-1]
                decoded = self.tokenizer.decode(
                    generated[0][input_len:], skip_special_tokens=True
                )
                logger.info(f"Decoded prediction: {decoded}")
                return [example["prompt"], decoded, example["completion"]]

        wandb.log(
            {
                "eval/completions": wandb.Table(
                    columns=["prompt", "prediction", "reference"],
                    data=[
                        _process_example(example) for example in self.example_dataset
                    ],
                )
            },
        )
