from peft import AutoPeftModelForCausalLM
from informalean.config import TrainConfig
from informalean.files import statements_models_path
import informalean.common.dependencies as dependencies
import torch


class TestInference:

    def __init__(self, model_subdir: str, train_config: TrainConfig):
        self.train_config = train_config
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            statements_models_path / model_subdir, device_map="cuda"
        )
        self.model.eval()
        self.tokenizer = dependencies.tokenizer(train_config.model_name)

    def infer(self, prompt):
        inputs = self.tokenizer.apply_chat_template(prompt, tokenize=True, return_dict=True, add_generation_prompt=True, return_tensors="pt")
        inputs.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return self.tokenizer.decode(out[0])