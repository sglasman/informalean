from pathlib import Path
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class TestInference:
    def __init__(self, model, tokenizer):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer

    @classmethod
    def from_path(cls, path: Path):
        return cls(
            AutoPeftModelForCausalLM.from_pretrained(path, device_map="cuda"),
            AutoTokenizer.from_pretrained(path),
        )

    @classmethod
    def from_model_name(cls, model_name):
        return cls(
            AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda"),
            AutoTokenizer.from_pretrained(model_name),
        )

    def infer(self, prompt):
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        return self.tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
