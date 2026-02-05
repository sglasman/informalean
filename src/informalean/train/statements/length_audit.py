from datasets import concatenate_datasets
from tqdm import tqdm
def get_lengths(data, tokenizer):
    return [
        len(
            tokenizer.apply_chat_template(
                r["prompt"] + r["completion"],
                tokenize=True,
                add_generation_prompt=True
            )
        )
        for r in tqdm(concatenate_datasets(list(data.values())), desc="Getting lengths of data")
    ]
