from pydantic import BaseModel
from pathlib import Path
import yaml

class DataConfig(BaseModel):
    minhash_shingle_length: int
    minhash_num_perm: int
    minhash_lsh_threshold: float


class Config(BaseModel):
    data: DataConfig


config_path = Path("config/config.yaml")


def load_config() -> Config:
    return Config.model_validate(yaml.safe_load(config_path.read_text()))
