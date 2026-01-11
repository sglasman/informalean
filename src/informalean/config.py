from pydantic import BaseModel
from pathlib import Path
import yaml

class DataConfig(BaseModel):
    faiss_statement_nlist: int
    faiss_statement_n_train: int
    faiss_statement_nprobe: int
    n_nearest_neighbors: int
    group_distance_threshold: float


class Config(BaseModel):
    data: DataConfig


config_path = Path("config/config.yaml")


def load_config() -> Config:
    return Config.model_validate(yaml.safe_load(config_path.read_text()))
