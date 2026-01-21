from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class TrainingModelConfig:
    data_classification: Path
    data_segmentation: Path
    repo_name: str
    repo_owner: str
    n_classes: int
    n_segment: int
    img_size: int
    batch_size: int
    lr: float
    seed: int
    in_channels: int
    epochs: int
    num_worker: int
    task_num: int
    augmentation: bool
    patience: int
    model_name: str

@dataclass(frozen=True)
class EvaluationModelConfig:
    model_path: Path
    root_dir: Path
    repo_name: str
    repo_owner: str
    report_path: Path
    batch_size: int
    n_classes: int
    n_segment: int
    in_channels: int
    num_workers: int
    img_size: int
    seed: int
    augmentation: bool
    all_params: dict
    model_name: str