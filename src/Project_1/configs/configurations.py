import os
from Project_1.constants import *
from Project_1.utils.common import read_yaml, create_directories
from Project_1.entity.config_entity import TrainingModelConfig, EvaluationModelConfig
from pathlib import Path

class ConfigureManager:
    def __init__(self,
                config_filepath: Path = CONFIG_FILE_PATH,
                params_filepath: Path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

        self.repo_name = os.getenv('MLFLOW_TRACKING_REPO')
        self.repo_owner = os.getenv('MLFLOW_TRACKING_USERNAME')

    def get_training_model_config(self) -> TrainingModelConfig:
        training = self.config.training_model
        params = self.params

        training_model_config = TrainingModelConfig(
            data_classification=Path(training.data_classification),
            data_segmentation=Path(training.data_segmentation),
            repo_name=self.repo_name,
            repo_owner=self.repo_owner,
            n_classes = params.N_CLASSES,
            n_segment = params.N_SEGMENT,
            img_size = params.IMAGE_SIZE,
            batch_size = params.BATCH_SIZE,
            lr = params.LEARNING_RATE,
            seed = params.SEED,
            in_channels = params.IN_CHANNELS,
            epochs = params.EPOCHS,
            num_worker = params.NUM_WORKERS,
            task_num = params.TASK_NUM,
            augmentation = params.AUGMENTATION,
            patience= params.PATIENCE,
            model_name= params.MODEL_NAME
        )
        return training_model_config

    def get_evaluation_model_config(self) -> EvaluationModelConfig:
        evaluation = self.config.evaluation_model
        params = self.params
        create_directories([evaluation.root_dir])

        evaluation_model_config = EvaluationModelConfig(
            model_path=Path(evaluation.model_path),
            root_dir=Path(evaluation.root_dir),
            repo_name=self.repo_name,
            repo_owner=self.repo_owner,
            report_path=Path(evaluation.report_path),
            n_classes = params.N_CLASSES,
            n_segment = params.N_SEGMENT,
            img_size = params.IMAGE_SIZE,
            batch_size = params.BATCH_SIZE,
            seed = params.SEED,
            num_worker = params.NUM_WORKERS,
            in_channels = params.IN_CHANNELS,
            augmentation = params.AUGMENTATION,
            all_params= params,
            model_name= params.MODEL_NAME
        )
        return evaluation_model_config