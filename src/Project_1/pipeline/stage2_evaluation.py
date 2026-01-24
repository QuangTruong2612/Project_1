from Project_1.configs.configurations import ConfigureManager
from Project_1.components.evaluation import EvaluationModel
from Project_1 import logger

STAGE_NAME = "Evaluation Model Stage"
class EvaluationModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigureManager()
        evaluation_model = config.get_evaluation_model_config()
        evaluation_model = EvaluationModel(config=evaluation_model)
        evaluation_model.evaluation()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationModelPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e