from Project_1 import logger
from Project_1.pipeline.stage2_evaluation import EvaluationModelPipeline
from Project_1.pipeline.stage1_training_model import TrainingModelPipeline

logger.info("Starting Training Model Pipeline...")

STAGE_NAME = "Training Model Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    obj = TrainingModelPipeline()
    obj.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluation Model Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    obj = EvaluationModelPipeline()
    obj.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e