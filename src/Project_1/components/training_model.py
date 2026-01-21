import os
import torch
import torchvision.models as models
from models.multi_task_model import MultiTaskModelResNet
from data.loader_data import data_loader
from loss_func.combined_loss import UncertainlyLoss
from metrics import calculate_dice, calculate_iou, EarlyStopping
from Project_1.entity.config_entity import TrainingModelConfig
from pathlib import Path
import dagshub
import mlflow
import mlflow.pytorch
from urllib.parse import urlparse

class TrainingModel:
    def __init__(self, config: TrainingModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def loader_data(self):

            train_class_path = os.path.join(self.config.data_classification, 'train')
            train_seg_path = os.path.join(self.config.data_segmentation, 'train')

            train_loader = data_loader(
                data_classification_path=train_class_path,
                data_segmentation_path=train_seg_path,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                augmentation=self.config.augmentation,
                seed=self.config.seed,
                img_size=self.config.image_size
            )

            return train_loader

    def train_model(self):
        if "MLFLOW_TRACKING_URI" not in os.environ:
            print("Running locally, initializing DagsHub...")
            dagshub.init(repo_owner=self.config.repo_owner, repo_name=self.config.repo_name)
        else:
            print("Running in CI/CD, using existing environment variables.")
        exp_name = "Project_1_Tracking_Model"
        print(f"Setting MLflow experiment to: {exp_name}")
        mlflow.set_experiment(exp_name)

        with mlflow.start_run():
            model = MultiTaskModelResNet(n_classes=self.config.n_classes,
                                        n_segment=self.config.n_segment,
                                        in_channels=self.config.in_channels,
                                        pretrained=True)
            train_loader = self.loader_data()
            model = model.to(self.device)
            criterion = UncertainlyLoss(task_num=self.config.task_num)
            optimizer = torch.optim.Adam(model.parameters(), lr= self.config.lr)

            for epoch in range(self.config.epochs):
                model.train()
                running_loss = 0.0
                running_dice = 0.0
                running_iou = 0.0
                correct = 0
                for images, masks, labels in train_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs_class, outputs_seg = model(images)
                    _, pred = torch.max(outputs_class, 1)

                    loss = criterion(outputs_seg, masks, outputs_class, labels)
                    loss.backward()
                    optimizer.step()

                    correct += (pred == labels).sum().item()

                    running_loss += loss.item()
                    dice = calculate_dice(outputs_seg, masks)
                    iou = calculate_iou(outputs_seg, masks)
                    running_dice += dice.item()
                    running_iou += iou.item()

                epoch_loss = running_loss / len(train_loader)
                epoch_dice = running_dice / len(train_loader)
                epoch_iou = running_iou / len(train_loader)
                accuracy = correct / (len(train_loader.dataset))
                print(f"Epoch {epoch+1}/{self.config.epochs} - Loss: {epoch_loss:.4f} - Acc: {accuracy:.4f}")

                mlflow.log_metric("Loss", epoch_loss, step=epoch)
                mlflow.log_metric("Dice", epoch_dice, step=epoch)
                mlflow.log_metric("IoU", epoch_iou, step=epoch)
                mlflow.log_metric("Accuracy", accuracy, step=epoch)

            print(f'Training model completed')
            os.makedirs("artifacts/training", exist_ok=True)
            torch.save(model.state_dict(), "artifacts/training/model.pth")

            print("Đã lưu local model checkpoint cho DVC!")

            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",          # Tên thư mục chứa model trong Run
                registered_model_name=self.config.model_name, # Tên Model trong Registry (Quan trọng)
                pip_requirements="requirements.txt" # (Tùy chọn) Để tiện deploy sau này
            )

            print("Model registered to MLflow Registry successfully!")
