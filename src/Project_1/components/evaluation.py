import torch
import os
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from Project_1.entity.config_entity import EvaluationModelConfig
from Project_1.utils.common import save_json
from models.multi_task_model import MultiTaskModelResNet
from data.loader_data import data_loader
from metrics import calculate_dice, calculate_iou

class EvaluationModel:
    def __init__(self, config: EvaluationModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # FIX 1: Khởi tạo Client ngay tại __init__ để dùng xuyên suốt
        self.client = MlflowClient()

    def loader_data(self):
        test_class_path = os.path.join(self.config.data_classification, 'test')
        test_seg_path = os.path.join(self.config.data_segmentation, 'test')

        test_loader = data_loader(
            data_classification_path=test_class_path,
            data_segmentation_path=test_seg_path,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            augmentation=self.config.augmentation,
            seed=self.config.seed,
            img_size=self.config.img_size
        )
        return test_loader

    def load_local_model(self):
        print(f"--> Đang load model local từ: {self.config.model_path}")

        # Khởi tạo kiến trúc model (cần params khớp với training)
        model = MultiTaskModelResNet(
            n_classes=self.config.n_classes,
            n_segment=self.config.n_segment,
            in_channels=self.config.in_channels,
            pretrained=False
        )

        # Load weights đã train
        model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        return model

    def get_latest_model_version_number(self, model_name):
        """
        Hàm này chỉ dùng để lấy SỐ VERSION (metadata) để promote,
        chứ không dùng để load weights.
        """
        results = self.client.search_model_versions(f"name='{model_name}'")
        if not results:
            print(f"Warning: Chưa tìm thấy model {model_name} trên Registry.")
            return None

        latest_version = max([int(v.version) for v in results])
        return str(latest_version)

    def get_champion_metrics(self, model_name):
        try:
            champion_version = self.client.get_model_version_by_alias(model_name, "champion")
            run_id = champion_version.run_id
            run = mlflow.get_run(run_id)
            print(f"Comparing with Champion Version: {champion_version.version}")
            return run.data.metrics
        except Exception:
            print("--> Chưa có Champion nào. Model hiện tại sẽ là Champion đầu tiên.")
            return None

    def promote_to_champion(self, model_name, version, current_metrics, old_metrics):
        if version is None:
            print("Không tìm thấy version trên Registry để promote.")
            return

        promote = False
        if old_metrics is None:
            promote = True
        else:
            new_acc = current_metrics.get('accuracy', 0)
            old_acc = old_metrics.get('accuracy', 0)
            new_dice = current_metrics.get('dice', 0)
            old_dice = old_metrics.get('dice', 0)

            print(f"Battle: New Acc ({new_acc:.4f}) vs Old Acc ({old_acc:.4f})")

            if new_acc > old_acc:
                promote = True
            elif new_acc == old_acc and new_dice > old_dice:
                promote = True
                print("Accuracy bằng nhau, nhưng Dice cao hơn -> Promote!")

        if promote:
            print(f"--> CHÚC MỪNG! Version {version} đang được thăng cấp lên @champion")
            self.client.set_registered_model_alias(model_name, "champion", version)
        else:
            print(f"--> Rất tiếc. Version {version} không vượt qua được Champion hiện tại.")

    def evaluation(self):
        # 1. Load Local Model (DVC Compliance)
        model = self.load_local_model()
        model.to(self.device)

        val_loader = self.loader_data()

        running_dice = 0.0
        running_iou = 0.0
        correct = 0

        with torch.no_grad():
            model.eval()
            for images, masks, labels in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                outputs_class, outputs_seg = model(images)
                _, pred = torch.max(outputs_class, 1)

                dice = calculate_dice(outputs_seg, masks)
                iou = calculate_iou(outputs_seg, masks)

                running_dice += dice.item()
                running_iou += iou.item()
                correct += (pred == labels).sum().item()

            # Tính trung bình
            epoch_dice = running_dice / len(val_loader)
            epoch_iou = running_iou / len(val_loader)
            accuracy = correct / (len(val_loader.dataset))

            current_metrics = {
                "dice": epoch_dice,
                "iou": epoch_iou,
                "accuracy": accuracy
            }
            print(f"Evaluation Result: {current_metrics}")

            # FIX 3: Lưu metrics.json local cho DVC track sự thay đổi
            save_json(Path(self.config.report_path), current_metrics)
            print(f"Metrics saved to {self.config.report_path} for DVC tracking.")

            # --- PHẦN GIAO TIẾP VỚI MLFLOW REGISTRY ---
            # 2. Lấy thông tin Champion từ Server
            champion_metrics = self.get_champion_metrics(self.config.model_name)

            # 3. Lấy số version vừa train xong (để nếu thắng thì biết ai mà phong chức)
            latest_version = self.get_latest_model_version_number(self.config.model_name)

            # 4. Tổ chức cuộc đấu (Battle) và Promote
            self.promote_to_champion(
                self.config.model_name,
                latest_version,
                current_metrics,
                champion_metrics
            )