import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.multi_task_model import MultiTaskModelResNet
from PIL import Image
import cv2
import base64
import mlflow.pytorch
import os

class PredictPipeline:
    def __init__(self, model_name="Project_1_MultiTaskModel"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None # Lazy loading
        self.class_label = {0: 'Glioma', 1: 'Meningioma', 2: 'No-tumor', 3: 'Pituitary'}

    def load_model(self):
        """
        Hàm này chỉ chạy 1 lần duy nhất khi khởi động App.
        """
        if self.model is None:
            print(f"--> Đang tải model {self.model_name}@champion từ DagsHub...")
            try:
                # CÁCH 1: Load trực tiếp bằng Alias (Nhanh, gọn, chuẩn MLOps mới)
                # Lưu ý: Cần setup biến môi trường MLFLOW_TRACKING_URI, USERNAME, PASSWORD trước khi chạy App
                model_uri = f"models:/{self.model_name}@champion"

                # Load model về thẳng RAM (MLflow tự handle việc cache local)
                self.model = mlflow.pytorch.load_model(model_uri)
                self.model.to(self.device)
                self.model.eval()
                print("--> Model loaded successfully!")

            except Exception as e:
                print(f"Error loading model: {e}")
                # Fallback: Nếu không nối mạng được, load model local (dự phòng)
                print("--> Loading local fallback model...")
                self.model = MultiTaskModelResNet(n_classes=4, n_segment=1, in_channels=3, pretrained=False)
                # Đường dẫn này phải có sẵn trong container/folder deploy
                self.model.load_state_dict(torch.load("artifacts/training/model.pth", map_location=self.device))
                self.model.to(self.device)
                self.model.eval()

        return self.model

    def preprocess_image(self, image_input):
        """
        Input: Có thể là đường dẫn file (str) hoặc đối tượng PIL Image (từ Streamlit uploader)
        """
        if isinstance(image_input, str):
            image = np.array(Image.open(image_input).convert('RGB'))
        else:
            # Trường hợp input là PIL Image (vd: từ st.file_uploader)
            image = np.array(image_input.convert('RGB'))

        transform = A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        augmented = transform(image=image)
        return augmented['image'].unsqueeze(0).to(self.device), image

    def predict(self, image_input):
        # 1. Đảm bảo model đã load (Singleton Pattern)
        model = self.load_model()

        # 2. Preprocess
        tensor_img, original_img = self.preprocess_image(image_input)

        # 3. Inference
        with torch.no_grad():
            class_output, seg_output = model(tensor_img)

            # Classification Result
            pred_index = torch.argmax(class_output, dim=1).item()
            class_name = self.class_label[pred_index]
            confidence = torch.softmax(class_output, dim=1)[0][pred_index].item()

            # Segmentation Result
            mask = torch.sigmoid(seg_output)
            mask = (mask > 0.5).float()

        # 4. Post-process (Visualization)
        # Resize mask về kích thước ảnh gốc (để overlay cho đẹp)
        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)

        # Resize ảnh gốc về 256x256 để khớp với mask (hoặc ngược lại)
        # Ở đây ta dùng ảnh đã resize trong preprocess để hiển thị cho đồng bộ
        display_img = cv2.resize(original_img, (256, 256))
        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR) # OpenCV dùng BGR

        # Tạo lớp phủ màu đỏ
        colored_mask = np.zeros_like(display_img)
        colored_mask[:, :, 2] = 255 # Red channel in BGR

        # Áp dụng mask
        colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_np)

        # Overlay
        alpha = 1.0
        beta = 0.5
        overlay = cv2.addWeighted(display_img, alpha, colored_mask, beta, 0)

        # Convert to Base64
        _, buffer = cv2.imencode('.png', overlay)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "image": img_base64,
            "class": class_name,
            "confidence": f"{confidence:.2%}"
        }