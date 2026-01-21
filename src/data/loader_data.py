import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from PIL import Image
import os
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
class Data_Seg_Class(Dataset):
    def __init__(self, class_path, seg_path,  transform=None):
        
        self.classifi_path = class_path
        self.img_seg_path = seg_path + '/images'
        self.mask_seg_path = seg_path + '/masks'
        self.transform = transform
        self.class_dataset = datasets.ImageFolder(self.classifi_path)

        name_labels = self.class_dataset.classes

        # lấy tất các file image trong folder classification
        self.images_filename = []
        for i in name_labels:
            for f in os.listdir(self.classifi_path + f'/{i}') :
                if os.path.isfile(os.path.join(self.classifi_path + f'/{i}', f)):
                    self.images_filename.append(os.path.splitext(f)[0])

        # Load dataset phân loại 1 lần duy nhất
        self.label_map = {os.path.splitext(os.path.basename(p))[0]: label for p, label in self.class_dataset.samples}

    def __len__(self):
        return len(self.images_filename)

    def __getitem__(self, idx):
        filename = self.images_filename[idx]

        image_path = os.path.join(self.img_seg_path, filename + '.jpg')
        # kiểm tra file ảnh có trong folder segmentation không
        if os.path.isfile(image_path):
            
            mask_path = os.path.join(self.mask_seg_path, filename + '.png')
            
            # Đọc ảnh và mask
            image = np.array(Image.open(image_path).convert('RGB'))
            mask = np.array(Image.open(mask_path).convert('L'))
            mask = (mask > 127).astype(np.float32)
        else:
            image_path_new = os.path.join(self.classifi_path + '/no_tumor', filename + '.jpg')
            image = np.array(Image.open(image_path_new).convert('RGB'))
            mask = np.zeros((512, 512), dtype=np.float32)


        # Augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = torch.unsqueeze(mask, 0)
        # Tìm label tương ứng
        label = self.label_map.get(filename, -1)  # nếu không tìm thấy thì = -1

        return image, mask, label
    
def data_loader(data_classification_path, data_segmentation_path, batch_size, shuffle, num_workers, augmentation, seed, img_size):
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if augmentation:
        transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.GaussNoise(var_limit=(0.01, 0.05), p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], 
            is_check_shapes=False )
    
    else:
        transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], is_check_shapes=False)

    dataset = Data_Seg_Class(data_classification_path, data_segmentation_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader