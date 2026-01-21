import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # 1. Chuyển logits thành xác suất
        predictions = torch.sigmoid(predictions)

        # 2. Flatten: (Batch_Size, Channel, H, W) -> (Batch_Size, -1)
        # Giữ lại chiều Batch_Size để tính riêng từng ảnh
        predictions = predictions.view(predictions.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # 3. Tính Intersection và Union cho TỪNG ảnh (dim=1)
        intersection = (predictions * targets).sum(dim=1)
        union = predictions.sum(dim=1) + targets.sum(dim=1)

        # 4. Tính Dice cho từng ảnh
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # 5. Trả về 1 - dice trung bình của cả batch
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        # BCEWithLogitsLoss đã tích hợp Sigmoid bên trong, rất ổn định
        self.bce = nn.BCEWithLogitsLoss() 
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce

    def forward(self, predictions, targets):
        # predictions: Logits (chưa qua sigmoid)
        # targets: Binary mask (0 hoặc 1)
        
        dice_loss = self.dice(predictions, targets)
        bce_loss = self.bce(predictions, targets)
        
        return self.weight_dice * dice_loss + self.weight_bce * bce_loss
    
class UncertainlyLoss(nn.Module):
  def __init__(self, task_num):
    super(UncertainlyLoss, self).__init__()
    self.task_num = task_num
    # khởi tạo log(sigma^2) = 0
    self.log_vars = nn.Parameter(torch.zeros((task_num)))
    self.loss_seg = CombinedLoss()
    self.loss_classify = nn.CrossEntropyLoss()

  def forward(self, seg_logits, seg_labels, classify_logits, classify_labels):
    loss_seg = self.loss_seg(seg_logits, seg_labels)
    loss_classify = self.loss_classify(classify_logits, classify_labels)

    # log(sigma^2) -> sigma^2
    precision_sent = torch.exp(self.log_vars[0])
    precision_classify = torch.exp(self.log_vars[1])

    loss = torch.log(precision_sent**0.5) + torch.log(precision_classify**0.5) + loss_sent * (1 / (precision_sent*2)) + loss_classify * (1 / (precision_classify*2))

    return loss

# class StaticLoss(nn.Module):
#   def __init__(self, lambda_classify: float, lambda_segment: float):
#     super(StaticLoss, self).__init__()
#     self.lambda_classify = lambda_classify
#     self.lambda_segment = lambda_segment
#     self.loss_sent = CombinedLoss()
#     self.loss_classify = nn.CrossEntropyLoss()

#   def forward(self, sent_logits, sent_labels, classify_logits, classify_labels):
#     loss_sent = self.loss_sent(sent_logits, sent_labels)
#     loss_classify = self.loss_classify(classify_logits, classify_labels)
    
#     loss = loss_sent * self.lambda_segment + loss_classify * self.lambda_classify
    
#     return loss
  
# def MultitaskLoss(lambda_classify : float|None , lambda_segment: float|None, task_num : int|None, name_loss ='uncertainly'):
#     if name_loss == "uncertainly":
#       return UncertainlyLoss(task_num=task_num)
#     elif name_loss == 'static':
#       return StaticLoss(lambda_classify=lambda_classify, lambda_segment=lambda_segment)