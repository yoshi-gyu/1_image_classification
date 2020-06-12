import torch
from torchvision import models
import torch.nn.functional as F

class Model(torch.nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.num_classes = num_classes

        # VGG-19
        self.vgg = models.vgg19(pretrained=True)

        # VGG19の最後の出力層の出力ユニットをVOCの20クラスに付け替える
        self.vgg.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = self.vgg(x)
        # return torch.sigmoid(x)
        # return x
        return F.softmax(x, dim=1)