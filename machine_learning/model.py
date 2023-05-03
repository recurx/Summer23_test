from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import shufflenet_v2_x0_5, mobilenet_v3_small, MobileNet_V3_Small_Weights

import image


# Your models

class RGBDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.input_dir = 'input'
        self.label_dir = 'label'
        self.index_to_file = lambda i: 'num_' + str(4 + i // 10000) + '.txt'

    def __len__(self) -> int:
        return 120000

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = {
            'input': np.array(np.moveaxis(image.get_img(idx, 'input'), -1, 0), dtype=np.float32),
            'target': np.array(image.get_pos_angles(idx, 'label'), dtype=np.float32),
        }
        data_torch = dict()
        for key, value in data.items():
            data_torch[key] = torch.from_numpy(value)
        return data_torch


class Model(nn.Module):
    def __init__(self, pretrained=False, out_channels=15*3, **kwargs):
        super().__init__()
        # load backbone model
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=pretrained)
        # replace the last linear layer to change output dimention to 45
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        ic = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(
            in_features=ic, out_features=out_channels)
        self.model = model
        # normalize RGB input to zero mean and unit variance
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(x))

    def predict(self, x):
        return self.forward(x)


if __name__ == "__main__":
    model = Model()
    print(model.eval())