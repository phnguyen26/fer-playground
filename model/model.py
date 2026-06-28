import torch
import torch.nn as nn



class VGG8(nn.Module):
    def __init__(self, num_classes=7):
        super(VGG8, self).__init__()

        self.features = nn.Sequential(
            # Block 1: (1, 48, 48) -> (64, 24, 24)
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2:  (64, 24, 24) -> (128, 12, 12)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: (128, 12, 12) -> (256, 6, 6)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: (256, 6, 6) -> (512, 3, 3)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),


            # Block 5: (512, 3, 3) -> (512, 1, 1)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2) ,
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # FC 1
            nn.Linear(in_features=512 * 1 * 1, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            # FC 2
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            # FC 3 (Output Layer)
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_model(model_path, device):
    model = VGG8(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

