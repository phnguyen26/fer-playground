import torch
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from experiments.utils import get_transform, export_confusion_matrix, export_classification_report
from model.model import load_model

ROOT = Path(__file__).parents[1]

test_datas = ImageFolder(root=ROOT / "data" / "FACE-EMOTION-KAGGLE", transform=get_transform())
test_loader = DataLoader(test_datas, batch_size=256, shuffle=False)
model = load_model(str(ROOT / "weights.pth"), device=torch.device('cpu'))
y_true, y_pred = [], []
with torch.no_grad():
    for (inputs, labels) in test_loader:
        
        predictions = model(inputs)
        preds = torch.argmax(predictions, 1)
        y_pred.extend(preds.numpy())
        y_true.extend(labels.numpy())


export_confusion_matrix(y_true, y_pred, class_names=test_datas.classes, dataset="kaggle")
export_classification_report(y_true, y_pred, class_names=test_datas.classes, dataset="kaggle")