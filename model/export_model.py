import torch
from pathlib import Path
from model.model import load_model

ROOT = Path(__file__).parents[1]
model = load_model(str(ROOT / "api" / "weights.pth"), device=torch.device('cpu')) 

dummy_input = torch.zeros((1, 1, 48, 48), dtype=torch.float32)


onnx_path = str(ROOT / "api" / "vgg8.onnx")
torch.onnx.export(
    model,                  
    dummy_input,            
    onnx_path,              
    input_names=['input'],  
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Exported the model to {onnx_path} successfully.")