import torch
import torch.nn as nn
from torchvision import models
import os

# ==================== CONFIG ====================
# Path model .pth yang sudah jadi
INPUT_MODEL_PATH = 'models/fer_model_v1.2_fusion.pth' 
# Output nama file onnx
OUTPUT_ONNX_PATH = 'fer_resnet34_v1.2.onnx'

INPUT_SIZE = 112
NUM_CLASSES = 5

# ==================== MODEL DEFINITION ====================

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes=5, architecture='resnet34', pretrained=False):
        super(EmotionRecognitionModel, self).__init__()
        self.backbone = models.resnet34(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

def convert():
    if not os.path.exists(INPUT_MODEL_PATH):
        print(f"Error: File {INPUT_MODEL_PATH} tidak ditemukan!")
        return

    print("1. Memuat Model PyTorch...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionRecognitionModel(num_classes=NUM_CLASSES)
    
    # Load weights
    checkpoint = torch.load(INPUT_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    print("2. Membuat Dummy Input...")
    # Batch size 1, 3 channels, 112x112
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
    
    print(f"3. Mengekspor ke ONNX: {OUTPUT_ONNX_PATH}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        OUTPUT_ONNX_PATH,
        export_params=True,
        opset_version=11,          # Versi aman untuk Jetson/TensorRT
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        }
    )
    
    print(f" SUKSES! Model tersimpan: {OUTPUT_ONNX_PATH}")
    

if __name__ == "__main__":
    convert()
