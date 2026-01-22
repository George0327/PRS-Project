"""
Romanian License Plate Character Classifier - PyTorch Training Script
Trains an MLP and exports to ONNX format for use with OpenCV DNN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cv2
import os
from pathlib import Path
import math

# FEATURE EXTRACTION (Same as C++)

def extract_features(img):
    """Extract 30 features from a character image (same as C++ version)"""
    
    # Convert to grayscale 
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Resize to standard size
    resized = cv2.resize(gray, (32, 32))
    
    # Binarize (inverted - dark chars become white)
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    features = []
    
    # 1. Hu Moments (7 features)
    moments = cv2.moments(binary, True)
    hu_moments = cv2.HuMoments(moments).flatten()
    for h in hu_moments:
        # Log transform for numerical stability
        if h != 0:
            features.append(-1.0 * math.copysign(1.0, h) * math.log10(abs(h) + 1e-10))
        else:
            features.append(0.0)
    
    # 2. Pixel density in quadrants (4 features)
    mid_x = binary.shape[1] // 2
    mid_y = binary.shape[0] // 2
    total_pixels = (binary.shape[0] * binary.shape[1]) / 4.0
    
    features.append(np.count_nonzero(binary[0:mid_y, 0:mid_x]) / total_pixels)
    features.append(np.count_nonzero(binary[0:mid_y, mid_x:]) / total_pixels)
    features.append(np.count_nonzero(binary[mid_y:, 0:mid_x]) / total_pixels)
    features.append(np.count_nonzero(binary[mid_y:, mid_x:]) / total_pixels)
    
    # 3. Horizontal projections (8 features)
    for i in range(8):
        start_row = i * binary.shape[0] // 8
        end_row = (i + 1) * binary.shape[0] // 8
        roi = binary[start_row:end_row, :]
        features.append(np.count_nonzero(roi) / (roi.shape[0] * roi.shape[1]))
    
    # 4. Vertical projections (8 features)
    for i in range(8):
        start_col = i * binary.shape[1] // 8
        end_col = (i + 1) * binary.shape[1] // 8
        roi = binary[:, start_col:end_col]
        features.append(np.count_nonzero(roi) / (roi.shape[0] * roi.shape[1]))
    
    # 5. Bounding box features (2 features)
    points = np.column_stack(np.where(binary > 0))
    if len(points) > 0:
        y, x, h, w = cv2.boundingRect(points)
        features.append(w / (h + 1))
        features.append((w * h) / (binary.shape[0] * binary.shape[1]))
    else:
        features.append(1.0)
        features.append(0.0)
    
    # 6. Number of contours (1 feature)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features.append(len(contours) / 10.0)
    
    return np.array(features, dtype=np.float32)


# MLP MODEL

class CharacterMLP(nn.Module):
    def __init__(self, input_size=30, hidden1=128, hidden2=64, num_classes=33):
        super(CharacterMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class CharacterMLPSimple(nn.Module):
    """Simplified model for ONNX export (no Dropout)"""
    def __init__(self, input_size=30, hidden1=128, hidden2=64, num_classes=33):
        super(CharacterMLPSimple, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# TRAINING

def load_training_data(dataset_path):
    """Load training images and extract features"""
    
    # Romanian plates don't use I, O (confusion with 1, 0)
    valid_chars = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"
    
    all_features = []
    all_labels = []
    label_to_char = []
    char_to_label = {}
    
    label_idx = 0
    
    for char in valid_chars:
        char_folder = os.path.join(dataset_path, char)
        
        if not os.path.isdir(char_folder):
            print(f"Warning: Folder not found for '{char}'")
            continue
        
        label_to_char.append(char)
        char_to_label[char] = label_idx
        
        for filename in os.listdir(char_folder):
            if filename.lower().endswith(('.jpg', '.png', '.bmp')):
                filepath = os.path.join(char_folder, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                features = extract_features(img)
                all_features.append(features)
                all_labels.append(label_idx)
        
        print(f"Loaded {sum(1 for fn in os.listdir(char_folder) if fn.lower().endswith(('.jpg','.png','.bmp')))} samples for '{char}' (label {label_idx})")
        label_idx += 1
    
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    print(f"\nTotal: {len(X)} samples, {X.shape[1]} features, {len(label_to_char)} classes")
    
    return X, y, label_to_char, char_to_label


def train_model(X, y, num_classes, epochs=50, batch_size=64, lr=0.001):
    """Train the MLP model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")
    
    # Create data loaders
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)

    # Stratified split 90/10 with fixed seed
    y_np = y
    classes = np.unique(y_np)
    rng = np.random.default_rng(42)
    train_indices = []
    val_indices = []
    for c in classes:
        cls_idx = np.where(y_np == c)[0]
        rng.shuffle(cls_idx)
        n_train = int(0.9 * len(cls_idx))
        train_indices.extend(cls_idx[:n_train])
        val_indices.extend(cls_idx[n_train:])

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = CharacterMLP(input_size=X.shape[1], num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - "
                  f"Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    
    return model


def export_to_onnx(model, output_path, input_size=30):
    """Export model to ONNX format for OpenCV DNN using legacy exporter"""
    
    # Create simplified model without Dropout
    simple_model = CharacterMLPSimple()
    simple_model.fc1.weight.data = model.network[0].weight.data
    simple_model.fc1.bias.data = model.network[0].bias.data
    simple_model.fc2.weight.data = model.network[3].weight.data
    simple_model.fc2.bias.data = model.network[3].bias.data
    simple_model.fc3.weight.data = model.network[6].weight.data
    simple_model.fc3.bias.data = model.network[6].bias.data
    simple_model.eval()
    
    dummy_input = torch.randn(1, input_size)
    
    # Use legacy ONNX exporter (dynamo=False)
    torch.onnx.export(
        simple_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        dynamo=False,
        verbose=False
    )
    
    print(f"Model exported to: {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")


def save_label_mapping(label_to_char, output_path):
    """Save label mapping for C++ to load"""
    with open(output_path, 'w') as f:
        for idx, char in enumerate(label_to_char):
            f.write(f"{idx} {char}\n")
    print(f"Label mapping saved to: {output_path}")


#  MAIN

if __name__ == "__main__":
    # Paths
    script_dir = Path(__file__).parent
    # Prefer binarized dataset if available
    bw_path = script_dir / "TrainingData_BW" / "CNN letter Dataset"
    default_path = script_dir / "TrainingData" / "CNN letter Dataset"
    dataset_path = bw_path if bw_path.exists() else default_path
    model_path = script_dir / "mlp_classifier.onnx"
    label_path = script_dir / "label_mapping.txt"
    
    print("=" * 50)
    print("Romanian License Plate Character Classifier")
    print("=" * 50)
    
    # Load data
    print(f"\nLoading training data from: {dataset_path}")
    X, y, label_to_char, char_to_label = load_training_data(str(dataset_path))
    
    if len(X) == 0:
        print("Error: No training data found!")
        exit(1)
    
    # Train
    model = train_model(X, y, num_classes=len(label_to_char), epochs=50)
    
    # Export
    export_to_onnx(model, str(model_path))
    save_label_mapping(label_to_char, str(label_path))
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"ONNX model: {model_path}")
    print(f"Label mapping: {label_path}")
    print("=" * 50)
