import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import sys
import random

# --- Configuration Parameters ---
INPUT_DATA_DIR = "/home/dzw/Desktop/449-dataset/"           # Input image data (.xlsx) directory
OUTPUT_PARAM_FILE = "/home/dzw/Desktop/tag.xlsx"            # Output parameter (.xlsx) file
IMAGE_SIZE = 128                                           # Image size (Height=Width)
OUTPUT_SEQ_LEN = 7                                         # Output parameter sequence length
BATCH_SIZE = 32                                            # Batch size
NUM_EPOCHS = 200                                           # Training epochs
INITIAL_LEARNING_RATE = 0.0001                             # Initial learning rate
DROPOUT = 0.3                                              # Dropout rate
CLIP_GRAD = 1.0                                            # Gradient clipping threshold
LR_SCHEDULER_FACTOR = 0.2                                  # Learning rate scheduler decay factor
LR_SCHEDULER_PATIENCE = 10                                 # Learning rate scheduler patience
LR_SCHEDULER_MIN_LR = 1e-7                                 # Minimum learning rate
DEGRADE_MODEL_CAPACITY_FACTOR = 1.0                        # Model capacity factor

# --- Weighted Loss Configuration ---
WEIGHTED_LOSS_FACTORS = [2.0, 2.0, 5.0, 3.0, 1.0, 1.0, 2.0]
assert len(WEIGHTED_LOSS_FACTORS) == OUTPUT_SEQ_LEN, "Weighted loss factors list length must equal OUTPUT_SEQ_LEN"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {DEVICE} ---")
print(f"  - Model capacity factor: {DEGRADE_MODEL_CAPACITY_FACTOR}")

# --- Data Loading and Preprocessing ---
def load_and_scale_image_data_01(filepath):
    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        df = pd.read_excel(filepath, header=None)
        if df.empty or df.shape != (IMAGE_SIZE, IMAGE_SIZE):
            return np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        data = df.values.astype(np.float32)
        data = np.clip(data, 15, 63)
        scaled_data = (data - 15.0) / (63.0 - 15.0) # Scale to [0, 1]
        return scaled_data[np.newaxis, :, :]
    except Exception as e:
        print(f"Error: Exception occurred while loading or scaling file {filepath}: {e}")
        return np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

print("\n--- Starting dataset validation and matching ---")
if not os.path.isdir(INPUT_DATA_DIR): print(f"Error: Input image directory '{INPUT_DATA_DIR}' does not exist!"); sys.exit(1)
if not os.path.isfile(OUTPUT_PARAM_FILE): print(f"Error: Output parameter file '{OUTPUT_PARAM_FILE}' does not exist!"); sys.exit(1)
try:
    param_df_full = pd.read_excel(OUTPUT_PARAM_FILE)
    if param_df_full.shape[1] < (1 + OUTPUT_SEQ_LEN):
        print(f"Error: Parameter file '{OUTPUT_PARAM_FILE}' column count ({param_df_full.shape[1]}) is less than {1 + OUTPUT_SEQ_LEN}"); sys.exit(1)
except Exception as e: print(f"Error: Failed to read parameter file '{OUTPUT_PARAM_FILE}': {e}"); sys.exit(1)

param_filenames = set(param_df_full.iloc[:, 0].astype(str).tolist())
actual_files = set(f for f in os.listdir(INPUT_DATA_DIR) if f.lower().endswith('.xlsx'))
common_files = param_filenames.intersection(actual_files)
print(f"Parameter file filenames: {len(param_filenames)}, Image directory files: {len(actual_files)}, Successfully matched files: {len(common_files)}")
if not common_files: print("Error: No matching filenames between parameter file and image directory!"); sys.exit(1)

param_df_filtered = param_df_full[param_df_full.iloc[:, 0].astype(str).isin(common_files)].reset_index(drop=True)
all_indices = list(range(len(param_df_filtered)))

print("\n--- Starting dataset split (80% train, 10% validation, 10% test) ---")
train_indices, temp_indices = train_test_split(all_indices, test_size=0.2, random_state=42, shuffle=True)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42, shuffle=False)
print(f"Dataset index split completed: Train={len(train_indices)}, Validation={len(val_indices)}, Test={len(test_indices)}")
if not train_indices or not val_indices or not test_indices: print("Error: At least one dataset index set is empty after split!"); sys.exit(1)

print("\n--- Calculating normalization statistics for training set images ---")
pixel_sum_01, pixel_sq_sum_01, num_pixels_01, valid_train_files_count = 0.0, 0.0, 0, 0
train_filenames_to_calc = param_df_filtered.iloc[train_indices, 0].astype(str).tolist()
for filename in tqdm(train_filenames_to_calc, desc="Calculating image mean/std"):
    img_path = os.path.join(INPUT_DATA_DIR, filename)
    img_data_01 = load_and_scale_image_data_01(img_path)
    if np.any(img_data_01 != 0):
        pixel_sum_01 += img_data_01.sum()
        pixel_sq_sum_01 += (img_data_01 ** 2).sum()
        num_pixels_01 += img_data_01.size
        valid_train_files_count += 1
if num_pixels_01 == 0:
    mean_01, std_01 = 0.5, 0.2
    print("Warning: Using default image mean=0.5, std=0.2.")
else:
    mean_01 = pixel_sum_01 / num_pixels_01
    variance_01 = max((pixel_sq_sum_01 / num_pixels_01) - (mean_01 ** 2), 0)
    std_01 = max(np.sqrt(variance_01), 1e-6)
print(f"Calculated image mean: {mean_01:.4f}, std: {std_01:.4f} based on {valid_train_files_count} valid training samples")
img_transform = transforms.Compose([transforms.Normalize(mean=[mean_01], std=[std_01])])

print("\n--- Calculating and applying target parameter standardization ---")
train_params = param_df_filtered.iloc[train_indices, 1:1+OUTPUT_SEQ_LEN].values.astype(np.float32)
if np.isnan(train_params).any() or np.isinf(train_params).any():
    print("Error: Training set target parameters contain NaN or Inf values! Script will exit."); sys.exit(1)
target_scaler = StandardScaler()
target_scaler.fit(train_params)
print(f"Target parameter scaler mean: {target_scaler.mean_}")
print(f"Target parameter scaler standard deviation: {target_scaler.scale_}")

def standardize_targets(params, scaler):
    return scaler.transform(params.reshape(1, -1)).flatten() if params.ndim == 1 else scaler.transform(params)
def inverse_standardize_targets(scaled_params, scaler):
    return scaler.inverse_transform(scaled_params.reshape(1, -1)).flatten() if scaled_params.ndim == 1 else scaler.inverse_transform(scaled_params)

class ImageParameterDataset(Dataset):
    def __init__(self, input_dir, param_df, file_indices, img_transform, target_scaler, dataset_name="Dataset"):
        self.input_dir = input_dir
        self.img_transform = img_transform
        self.target_scaler = target_scaler
        self.param_subset_df = param_df.iloc[file_indices].reset_index(drop=True)
        initial_filenames = self.param_subset_df.iloc[:, 0].astype(str).tolist()
        initial_parameters_raw = self.param_subset_df.iloc[:, 1:1+OUTPUT_SEQ_LEN].values.astype(np.float32)
        
        self.filenames, self.parameters_scaled_list = [], []
        skipped_count = 0
        
        for i, f in enumerate(tqdm(initial_filenames, desc=f"Validating {dataset_name}")):
            filepath = os.path.join(self.input_dir, f)
            img_valid = np.any(load_and_scale_image_data_01(filepath) != 0)
            params_valid = not (np.isnan(initial_parameters_raw[i]).any() or np.isinf(initial_parameters_raw[i]).any())
            if img_valid and params_valid:
                self.filenames.append(f)
                self.parameters_scaled_list.append(standardize_targets(initial_parameters_raw[i], self.target_scaler))
            else:
                skipped_count += 1
                
        self.parameters_scaled = np.array(self.parameters_scaled_list, dtype=np.float32)
        if skipped_count > 0: print(f"Note: Skipped {skipped_count} invalid samples during {dataset_name} initialization.")
        print(f"{dataset_name} initialization completed, valid samples: {len(self.filenames)}")

    def __len__(self): return len(self.filenames)
    def __getitem__(self, idx):
        img_name = os.path.join(self.input_dir, self.filenames[idx])
        image_data_01 = load_and_scale_image_data_01(img_name)
        image = self.img_transform(torch.from_numpy(image_data_01).float())
        params_tensor = torch.from_numpy(self.parameters_scaled[idx]).float()
        return image, params_tensor

print("\n--- Creating datasets and data loaders ---")
train_dataset = ImageParameterDataset(INPUT_DATA_DIR, param_df_filtered, train_indices, img_transform, target_scaler, "Training set")
val_dataset = ImageParameterDataset(INPUT_DATA_DIR, param_df_filtered, val_indices, img_transform, target_scaler, "Validation set")
test_dataset = ImageParameterDataset(INPUT_DATA_DIR, param_df_filtered, test_indices, img_transform, target_scaler, "Test set")
effective_train_count, effective_val_count, effective_test_count = len(train_dataset), len(val_dataset), len(test_dataset)
print(f"Valid samples: Training set={effective_train_count}, Validation set={effective_val_count}, Test set={effective_test_count}")
if effective_train_count == 0 or effective_val_count == 0: print("Error: Training set or validation set is empty!"); sys.exit(1)

num_workers = 2 if DEVICE.type == 'cuda' else 0
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
print(f"Data loaders created: Training batches={len(train_loader)}, Validation batches={len(val_loader)}, Test batches={len(test_loader)}")

# --- Model Architecture (MLP: Multi-Layer Perceptron) ---
class ImageToParamsMLP(nn.Module):
    def __init__(self, input_features=IMAGE_SIZE*IMAGE_SIZE, output_dim=OUTPUT_SEQ_LEN, dropout_rate=DROPOUT, capacity_factor=1.0):
        super().__init__()
        
        h1 = max(16, int(2048 * capacity_factor))
        h2 = max(16, int(1024 * capacity_factor))
        h3 = max(16, int(512 * capacity_factor))
        h4 = max(16, int(256 * capacity_factor))
        
        print(f"\n--- Model Architecture (Capacity factor: {capacity_factor}) ---")
        print(f"  - Hidden layer sizes: {h1} -> {h2} -> {h3} -> {h4}")
        
        self.regressor = nn.Sequential(
            nn.Linear(input_features, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(h3, h4),
            nn.BatchNorm1d(h4),
            nn.ReLU(inplace=True),
            
            nn.Linear(h4, output_dim)
        )

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return self.regressor(x_flat)

print("\n--- Initializing model, loss function, optimizer and scheduler ---")
model = ImageToParamsMLP(capacity_factor=DEGRADE_MODEL_CAPACITY_FACTOR).to(DEVICE)
print(f'Model trainable parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, verbose=True, min_lr=LR_SCHEDULER_MIN_LR)
criterion = nn.MSELoss(reduction='none')
weighted_loss_factors_tensor = torch.tensor(WEIGHTED_LOSS_FACTORS, dtype=torch.float32, device=DEVICE)

# --- Training and Evaluation Functions ---
def train_epoch(model, dataloader, optimizer, criterion, clip, loss_weights):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for src, trg_scaled in pbar:
        src, trg_scaled = src.to(DEVICE), trg_scaled.to(DEVICE)
        optimizer.zero_grad()
        predictions_scaled = model(src)
        elementwise_loss = criterion(predictions_scaled, trg_scaled)
        weighted_loss = (elementwise_loss * loss_weights.unsqueeze(0)).mean()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += weighted_loss.item()
        pbar.set_postfix(weighted_loss=f"{weighted_loss.item():.6f}")
    return epoch_loss / len(dataloader)

def evaluate_epoch(model, dataloader, criterion, loss_weights):
    model.eval()
    epoch_loss = 0.0
    all_preds_scaled, all_trgs_scaled = [], []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for src, trg_scaled in pbar:
            src, trg_scaled = src.to(DEVICE), trg_scaled.to(DEVICE)
            predictions_scaled = model(src)
            elementwise_loss = criterion(predictions_scaled, trg_scaled)
            weighted_loss = (elementwise_loss * loss_weights.unsqueeze(0)).mean()
            epoch_loss += weighted_loss.item()
            all_preds_scaled.append(predictions_scaled.cpu().numpy())
            all_trgs_scaled.append(trg_scaled.cpu().numpy())
    avg_loss = epoch_loss / len(dataloader)
    final_preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    final_trgs_scaled = np.concatenate(all_trgs_scaled, axis=0)
    return avg_loss, final_preds_scaled, final_trgs_scaled

# --- Training Loop ---
best_model_wts = None
best_val_loss = float('inf')
train_losses, val_losses, learning_rates = [], [], []
print("\n--- Starting training ---")
for epoch in range(NUM_EPOCHS):
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} | Current learning rate: {current_lr:.7f} ---")
    
    train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP_GRAD, weighted_loss_factors_tensor)
    train_losses.append(train_loss)
    
    val_loss, _, _ = evaluate_epoch(model, val_loader, criterion, weighted_loss_factors_tensor)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    print(f"  Training loss (weighted MSE): {train_loss:.6f}")
    print(f"  Validation loss (weighted MSE): {val_loss:.6f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, 'best_model.pth')
        print(f"  Validation loss improved, saving model weights...")

print("\n--- Training completed ---")
if best_model