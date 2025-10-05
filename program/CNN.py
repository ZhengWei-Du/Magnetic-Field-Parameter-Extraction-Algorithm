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

# --- 配置参数 ---
INPUT_DATA_DIR = "/home/dzw/Desktop/449-dataset/"           # 输入图像数据 (.xlsx) 目录
OUTPUT_PARAM_FILE = "/home/dzw/Desktop/tag.xlsx"            # 输出参数 (.xlsx) 文件
IMAGE_SIZE = 128                                           # 图像尺寸 (Height=Width)
OUTPUT_SEQ_LEN = 7                                         # 输出参数序列的长度
BATCH_SIZE = 32                                            # 批处理大小
NUM_EPOCHS = 200                                           # 训练轮数
INITIAL_LEARNING_RATE = 0.0001                             # 初始学习率
DROPOUT = 0.3                                              # Dropout 比率
CLIP_GRAD = 1.0                                            # 梯度裁剪阈值
LR_SCHEDULER_FACTOR = 0.2                                  # 学习率调度器衰减因子
LR_SCHEDULER_PATIENCE = 10                                 # 学习率调度器耐心值
LR_SCHEDULER_MIN_LR = 1e-7                                 # 最小学习率
DEGRADE_MODEL_CAPACITY_FACTOR = 1.0                        # 模型容量因子

# --- 加权损失配置 ---
WEIGHTED_LOSS_FACTORS = [2.0, 2.0, 5.0, 3.0, 1.0, 1.0, 2.0]
assert len(WEIGHTED_LOSS_FACTORS) == OUTPUT_SEQ_LEN, "加权损失因子列表长度必须等于 OUTPUT_SEQ_LEN"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 使用设备: {DEVICE} ---")
print(f"  - 模型容量因子: {DEGRADE_MODEL_CAPACITY_FACTOR}")

# --- 数据加载与预处理---
def 加载并缩放图像数据_01(filepath):
    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        df = pd.read_excel(filepath, header=None)
        if df.empty or df.shape != (IMAGE_SIZE, IMAGE_SIZE):
            return np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        data = df.values.astype(np.float32)
        data = np.clip(data, 15, 63)
        scaled_data = (data - 15.0) / (63.0 - 15.0) # 缩放到 [0, 1]
        return scaled_data[np.newaxis, :, :]
    except Exception as e:
        print(f"错误: 加载或缩放文件 {filepath} 时发生异常: {e}")
        return np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

print("\n--- 开始数据集验证与匹配 ---")
if not os.path.isdir(INPUT_DATA_DIR): print(f"错误: 输入图像目录 '{INPUT_DATA_DIR}' 不存在！"); sys.exit(1)
if not os.path.isfile(OUTPUT_PARAM_FILE): print(f"错误: 输出参数文件 '{OUTPUT_PARAM_FILE}' 不存在！"); sys.exit(1)
try:
    param_df_full = pd.read_excel(OUTPUT_PARAM_FILE)
    if param_df_full.shape[1] < (1 + OUTPUT_SEQ_LEN):
        print(f"错误: 参数文件 '{OUTPUT_PARAM_FILE}' 列数 ({param_df_full.shape[1]}) 不足 {1 + OUTPUT_SEQ_LEN}"); sys.exit(1)
except Exception as e: print(f"错误: 读取参数文件 '{OUTPUT_PARAM_FILE}' 失败: {e}"); sys.exit(1)

param_filenames = set(param_df_full.iloc[:, 0].astype(str).tolist())
actual_files = set(f for f in os.listdir(INPUT_DATA_DIR) if f.lower().endswith('.xlsx'))
common_files = param_filenames.intersection(actual_files)
print(f"参数文件文件名: {len(param_filenames)}, 图像目录文件数: {len(actual_files)}, 成功匹配文件数: {len(common_files)}")
if not common_files: print("错误: 参数文件中的文件名与图像目录中的文件名无匹配项！"); sys.exit(1)

param_df_filtered = param_df_full[param_df_full.iloc[:, 0].astype(str).isin(common_files)].reset_index(drop=True)
all_indices = list(range(len(param_df_filtered)))

print("\n--- 开始划分数据集 (80% 训练, 10% 验证, 10% 测试) ---")
train_indices, temp_indices = train_test_split(all_indices, test_size=0.2, random_state=42, shuffle=True)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42, shuffle=False)
print(f"数据集索引划分完成: 训练集={len(train_indices)}, 验证集={len(val_indices)}, 测试集={len(test_indices)}")
if not train_indices or not val_indices or not test_indices: print("错误：划分后至少有一个数据集索引集为空！"); sys.exit(1)

print("\n--- 计算训练集图像的归一化统计量 ---")
pixel_sum_01, pixel_sq_sum_01, num_pixels_01, valid_train_files_count = 0.0, 0.0, 0, 0
train_filenames_to_calc = param_df_filtered.iloc[train_indices, 0].astype(str).tolist()
for filename in tqdm(train_filenames_to_calc, desc="计算图像均值/标准差"):
    img_path = os.path.join(INPUT_DATA_DIR, filename)
    img_data_01 = 加载并缩放图像数据_01(img_path)
    if np.any(img_data_01 != 0):
        pixel_sum_01 += img_data_01.sum()
        pixel_sq_sum_01 += (img_data_01 ** 2).sum()
        num_pixels_01 += img_data_01.size
        valid_train_files_count += 1
if num_pixels_01 == 0:
    mean_01, std_01 = 0.5, 0.2
    print("警告：使用默认图像均值=0.5, 标准差=0.2。")
else:
    mean_01 = pixel_sum_01 / num_pixels_01
    variance_01 = max((pixel_sq_sum_01 / num_pixels_01) - (mean_01 ** 2), 0)
    std_01 = max(np.sqrt(variance_01), 1e-6)
print(f"基于 {valid_train_files_count} 个有效训练样本计算得到图像均值: {mean_01:.4f}, 标准差: {std_01:.4f}")
img_transform = transforms.Compose([transforms.Normalize(mean=[mean_01], std=[std_01])])

print("\n--- 计算并应用目标参数标准化 ---")
train_params = param_df_filtered.iloc[train_indices, 1:1+OUTPUT_SEQ_LEN].values.astype(np.float32)
if np.isnan(train_params).any() or np.isinf(train_params).any():
    print("错误：训练集的目标参数中包含 NaN 或 Inf 值！脚本将退出。"); sys.exit(1)
target_scaler = StandardScaler()
target_scaler.fit(train_params)
print(f"目标参数标准化器均值: {target_scaler.mean_}")
print(f"目标参数标准化器标准差: {target_scaler.scale_}")

def standardize_targets(params, scaler):
    return scaler.transform(params.reshape(1, -1)).flatten() if params.ndim == 1 else scaler.transform(params)
def inverse_standardize_targets(scaled_params, scaler):
    return scaler.inverse_transform(scaled_params.reshape(1, -1)).flatten() if scaled_params.ndim == 1 else scaler.inverse_transform(scaled_params)

class 图像参数数据集(Dataset):
    def __init__(self, input_dir, param_df, file_indices, img_transform, target_scaler, dataset_name="数据集"):
        self.input_dir = input_dir
        self.img_transform = img_transform
        self.target_scaler = target_scaler
        self.param_subset_df = param_df.iloc[file_indices].reset_index(drop=True)
        initial_filenames = self.param_subset_df.iloc[:, 0].astype(str).tolist()
        initial_parameters_raw = self.param_subset_df.iloc[:, 1:1+OUTPUT_SEQ_LEN].values.astype(np.float32)
        
        self.filenames, self.parameters_scaled_list = [], []
        skipped_count = 0
        
        for i, f in enumerate(tqdm(initial_filenames, desc=f"验证 {dataset_name}")):
            filepath = os.path.join(self.input_dir, f)
            img_valid = np.any(加载并缩放图像数据_01(filepath) != 0)
            params_valid = not (np.isnan(initial_parameters_raw[i]).any() or np.isinf(initial_parameters_raw[i]).any())
            if img_valid and params_valid:
                self.filenames.append(f)
                self.parameters_scaled_list.append(standardize_targets(initial_parameters_raw[i], self.target_scaler))
            else:
                skipped_count += 1
                
        self.parameters_scaled = np.array(self.parameters_scaled_list, dtype=np.float32)
        if skipped_count > 0: print(f"注意: {dataset_name} 初始化时跳过了 {skipped_count} 个无效样本。")
        print(f"{dataset_name} 初始化完成，有效样本数: {len(self.filenames)}")

    def __len__(self): return len(self.filenames)
    def __getitem__(self, idx):
        img_name = os.path.join(self.input_dir, self.filenames[idx])
        image_data_01 = 加载并缩放图像数据_01(img_name)
        image = self.img_transform(torch.from_numpy(image_data_01).float())
        params_tensor = torch.from_numpy(self.parameters_scaled[idx]).float()
        return image, params_tensor

print("\n--- 创建数据集和数据加载器 ---")
train_dataset = 图像参数数据集(INPUT_DATA_DIR, param_df_filtered, train_indices, img_transform, target_scaler, "训练集")
val_dataset = 图像参数数据集(INPUT_DATA_DIR, param_df_filtered, val_indices, img_transform, target_scaler, "验证集")
test_dataset = 图像参数数据集(INPUT_DATA_DIR, param_df_filtered, test_indices, img_transform, target_scaler, "测试集")
effective_train_count, effective_val_count, effective_test_count = len(train_dataset), len(val_dataset), len(test_dataset)
print(f"有效样本数: 训练集={effective_train_count}, 验证集={effective_val_count}, 测试集={effective_test_count}")
if effective_train_count == 0 or effective_val_count == 0: print("错误：训练集或验证集为空！"); sys.exit(1)

num_workers = 2 if DEVICE.type == 'cuda' else 0
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
print(f"数据加载器创建完成: 训练批次={len(train_loader)}, 验证批次={len(val_loader)}, 测试批次={len(test_loader)}")

# --- 模型架构 (纯CNN) ---
class ImageToParamsCNN(nn.Module):
    def __init__(self, output_dim=OUTPUT_SEQ_LEN, dropout_rate=DROPOUT, capacity_factor=1.0):
        super().__init__()
        
        # 计算各层通道数，根据容量因子调整
        c1 = max(8, int(64 * capacity_factor))
        c2 = max(8, int(128 * capacity_factor))
        c3 = max(8, int(256 * capacity_factor))
        c4 = max(8, int(512 * capacity_factor))
        
        print(f"\n--- CNN模型架构 (容量因子: {capacity_factor}) ---")
        print(f"  - 卷积层通道数: {c1} -> {c2} -> {c3} -> {c4}")
        
        # 特征提取部分 - 卷积层
        self.features = nn.Sequential(
            # 输入: [batch, 1, 128, 128]
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [batch, c1, 64, 64]
            
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [batch, c2, 32, 32]
            
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [batch, c3, 16, 16]
            
            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [batch, c4, 8, 8]
            
            nn.Conv2d(c4, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [batch, c4, 4, 4]
        )
        
        # 全局池化和回归部分
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 计算全连接层输入特征数
        self.fc_input_features = c4
        
        # 回归器
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.fc_input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        
        # 全局池化
        x = self.global_pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 回归预测
        x = self.regressor(x)
        
        return x

print("\n--- 初始化模型、损失函数、优化器和调度器 ---")
model = ImageToParamsCNN(capacity_factor=DEGRADE_MODEL_CAPACITY_FACTOR).to(DEVICE)
print(f'模型可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, verbose=True, min_lr=LR_SCHEDULER_MIN_LR)
criterion = nn.MSELoss(reduction='none')
weighted_loss_factors_tensor = torch.tensor(WEIGHTED_LOSS_FACTORS, dtype=torch.float32, device=DEVICE)

# --- 训练与评估函数 ---
def train_epoch(model, dataloader, optimizer, criterion, clip, loss_weights):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(dataloader, desc="训练中", leave=False)
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
    epoch_loss = 0