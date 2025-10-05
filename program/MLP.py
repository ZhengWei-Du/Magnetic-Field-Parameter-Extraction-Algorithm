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


# --- 模型架构 (MLP: 多层感知机) ---
class ImageToParamsMLP(nn.Module):
    def __init__(self, input_features=IMAGE_SIZE*IMAGE_SIZE, output_dim=OUTPUT_SEQ_LEN, dropout_rate=DROPOUT, capacity_factor=1.0):
        super().__init__()
        
        h1 = max(16, int(2048 * capacity_factor))
        h2 = max(16, int(1024 * capacity_factor))
        h3 = max(16, int(512 * capacity_factor))
        h4 = max(16, int(256 * capacity_factor))
        
        print(f"\n--- 模型架构 (容量因子: {capacity_factor}) ---")
        print(f"  - 隐藏层大小: {h1} -> {h2} -> {h3} -> {h4}")
        
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

print("\n--- 初始化模型、损失函数、优化器和调度器 ---")
model = ImageToParamsMLP(capacity_factor=DEGRADE_MODEL_CAPACITY_FACTOR).to(DEVICE)
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
    epoch_loss = 0.0
    all_preds_scaled, all_trgs_scaled = [], []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="评估中", leave=False)
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

# --- 训练循环 ---
best_model_wts = None
best_val_loss = float('inf')
train_losses, val_losses, learning_rates = [], [], []
print("\n--- 开始训练 ---")
for epoch in range(NUM_EPOCHS):
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} | 当前学习率: {current_lr:.7f} ---")
    
    train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP_GRAD, weighted_loss_factors_tensor)
    train_losses.append(train_loss)
    
    val_loss, _, _ = evaluate_epoch(model, val_loader, criterion, weighted_loss_factors_tensor)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    print(f"  训练损失 (加权 MSE): {train_loss:.6f}")
    print(f"  验证损失 (加权 MSE): {val_loss:.6f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, 'best_model.pth')
        print(f"  验证损失提升，保存模型权重...")

print("\n--- 训练完成 ---")
if best_model_wts is None:
    if NUM_EPOCHS > 0:
        print("警告: 未保存任何最佳模型，将使用最后一轮的模型进行测试。")
        best_model_wts = model.state_dict()
    else:
        print("错误: 未进行训练，无法测试。")
        sys.exit(1)

model.load_state_dict(best_model_wts)
print("已加载训练过程中保存的最佳模型权重。")


# --- 最终评估 ---
test_predictions_original = None
if effective_test_count > 0:
    print("\n--- 使用最佳模型在测试集上进行测试 ---")
    test_loss_weighted_scaled, test_predictions_scaled, test_targets_scaled = evaluate_epoch(
        model, test_loader, criterion, weighted_loss_factors_tensor
    )
    print(f"测试集损失 (加权标准化 MSE): {test_loss_weighted_scaled:.6f}")

    print("\n--- 计算原始尺度的测试集评估指标 ---")
    test_predictions_original = inverse_standardize_targets(test_predictions_scaled, target_scaler)
    test_targets_original = inverse_standardize_targets(test_targets_scaled, target_scaler)

    mae_per_param = np.mean(np.abs(test_predictions_original - test_targets_original), axis=0)
    print("测试集各参数平均绝对误差 (MAE, 原始尺度):")
    for i, mae_val in enumerate(mae_per_param):
        print(f"  参数 {i+1}: {mae_val:.4f}")
else:
    print("\n--- 测试集为空，跳过最终评估 ---")


# --- 保存训练过程数据和曲线图 ---
if NUM_EPOCHS > 0:
    print("\n--- 保存训练过程数据和绘图 ---")
    try:
        training_log_df = pd.DataFrame({
            'Epoch': range(1, NUM_EPOCHS + 1),
            'Train Loss (Weighted Std)': train_losses,
            'Validation Loss (Weighted Std)': val_losses,
            'Learning Rate': learning_rates
        })
        training_log_df.to_csv('training_log.csv', index=False, float_format='%.8f')
        print("训练过程数据已保存到 training_log.csv")
    except Exception as e:
        print(f"错误: 保存训练日志到 CSV 时发生异常: {e}")

    fig, ax1 = plt.subplots(dpi=300, figsize=(12, 6))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Weighted Standardized MSE Loss', color='tab:red')
    ax1.plot(range(1, len(train_losses) + 1), train_losses, color='tab:red', linestyle='-', label='Training Loss')
    ax1.plot(range(1, len(val_losses) + 1), val_losses, color='tab:orange', linestyle='--', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='tab:blue')
    ax2.plot(range(1, len(learning_rates) + 1), learning_rates, color='tab:blue', marker='.', linestyle=':', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right')
    plt.title('Training/Validation Loss and Learning Rate')
    plt.savefig('loss_lr_curve_weighted_scaled.png')
    plt.close(fig)
    print("损失和学习率曲线图已保存为 loss_lr_curve_weighted_scaled.png")

# --- 保存测试集详细预测结果与误差到 CSV (格式不变) ---
if test_predictions_original is not None:
    print("\n--- 保存测试集详细预测结果与误差到 CSV ---")
    try:
        errors = np.abs(test_predictions_original - test_targets_original)
        sample_mae = np.mean(errors, axis=1)
        data_to_save = {'文件名': test_dataset.filenames}
        column_order = ['文件名']
        for i in range(OUTPUT_SEQ_LEN):
            param_index = i + 1
            data_to_save[f'真实参数_{param_index}'] = test_targets_original[:, i]
            data_to_save[f'预测参数_{param_index}'] = test_predictions_original[:, i]
            data_to_save[f'绝对误差_{param_index}'] = errors[:, i]
            column_order.extend([f'真实参数_{param_index}', f'预测参数_{param_index}', f'绝对误差_{param_index}'])
        data_to_save['样本平均绝对误差'] = sample_mae
        column_order.append('样本平均绝对误差')
        results_df = pd.DataFrame(data_to_save)[column_order]
        results_df.to_csv('test_predictions_and_errors.csv', index=False, float_format='%.6f', encoding='utf-8-sig')
        print(f"测试集预测结果与误差详细对比已保存到: test_predictions_and_errors.csv")
    except Exception as e:
        print(f"错误: 保存测试集结果与误差到 CSV 时发生异常: {e}")
else:
    print("警告: 未生成测试集预测结果，无法保存详细误差文件。")

print("\n--- 脚本执行完毕 ---")
