import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
# 导入 StandardScaler 用于目标值标准化
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import math
import sys


# --- 配置参数 ---
INPUT_DATA_DIR = "/home/dzw/Desktop/449-dataset/"           # 输入图像数据 (.xlsx) 目录
OUTPUT_PARAM_FILE = "/home/dzw/Desktop/tag.xlsx"            # 输出参数 (.xlsx) 文件
IMAGE_SIZE = 128                                           # 图像尺寸 (Height=Width)
OUTPUT_SEQ_LEN = 7                                         # 输出参数序列的长度
BATCH_SIZE = 32                                            # 批处理大小
NUM_EPOCHS = 200                                           # 训练轮数 (增加以更好地利用数据)
INITIAL_LEARNING_RATE = 0.0001                             # 初始学习率
CNN_OUTPUT_CHANNELS = 512                                  # CNN编码器输出通道数
EMBED_DIM = 256                                            # 步骤嵌入维度
HIDDEN_DIM = 512                                           # 解码器RNN隐藏层维度 (需能被 NUM_HEADS 整除)
NUM_DECODER_LAYERS = 1                                     # 解码器RNN层数 (增加以提高模型容量)
NUM_HEADS = 8                                              # 多头注意力头数 (HIDDEN_DIM 需能被 NUM_HEADS 整除)
DROPOUT = 0.2                                              # Dropout 比率 (可适当调整)
CLIP_GRAD = 1.0                                            # 梯度裁剪阈值
LR_SCHEDULER_FACTOR = 0.2                                  # 学习率调度器衰减因子
LR_SCHEDULER_PATIENCE = 10                                  # 学习率调度器耐心值 (可适当增加)
LR_SCHEDULER_MIN_LR = 1e-7                                 # 最小学习率

# --- 新增：加权损失配置 ---
# 用于侧重改进特定参数的预测精度，列表长度必须等于 OUTPUT_SEQ_LEN
# 示例：将第 4 (索引3) 和第 7 (索引6) 个参数的损失权重设为 2.0
WEIGHTED_LOSS_FACTORS = [2.0, 2.0, 5.0, 3.0, 1.0, 1.0, 2.0]
assert len(WEIGHTED_LOSS_FACTORS) == OUTPUT_SEQ_LEN, "加权损失因子列表长度必须等于 OUTPUT_SEQ_LEN"
assert HIDDEN_DIM % NUM_HEADS == 0, f"解码器隐藏层维度 (HIDDEN_DIM={HIDDEN_DIM}) 必须能被注意力头数 (NUM_HEADS={NUM_HEADS}) 整除"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 使用设备: {DEVICE} ---")
# (Matplotlib 中文设置，可选)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# --- 数据加载与预处理 ---
def 加载并缩放图像数据_01(filepath):
    """加载单个 Excel 文件表示的图像数据，并进行缩放和检查。"""
    try:
        # 检查文件是否存在且非空
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            # print(f"警告: 文件 {filepath} 不存在或为空，返回零矩阵。")
            return np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        # 读取 Excel 文件
        df = pd.read_excel(filepath, header=None)
        # 检查 DataFrame 是否为空或形状不正确
        if df.empty or df.shape != (IMAGE_SIZE, IMAGE_SIZE):
            # print(f"警告: 文件 {filepath} 为空或形状 ({df.shape}) 不等于 ({IMAGE_SIZE}, {IMAGE_SIZE})，返回零矩阵。")
            return np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        # 转换为 NumPy 数组
        data = df.values.astype(np.float32)
        # 检查并裁剪像素值范围 (15 到 63)
        min_val, max_val = data.min(), data.max()
        if min_val < 15 or max_val > 63:
            # print(f"注意: 文件 {filepath} 像素值范围 [{min_val}, {max_val}] 超出 [15, 63]，进行裁剪。")
            data = np.clip(data, 15, 63)
        # 缩放到 [0, 1] 范围
        scaled_data = ((data - 15.0) / (63.0 - 15.0))
        # 增加通道维度 (1, H, W)
        return scaled_data[np.newaxis, :, :]
    except Exception as e:
        print(f"错误: 加载或缩放文件 {filepath} 时发生异常: {e}")
        return np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) # 发生错误时返回零矩阵

# --- 数据集验证与匹配 ---
print("\n--- 开始数据集验证与匹配 ---")
if not os.path.isdir(INPUT_DATA_DIR): print(f"错误: 输入图像目录 '{INPUT_DATA_DIR}' 不存在！"); sys.exit(1)
if not os.path.isfile(OUTPUT_PARAM_FILE): print(f"错误: 输出参数文件 '{OUTPUT_PARAM_FILE}' 不存在！"); sys.exit(1)
try:
    # 读取参数文件
    param_df_full = pd.read_excel(OUTPUT_PARAM_FILE)
    # 检查参数文件列数是否足够 (文件名 + OUTPUT_SEQ_LEN 个参数)
    if param_df_full.shape[1] < (1 + OUTPUT_SEQ_LEN):
        print(f"错误: 参数文件 '{OUTPUT_PARAM_FILE}' 列数 ({param_df_full.shape[1]}) 不足 {1 + OUTPUT_SEQ_LEN}"); sys.exit(1)
except Exception as e: print(f"错误: 读取参数文件 '{OUTPUT_PARAM_FILE}' 失败: {e}"); sys.exit(1)

# 获取参数文件中的所有文件名 (转换为字符串)
param_filenames = set(param_df_full.iloc[:, 0].astype(str).tolist())
print(f"参数文件中找到 {len(param_filenames)} 个唯一文件名。")

try:
    # 获取图像目录下所有 .xlsx 文件名
    actual_files = set(f for f in os.listdir(INPUT_DATA_DIR) if f.lower().endswith('.xlsx'))
    print(f"图像目录 '{INPUT_DATA_DIR}' 中找到 {len(actual_files)} 个 .xlsx 文件。")
except Exception as e: print(f"错误: 访问输入目录 '{INPUT_DATA_DIR}' 失败: {e}"); sys.exit(1)

# 找到参数文件和图像目录中共有的文件名
common_files = param_filenames.intersection(actual_files)
print(f"成功匹配的文件数: {len(common_files)}")
if not common_files: print("错误: 参数文件中的文件名与图像目录中的文件名无匹配项！请检查文件命名和路径。"); sys.exit(1)

# 过滤参数 DataFrame，只保留匹配到的文件对应的行
param_df_filtered = param_df_full[param_df_full.iloc[:, 0].astype(str).isin(common_files)].reset_index(drop=True)
filtered_filenames = param_df_filtered.iloc[:, 0].astype(str).tolist()
all_indices = list(range(len(filtered_filenames))) # 获取过滤后数据的索引

# --- 数据集划分 ---
print("\n--- 开始划分数据集 (80% 训练, 10% 验证, 10% 测试) ---")
# 划分训练集 (80%) 和临时集 (20%)
train_indices, temp_indices = train_test_split(all_indices, test_size=0.2, random_state=42, shuffle=True)
# 从临时集中划分验证集 (50% of 20% -> 10%) 和测试集 (50% of 20% -> 10%)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42, shuffle=False) # 验证/测试集不需打乱
print(f"数据集索引划分完成: 训练集={len(train_indices)}, 验证集={len(val_indices)}, 测试集={len(test_indices)}")
# 检查划分后是否有空集
if not train_indices or not val_indices or not test_indices: print("错误：划分后至少有一个数据集索引集为空！请增加数据量或调整划分比例。"); sys.exit(1)

# --- 计算图像归一化统计量 (仅在训练集图像上) ---
print("\n--- 计算训练集图像的归一化统计量 (均值和标准差) ---")
pixel_sum_01, pixel_sq_sum_01, num_pixels_01, valid_train_files_count = 0.0, 0.0, 0, 0
# 获取训练集对应的文件名列表
train_filenames_to_calc = param_df_filtered.iloc[train_indices, 0].astype(str).tolist()
# 遍历训练集文件，计算像素值的总和、平方和以及总像素数
for filename in tqdm(train_filenames_to_calc, desc="计算图像均值/标准差"):
    img_path = os.path.join(INPUT_DATA_DIR, filename)
    img_data_01 = 加载并缩放图像数据_01(img_path) # 加载并缩放到 [0, 1]
    # 确保图像加载成功且非全零
    if img_data_01.shape == (1, IMAGE_SIZE, IMAGE_SIZE) and np.any(img_data_01 != 0):
        pixel_sum_01 += img_data_01.sum()
        pixel_sq_sum_01 += (img_data_01 ** 2).sum()
        num_pixels_01 += img_data_01.size
        valid_train_files_count += 1
# 计算均值和标准差
if num_pixels_01 == 0:
    mean_01, std_01 = 0.5, 0.2 # 如果没有有效训练图像，使用默认值
    print("警告：计算归一化统计量时未找到有效的训练图像！使用默认均值=0.5, 标准差=0.2。")
else:
    mean_01 = pixel_sum_01 / num_pixels_01
    # 计算方差，并确保非负
    variance_01 = max((pixel_sq_sum_01 / num_pixels_01) - (mean_01 ** 2), 0)
    # 计算标准差，并确保最小值防止除零错误
    std_01 = max(np.sqrt(variance_01), 1e-6)
print(f"基于 {valid_train_files_count} 个有效训练样本计算得到图像均值: {mean_01:.4f}, 标准差: {std_01:.4f}")
# 定义图像标准化转换
img_transform = transforms.Compose([transforms.Normalize(mean=[mean_01], std=[std_01])])

# --- 计算并应用目标参数标准化 ---
print("\n--- 计算并应用目标参数标准化 (StandardScaler) ---")
# 1. 提取训练集的目标参数 (第 1 到 1+OUTPUT_SEQ_LEN 列)
train_params = param_df_filtered.iloc[train_indices, 1:1+OUTPUT_SEQ_LEN].values.astype(np.float32)

# 检查训练参数中是否有 NaN 或 Inf
if np.isnan(train_params).any() or np.isinf(train_params).any():
    print("错误：训练集的目标参数中包含 NaN 或 Inf 值！请检查原始参数文件。")
    # 可以选择移除包含 NaN/Inf 的行或填充它们，这里选择退出
    nan_rows = np.isnan(train_params).any(axis=1)
    inf_rows = np.isinf(train_params).any(axis=1)
    problem_indices = np.where(nan_rows | inf_rows)[0]
    print(f"包含 NaN/Inf 的训练样本索引（相对于 train_indices）: {problem_indices}")
    print("建议清理原始数据文件或从训练集中移除这些样本。脚本将退出。")
    sys.exit(1)

# 2. 初始化并拟合 StandardScaler (仅使用训练数据！)
target_scaler = StandardScaler()
target_scaler.fit(train_params)

# 3. 打印训练集参数的原始范围和标准化后的统计量 (可选)
print(f"训练集目标参数原始 Min: {np.min(train_params, axis=0)}")
print(f"训练集目标参数原始 Max: {np.max(train_params, axis=0)}")
print(f"目标参数标准化器均值 (Scaler Mean): {target_scaler.mean_}")
print(f"目标参数标准化器标准差 (Scaler Scale): {target_scaler.scale_}") # scale_ 对应标准差

# 4. 定义一个函数来应用标准化 (将在 Dataset 中使用)
def standardize_targets(params, scaler):
    """对输入的参数（单个样本或批量）应用标准化"""
    # scaler.transform 需要 [n_samples, n_features] 输入
    # 如果 params 是一维的 [n_features], 需要 reshape 为 [1, n_features]
    if params.ndim == 1:
        params_reshaped = params.reshape(1, -1)
        return scaler.transform(params_reshaped).flatten() # 返回一维
    else:
        # 如果已经是二维 [n_samples, n_features]，直接 transform
        return scaler.transform(params)

# 5. 定义一个函数来反标准化 (将在评估时使用)
def inverse_standardize_targets(scaled_params, scaler):
    """对输入的标准化参数（单个样本或批量）进行反标准化"""
    if scaled_params.ndim == 1:
        scaled_params_reshaped = scaled_params.reshape(1, -1)
        return scaler.inverse_transform(scaled_params_reshaped).flatten()
    else:
        return scaler.inverse_transform(scaled_params)

# --- 自定义数据集类 (应用目标标准化) ---
class 图像参数数据集(Dataset):
    """
    用于加载图像和对应参数的 PyTorch 数据集类。
    在初始化时加载所有文件名和原始参数，并在 __getitem__ 中加载图像并返回标准化后的参数。
    """
    def __init__(self, input_dir, param_df, file_indices, img_transform, target_scaler, dataset_name="数据集"):
        """
        初始化数据集。
        Args:
            input_dir (str): 图像文件目录。
            param_df (pd.DataFrame): 包含文件名和参数的 DataFrame (已过滤匹配的文件)。
            file_indices (list): 属于该数据集的样本在 param_df 中的索引列表。
            img_transform (transforms.Compose): 应用于图像的转换 (标准化)。
            target_scaler (StandardScaler): 用于目标参数标准化的拟合好的 scaler。
            dataset_name (str): 数据集名称 (用于打印信息)。
        """
        self.input_dir = input_dir
        self.img_transform = img_transform
        self.target_scaler = target_scaler # 存储目标值标准化器
        self.dataset_name = dataset_name

        # 根据提供的索引，获取该数据集对应的子 DataFrame
        self.param_subset_df = param_df.iloc[file_indices].reset_index(drop=True)
        initial_filenames = self.param_subset_df.iloc[:, 0].astype(str).tolist()
        # 加载原始参数值 (暂不标准化)
        initial_parameters_raw = self.param_subset_df.iloc[:, 1:1+OUTPUT_SEQ_LEN].values.astype(np.float32)

        self.filenames = []               # 存储有效样本的文件名
        self.parameters_scaled_list = []  # 存储有效样本的标准化后参数
        skipped_count = 0                 # 记录跳过的无效样本数

        # 预检查数据有效性，过滤掉无效图像或参数包含 NaN/Inf 的样本
        print(f"--- 正在验证 {self.dataset_name} 数据有效性 ---")
        for i, f in enumerate(tqdm(initial_filenames, desc=f"验证 {self.dataset_name}")):
            filepath = os.path.join(self.input_dir, f)
            img_data_check = 加载并缩放图像数据_01(filepath) # 加载并检查图像
            params_raw = initial_parameters_raw[i]           # 获取原始参数

            # 检查图像是否有效 (形状正确且非全零)
            img_valid = img_data_check.shape == (1, IMAGE_SIZE, IMAGE_SIZE) and np.any(img_data_check != 0)
            # 检查参数是否有效 (不含 NaN 或 Inf)
            params_valid = not (np.isnan(params_raw).any() or np.isinf(params_raw).any())

            if img_valid and params_valid:
                self.filenames.append(f)
                # 对单个样本参数进行标准化并存储
                params_scaled = standardize_targets(params_raw, self.target_scaler)
                self.parameters_scaled_list.append(params_scaled)
            else:
                skipped_count += 1
                # (可选) 打印详细跳过原因
                # reason = []
                # if not img_valid: reason.append("图像无效")
                # if not params_valid: reason.append("参数含NaN/Inf")
                # print(f"警告 ({self.dataset_name}): 文件 {f} 因 '{', '.join(reason)}' 被跳过。")

        # 将列表转换为 NumPy 数组，提高后续索引效率
        self.parameters_scaled = np.array(self.parameters_scaled_list, dtype=np.float32) if self.parameters_scaled_list else np.empty((0, OUTPUT_SEQ_LEN), dtype=np.float32)
        final_count = len(self.filenames)

        if final_count == 0 and len(file_indices) > 0:
            print(f"警告: {self.dataset_name} 创建后有效样本数为 0！请检查数据质量或过滤逻辑。")
        elif skipped_count > 0:
             print(f"注意: {self.dataset_name} 初始化时跳过了 {skipped_count} 个无效样本。最终有效样本数: {final_count}")
        else:
             print(f"{self.dataset_name} 初始化完成，有效样本数: {final_count}")


    def __len__(self):
        """返回数据集中有效样本的数量。"""
        return len(self.filenames)

    def __getitem__(self, idx):
        """根据索引加载单个样本的图像和标准化后的参数。"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取文件名和图像路径
        img_name = os.path.join(self.input_dir, self.filenames[idx])
        # 加载图像数据 (已经缩放到 [0, 1])
        image_data_01 = 加载并缩放图像数据_01(img_name)
        # 转换为 PyTorch 张量
        image = torch.from_numpy(image_data_01).float()
        # 应用图像标准化 (减均值除以标准差)
        if self.img_transform:
            image = self.img_transform(image)

        # 获取预先计算好的、对应索引的标准化参数
        params_scaled = self.parameters_scaled[idx]
        params_tensor = torch.from_numpy(params_scaled).float()

        # 增加 NaN/Inf 检查 (防御性编程，理论上初始化时已过滤)
        if torch.isnan(image).any() or torch.isinf(image).any():
            print(f"错误 ({self.dataset_name}): 在 __getitem__ 中发现图像 NaN/Inf for {self.filenames[idx]} (索引 {idx})")
            # 返回一个固定的或零值的样本，或者引发错误
            image = torch.zeros_like(image) # 示例：返回零值图像
        if torch.isnan(params_tensor).any() or torch.isinf(params_tensor).any():
            print(f"错误 ({self.dataset_name}): 在 __getitem__ 中发现参数 NaN/Inf for {self.filenames[idx]} (索引 {idx})")
            params_tensor = torch.zeros_like(params_tensor) # 示例：返回零值参数

        return image, params_tensor # 返回标准化后的图像和参数


# --- 创建数据集和数据加载器 (传入 target_scaler) ---
print("\n--- 创建数据集实例 (应用目标标准化) ---")
train_dataset = 图像参数数据集(INPUT_DATA_DIR, param_df_filtered, train_indices, img_transform, target_scaler, dataset_name="训练集")
val_dataset = 图像参数数据集(INPUT_DATA_DIR, param_df_filtered, val_indices, img_transform, target_scaler, dataset_name="验证集")
test_dataset = 图像参数数据集(INPUT_DATA_DIR, param_df_filtered, test_indices, img_transform, target_scaler, dataset_name="测试集")

print("\n--- 有效数据集样本数 ---")
effective_train_count = len(train_dataset)
effective_val_count = len(val_dataset)
effective_test_count = len(test_dataset)
print(f"有效训练集样本数: {effective_train_count}")
print(f"有效验证集样本数: {effective_val_count}")
print(f"有效测试集样本数: {effective_test_count}")
# 再次检查是否有空的数据集实例
if effective_train_count == 0 or effective_val_count == 0 or effective_test_count == 0:
    print("错误：创建后至少有一个数据集实例的有效样本数为 0！请检查数据或过滤逻辑。"); sys.exit(1)

print("\n--- 创建数据加载器 ---")
# 根据操作系统确定 num_workers
num_workers = 2 if DEVICE.type == 'cuda' else 0 # 在 CUDA 上可以使用多进程，CPU 上或 Windows 设为 0
print(f"数据加载器使用的 num_workers: {num_workers}")

# 创建训练数据加载器，启用 shuffle 和 pin_memory (如果使用 GPU)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'), drop_last=False) # drop_last=False 保留最后一个不完整批次
# 创建验证和测试数据加载器，不启用 shuffle
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
print(f"数据加载器创建完成: 训练批次={len(train_loader)}, 验证批次={len(val_loader)}, 测试批次={len(test_loader)}")


# --- 模型架构 ---

class EncoderCNN(nn.Module):
    """ CNN 编码器，用于从图像中提取空间特征 """
    def __init__(self, output_channels=CNN_OUTPUT_CHANNELS):
        super().__init__()
        # 定义卷积网络层
        self.cnn = nn.Sequential(
            # 输入: [B, 1, 128, 128]
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2), # 输出: [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: [B, 64, 32, 32]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 输出: [B, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: [B, 128, 16, 16]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 增加一层卷积
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2), # 输出: [B, 256, 8, 8]

            nn.Conv2d(256, output_channels, kernel_size=3, stride=1, padding=1), # 输出: [B, C, 16, 16] -> 改为 [B, C, 16, 16]
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 输出: [B, C, 8, 8] -> C=output_channels
        )
        # 计算并存储最终特征图的空间维度
        # self.output_H = 16 # 对应不加最后一个 MaxPool 的 16x16
        # self.output_W = 16
        self.output_H = 8 # 对应加了最后一个 MaxPool 的 8x8
        self.output_W = 8
        self.num_features_spatial = output_channels * self.output_H * self.output_W # L = H'*W'
        self.output_dim = output_channels # C

    def forward(self, x):
        """
        前向传播，提取特征。
        Args:
            x (torch.Tensor): 输入图像张量 [B, 1, H, W]
        Returns:
            torch.Tensor: CNN提取的特征图 [B, C, H', W']
        """
        features = self.cnn(x) # Shape: [B, output_channels, H', W']
        return features



# --- 注意力可视化 ---
print("\n--- 可视化测试集样本的注意力分布 (显示原始参数值) ---")

# --- (修改) 注意力可视化函数，增加保存 128x128 注意力数据到 CSV 的功能 ---
def plot_attention(image_01, result_original, attention, image_size=IMAGE_SIZE, feature_map_size=(8, 8), save_filename="attention_visualization.png"):
    """
    绘制原始图像、预测参数(原始尺度)和每个解码步骤的注意力图。
    同时，将每个步骤上采样后的 128x128 注意力数据保存到对应的 CSV 文件。

    Args:
        image_01 (torch.Tensor): 原始图像张量 [1, H, W] (已缩放到 0-1 范围，未归一化)。
        result_original (np.ndarray): 预测的参数数组 [SEQ_LEN] (原始尺度)。
        attention (np.ndarray): 注意力权重数组 [SEQ_LEN, L] (L = feature_map_H * feature_map_W)，应为平均权重。
        image_size (int): 原始图像的高度/宽度。
        feature_map_size (tuple): CNN 输出特征图的空间尺寸 (H', W')。
        save_filename (str): 保存绘图的基础文件名 (例如 'attention_vis_sample_0_image1.png')。
                                CSV 文件名将由此派生。
    """
    n_steps = attention.shape[0] # 注意力权重的步数 (应为 OUTPUT_SEQ_LEN)
    L = attention.shape[1]       # 展平后的特征数量
    feature_h, feature_w = feature_map_size # 特征图的高和宽

    if L != feature_h * feature_w:
        print(f"错误: 注意力权重长度 {L} 与特征图尺寸 {feature_h}x{feature_w}={feature_h*feature_w} 不匹配!")
        return

    # --- 准备绘制图形 ---
    fig = plt.figure(dpi=300, figsize=(17, 15)) # 画布大小

    # 1. 绘制原始图像 (灰度图, 0-1 范围)
    ax = fig.add_subplot(3, 3, 1) # 3x3 网格的第 1 个位置
    ax.imshow(image_01.squeeze(0).cpu().numpy(), cmap='gray', vmin=0, vmax=1) # 明确数值范围
    ax.set_title("原始图像 (0-1 范围)") # 中文标题
    ax.axis('off') # 不显示坐标轴

    # 2. 显示预测的参数值 (原始尺度)
    param_text = "\n".join([f"参数 {i+1}: {res:.2f}" for i, res in enumerate(result_original)])
    ax = fig.add_subplot(3, 3, 2) # 3x3 网格的第 2 个位置
    ax.text(0.1, 0.5, param_text, fontsize=10, va='center') # 在子图中心偏左显示文本
    ax.set_title("预测参数 (原始尺度)") # 中文标题
    ax.axis('off')

    # --- 准备存储上采样后的注意力数据 ---
    all_upsampled_maps = [] # 用于存储每个步骤的 128x128 注意力图数据
    csv_column_names = []   # 用于存储 CSV 的列名

    # 3. 绘制每个解码步骤的注意力图 (最多显示前 7 个, 适配 3x3 网格)
    plot_limit = min(n_steps, 7)
    print(f"--- 正在为 {os.path.basename(save_filename)} 生成和准备保存 {plot_limit} 个步骤的 128x128 注意力数据 ---")
    for i in range(plot_limit): # 循环每个解码步骤
        ax = fig.add_subplot(3, 3, i + 3) # 从第 3 个子图开始绘制注意力图

        # 获取当前步骤的平均注意力权重并 reshape 为 2D 图
        attn_map_raw = attention[i, :].reshape((feature_h, feature_w))
        # 使用双线性插值将注意力图上采样到原始图像大小
        attn_map_tensor = torch.tensor(attn_map_raw).unsqueeze(0).unsqueeze(0) # [1, 1, H', W']
        # --- 这就是我们需要的 128x128 数据 ---
        upsampled_attn_tensor = nn.functional.interpolate(
            attn_map_tensor, size=(image_size, image_size), mode='bilinear', align_corners=False
        )
        # 转换为 NumPy 数组用于绘图和保存
        upsampled_attn = upsampled_attn_tensor.squeeze().cpu().numpy() # [128, 128]

        # --- 存储当前步骤的上采样数据 (用于后续保存 CSV) ---
        all_upsampled_maps.append(upsampled_attn.flatten()) # 展平成一维 (16384,)
        csv_column_names.append(f"参数{i+1}_注意力权重") # 添加中文列名

        # --- 继续绘图 ---
        # 先绘制背景灰度图
        ax.imshow(image_01.squeeze(0).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        # 再叠加注意力热力图 (使用 'inferno' colormap, alpha 控制透明度)
        im = ax.imshow(upsampled_attn, cmap='inferno', alpha=0.7) # 可调整 alpha
        ax.set_title(f"注意力分布 (参数 {i+1})") # 中文标题
        ax.axis('off')
        # 添加颜色条 (图例) 显示注意力权重范围
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) # 自动确定刻度

    # --- 完成绘图并保存 PNG ---
    plt.tight_layout() # 调整子图布局
    # 从 PNG 文件名提取样本标识用于标题
    try:
        sample_id_for_title = "_".join(os.path.basename(save_filename).split('_')[2:]) # 提取 sample_idx_cleanfilename
        sample_id_for_title = sample_id_for_title.replace('.png','')
    except:
        sample_id_for_title = os.path.basename(save_filename) # 备用标题
    plt.suptitle(f"样本 {sample_id_for_title} 注意力可视化", fontsize=14) # 添加总标题 (中文)
    fig.subplots_adjust(top=0.92) # 为总标题留出空间
    plt.savefig(save_filename) # 保存 PNG 图像
    print(f"注意力可视化图像已保存为: {save_filename}")
    plt.close(fig) # 关闭图形，释放内存

    # --- (新增) 保存 128x128 注意力数据到 CSV ---
    if all_upsampled_maps:
        try:
            # 定义 CSV 文件名 (基于 PNG 文件名)
            csv_filename = save_filename.replace('attention_vis', 'attention_data_128x128').replace('.png', '.csv')

            # 将存储的所有展平后的注意力图 (每个都是 16384,) 按列组合
            # all_upsampled_maps 是 list of arrays, shape [(16384,), (16384,), ...]
            # np.stack(..., axis=1) 将它们按列堆叠成 [16384, n_steps]
            csv_data_array = np.stack(all_upsampled_maps, axis=1)

            # 创建 Pandas DataFrame
            # 行代表 128x128 图像展平后的像素位置 (0 到 16383)
            # 列代表每个参数预测步骤的注意力权重
            attn_data_df = pd.DataFrame(csv_data_array, columns=csv_column_names)

            # (可选) 添加像素位置信息作为索引或列
            # 创建多重索引 (行号, 列号)
            # row_indices = np.arange(image_size)
            # col_indices = np.arange(image_size)
            # multi_index = pd.MultiIndex.from_product([row_indices, col_indices], names=['图像行号', '图像列号'])
            # attn_data_df.index = multi_index
            # 或者添加两列
            pixel_indices = np.arange(image_size * image_size)
            row_coords = pixel_indices // image_size
            col_coords = pixel_indices % image_size
            attn_data_df.insert(0, '图像列号', col_coords)
            attn_data_df.insert(0, '图像行号', row_coords)

            # 保存到 CSV 文件，不包含 Pandas 的默认数字索引
            attn_data_df.to_csv(csv_filename, index=False, float_format='%.8f', encoding='utf-8-sig') # 使用 utf-8-sig 确保中文在 Excel 中正确显示
            print(f"对应的 128x128 注意力分布数据已保存到: {csv_filename}")
            print(f"  - CSV 文件包含 {attn_data_df.shape[0]} 行 (每个像素) 和 {len(csv_column_names)+2} 列。")
            print(f"  - 前两列为像素在 128x128 图像中的'行号'和'列号' (从0开始)。")
            print(f"  - 后续列 ('参数X_注意力权重') 对应 PNG 图中各子图叠加的热力图数据（上采样后）。")

        except Exception as e_csv_save:
            print(f"错误: 保存 128x128 注意力数据到 CSV ({csv_filename}) 时发生异常: {e_csv_save}")
    else:
        print("警告: 未能收集到任何上采样的注意力图数据，无法保存 CSV。")

# --- 确保在调用 plot_attention 的地方没有改动 ---
# 例如，在脚本大约第 768 行附近的可视化循环中，调用方式应保持不变：
# plot_attention(image_01=original_image_01,
#                result_original=predicted_params_original_sample,
#                attention=attention_weights_sample,
#                image_size=IMAGE_SIZE,
#                feature_map_size=(encoder.output_H, encoder.output_W),
#                save_filename=save_filename) # 保持传递 save_filename
class DecoderRNN(nn.Module):
    """ 带多头注意力机制的 GRU 解码器 """
    def __init__(self, embed_dim, decoder_dim, encoder_dim, output_dim, num_layers, num_heads, dropout):
        """
        Args:
            embed_dim (int): 步骤嵌入维度。
            decoder_dim (int): GRU 隐藏层维度 (必须能被 num_heads 整除)。
            encoder_dim (int): Encoder 输出特征维度 (CNN 通道数 C)。
            output_dim (int): 每个时间步的输出维度 (预测单个参数，所以是 1)。
            num_layers (int): GRU 层数。
            num_heads (int): 多头注意力机制的头数。
            dropout (float): Dropout 比率。
        """
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.output_dim = output_dim # 通常为 1
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # --- 解码器组件 ---
        # 1. 步骤嵌入层: 为每个输出步骤 (0 到 OUTPUT_SEQ_LEN-1) 创建可学习的嵌入向量
        self.step_embedding = nn.Embedding(OUTPUT_SEQ_LEN, embed_dim)

        # 2. 多头注意力机制 (Multi-Head Attention, MHA)
        #    Query: 来自解码器隐藏状态 (decoder_dim)
        #    Key/Value: 来自编码器输出 (encoder_dim)
        #    embed_dim (MHA内部维度) 通常设为 query 的维度，即 decoder_dim
        #    batch_first=False 是 PyTorch MHA 的默认值，需要处理输入输出的维度顺序
        self.attention = nn.MultiheadAttention(embed_dim=decoder_dim, # MHA 内部和输出维度
                                               num_heads=num_heads,
                                               kdim=encoder_dim,      # Key 的维度
                                               vdim=encoder_dim,      # Value 的维度
                                               dropout=dropout)        # 注意力内部的 dropout
                                                   # 重要：输入输出格式 (SeqLen, Batch, Dim)

        # 3. GRU 层
        #    输入维度 = 步骤嵌入维度 (embed_dim) + 上下文向量维度 (来自 MHA，维度为 decoder_dim)
        self.rnn = nn.GRU(input_size=embed_dim + decoder_dim,
                          hidden_size=decoder_dim,
                          num_layers=num_layers,
                          batch_first=True,       # GRU 输入/输出使用 (Batch, SeqLen, Feature) 格式
                          dropout=dropout if num_layers > 1 else 0) # 仅在多层 GRU 之间应用 dropout

        # 4. 输出线性层: 将 GRU 的输出 (decoder_dim) 映射到最终的参数值 (output_dim=1)
        self.fc_out = nn.Linear(decoder_dim, output_dim)

        # 5. Dropout 层 (用于嵌入层输出)
        self.dropout_layer = nn.Dropout(p=dropout)

        # 6. (可选但推荐) 用于从 Encoder 输出生成初始隐藏状态的线性层
        #    输入是池化后的 Encoder 特征 [B, encoder_dim]
        #    输出是 [num_layers, B, decoder_dim] (需要 reshape 和 permute)
        self.init_h = nn.Linear(encoder_dim, num_layers * decoder_dim) # 输出展平的隐藏状态
        self.tanh = nn.Tanh() # 可以加个激活函数

    def 创建初始隐藏状态(self, encoder_out_pooled):
        """
        使用池化后的 Encoder 特征来初始化 Decoder 的隐藏状态。
        Args:
            encoder_out_pooled (torch.Tensor): 池化后的 Encoder 特征 [B, encoder_dim]
        Returns:
            torch.Tensor: 初始隐藏状态 [num_layers, B, decoder_dim]
        """
        # [B, encoder_dim] -> [B, num_layers * decoder_dim]
        h0_flat = self.init_h(encoder_out_pooled)
        h0_flat = self.tanh(h0_flat) # 应用激活函数
        # 重新塑形为 [B, num_layers, decoder_dim]
        h0 = h0_flat.view(-1, self.num_layers, self.decoder_dim)
        # 调整维度顺序以匹配 GRU 要求: [num_layers, B, decoder_dim]
        h0 = h0.permute(1, 0, 2).contiguous()
        return h0

    def forward(self, step_index, decoder_hidden, encoder_out_reshaped):
        """
        执行解码器的一个时间步。
        Args:
            step_index (torch.Tensor): 当前时间步的索引 [B], (值为 0 到 OUTPUT_SEQ_LEN-1)
            decoder_hidden (torch.Tensor): 上一个时间步的隐藏状态 [num_layers, B, decoder_dim]
            encoder_out_reshaped (torch.Tensor): 展平并调整维度后的 Encoder 输出 [L, B, encoder_dim]
                                                (L = H'*W', 注意维度顺序为 MHA 服务)
        Returns:
            torch.Tensor: 当前时间步的预测值 [B, output_dim] (output_dim=1)
            torch.Tensor: 当前时间步的隐藏状态 [num_layers, B, decoder_dim]
            torch.Tensor: 当前时间步的平均注意力权重 [B, L] (L=H'*W')
        """
        # --- 1. 获取当前步骤的嵌入向量 ---
        # step_index: [B] -> embedded: [B, embed_dim]
        embedded = self.step_embedding(step_index)
        embedded = self.dropout_layer(embedded) # 应用 dropout

        # --- 2. 计算多头注意力 ---
        # a) 准备 Query: 使用 RNN 最顶层的隐藏状态 [B, decoder_dim]。
        #    MHA 需要 (TargetSeqLen, Batch, QueryDim)，这里 TargetSeqLen=1。
        query = decoder_hidden[-1].unsqueeze(0) # [1, B, decoder_dim]

        # b) Key 和 Value 来自 encoder_out_reshaped [L, B, encoder_dim]。
        #    MHA 会自动处理 kdim 和 vdim。
        # c) 调用 MHA
        #    attn_output: [TargetSeqLen=1, B, embed_dim=decoder_dim] (上下文向量)
        #    attn_weights: [B, TargetSeqLen=1, SourceSeqLen=L] (原始注意力权重)
        context, attn_weights = self.attention(query=query,
                                               key=encoder_out_reshaped,
                                               value=encoder_out_reshaped,
                                               need_weights=True) # 明确需要权重

        # d) 处理输出
        context = context.squeeze(0) # [B, decoder_dim] - 去掉 TargetSeqLen 维度
        # 计算平均注意力权重 (跨头平均，并去掉 TargetSeqLen 维度)
        # attn_weights [B, 1, L] -> alpha [B, L]
        alpha = attn_weights.squeeze(1) # 假设 MHA 实现中如果 num_heads>1，输出已经是平均或需要我们处理？
                                        # Pytorch MHA 的 attn_weights 输出是 (N, L, S) 即 (Batch, TargetSeq, SourceSeq)
                                        # 这里 TargetSeq=1, SourceSeq=L. 所以 squeeze(1) 得到 (B, L) 是正确的。

        # --- 3. 准备 GRU 输入 ---
        # 拼接 (concatenate) 步骤嵌入和上下文向量
        # embedded: [B, embed_dim], context: [B, decoder_dim] -> rnn_input: [B, embed_dim + decoder_dim]
        rnn_input = torch.cat((embedded, context), dim=1)
        # GRU 需要输入形状为 [B, SeqLen=1, Feature]
        rnn_input = rnn_input.unsqueeze(1) # [B, 1, embed_dim + decoder_dim]

        # --- 4. 通过 GRU 层 ---
        # decoder_hidden 已经是 [num_layers, B, decoder_dim]
        # output: [B, 1, decoder_dim], hidden: [num_layers, B, decoder_dim]
        # 注意 GRU 使用 batch_first=True
        output, hidden = self.rnn(rnn_input, decoder_hidden)

        # --- 5. 预测参数值 ---
        # output: [B, 1, decoder_dim] -> squeeze(1): [B, decoder_dim]
        # fc_out: [B, decoder_dim] -> [B, output_dim=1]
        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, alpha

# 注意：不再需要单独的 Attention 类，将使用 nn.MultiheadAttention

class Seq2Seq(nn.Module):
    """ 结合 Encoder 和 Decoder 的 Seq2Seq 模型 """
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src_img):
        """
        模型的前向传播。不再需要 target_params 或 teacher_forcing_ratio。
        Args:
            src_img (torch.Tensor): 输入图像批次 [B, 1, H, W]
        Returns:
            torch.Tensor: 预测的参数序列 [B, OUTPUT_SEQ_LEN] (标准化尺度)
            torch.Tensor: 每一步的平均注意力权重 [B, OUTPUT_SEQ_LEN, L] (L=H'*W')
        """
        batch_size = src_img.shape[0]
        target_len = OUTPUT_SEQ_LEN # 要生成的参数数量
        encoder_H = self.encoder.output_H
        encoder_W = self.encoder.output_W
        L = encoder_H * encoder_W # 展平后的特征数量

        # 创建用于存储解码器输出和注意力权重的张量
        # 注意 decoder 输出维度是 1
        outputs = torch.zeros(batch_size, target_len, self.decoder.output_dim).to(self.device)
        # 存储每一步的注意力权重 (平均权重)
        attentions = torch.zeros(batch_size, target_len, L).to(self.device)

        # --- 1. 通过 Encoder 提取图像特征 ---
        # encoder_out: [B, C, H', W'] C=encoder_dim
        encoder_out = self.encoder(src_img)
        encoder_dim = encoder_out.shape[1]

        # --- 2. 准备 Encoder 输出以供 Decoder 使用 ---
        # a) 展平空间维度并调整顺序以匹配 MHA 的 Key/Value 输入要求 (L, B, C)
        #    [B, C, H', W'] -> [B, C, L] -> [B, L, C] -> [L, B, C]
        encoder_out_reshaped = encoder_out.view(batch_size, encoder_dim, L)
        encoder_out_reshaped = encoder_out_reshaped.permute(2, 0, 1).contiguous() # [L, B, C]

        # b) 对特征图进行全局平均池化，用于生成初始隐藏状态
        #    [B, C, H', W'] -> [B, C, 1, 1] -> [B, C]
        # pool = nn.AdaptiveAvgPool2d((1, 1)) # 可以在 Encoder 内部做，或者在这里做
        # encoder_out_pooled = pool(encoder_out).squeeze(-1).squeeze(-1) # [B, C]
        # 或者直接用 mean
        encoder_out_pooled = torch.mean(encoder_out, dim=[2, 3]) # [B, C]


        # --- 3. 初始化 Decoder 的隐藏状态 ---
        # hidden: [num_layers, B, decoder_dim]
        hidden = self.decoder.创建初始隐藏状态(encoder_out_pooled)

        # --- 4. Decoder 循环，生成输出序列 ---
        # 第一个时间步的输入是步骤索引 0
        step_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for t in range(target_len): # 循环 OUTPUT_SEQ_LEN 次
            # 调用 Decoder 进行单步解码
            # output: [B, 1], hidden: [num_layers, B, dec_dim], alpha: [B, L]
            # 注意传入调整维度后的 encoder_out_reshaped
            output, hidden, alpha = self.decoder(step_idx, hidden, encoder_out_reshaped)

            # 存储当前时间步的预测值和注意力权重
            outputs[:, t, :] = output # 存储到 [B, target_len, 1]
            attentions[:, t, :] = alpha # 存储到 [B, target_len, L]

            # 准备下一个时间步的输入索引 (1, 2, ..., target_len-1)
            next_step_idx_val = t + 1
            # 确保索引不超过最大值 (OUTPUT_SEQ_LEN - 1)
            # （实际上 GRU 的最后一步 hidden state 还会被计算，但对应的 output 不会用到 t+1 的索引）
            step_idx = torch.full((batch_size,), fill_value=next_step_idx_val, dtype=torch.long, device=self.device)


        # outputs 的形状是 [B, target_len, 1]，移除最后一个维度
        outputs = outputs.squeeze(-1) # [B, target_len]
        # attentions 的形状是 [B, target_len, L]

        return outputs, attentions


print("\n--- 初始化模型、损失函数、优化器和调度器 ---")


# --- 实例化模型、损失函数、优化器、调度器 ---
# 实例化 Encoder 和 Decoder
encoder = EncoderCNN(output_channels=CNN_OUTPUT_CHANNELS).to(DEVICE)
decoder = DecoderRNN(embed_dim=EMBED_DIM,
                     decoder_dim=HIDDEN_DIM,
                     encoder_dim=CNN_OUTPUT_CHANNELS,
                     output_dim=1, # 每个时间步预测一个值
                     num_layers=NUM_DECODER_LAYERS,
                     num_heads=NUM_HEADS,
                     dropout=DROPOUT).to(DEVICE)
# 实例化 Seq2Seq 模型
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
print(f'模型可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

# 打印模型参数量
optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)

# 定义优化器
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',        # 监控验证损失 (最小化)
                                           factor=LR_SCHEDULER_FACTOR,    # 学习率衰减因子
                                           patience=LR_SCHEDULER_PATIENCE,# 容忍多少个 epoch 验证损失不下降
                                           verbose=True,                  # 打印学习率变化信息
                                           min_lr=LR_SCHEDULER_MIN_LR)    # 最小学习率

# 定义学习率调度器 (当验证损失不再下降时，降低学习率)
criterion = nn.MSELoss(reduction='none')

# 定义损失函数 (基础MSE，加权将在训练循环中处理)
# reduction='none' 保留每个样本每个参数的损失，方便加权
weighted_loss_factors_tensor = torch.tensor(WEIGHTED_LOSS_FACTORS, dtype=torch.float32, device=DEVICE)

# 将加权因子转换为 Tensor 并移到设备
def train_epoch(model, dataloader, optimizer, criterion, clip, loss_weights):
    """执行一个训练轮次，使用加权损失。"""
    model.train() # 设置模型为训练模式
    epoch_loss = 0.0 # 记录整个 epoch 的累计（加权）损失

    pbar = tqdm(dataloader, desc="训练中", leave=False)
    for src, trg_scaled in pbar: # src: [B, 1, H, W], trg_scaled: [B, SEQ_LEN] (标准化目标)
        src, trg_scaled = src.to(DEVICE), trg_scaled.to(DEVICE)

        optimizer.zero_grad() # 清空梯度

        # 前向传播，获取标准化尺度的预测值
        predictions_scaled, _ = model(src) # predictions_scaled: [B, SEQ_LEN]

        # 计算原始 MSE 损失 (每个元素)
        # criterion(preds, targets) -> [B, SEQ_LEN]
        elementwise_loss = criterion(predictions_scaled, trg_scaled)

        # 应用加权因子
        # loss_weights: [SEQ_LEN] -> unsqueeze(0): [1, SEQ_LEN]
        # elementwise_loss: [B, SEQ_LEN]
        # 使用广播机制进行逐元素相乘
        weighted_elementwise_loss = elementwise_loss * loss_weights.unsqueeze(0)

        # 计算加权损失的平均值 (作为最终的损失值进行反向传播)
        loss = weighted_elementwise_loss.mean()

        # 反向传播
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # 更新模型参数
        optimizer.step()

        # 累加当前批次的加权损失值
        epoch_loss += loss.item()

        # 更新 tqdm 进度条后缀，显示当前批次的加权损失
        pbar.set_postfix(weighted_loss=f"{loss.item():.6f}")

    # 返回平均加权损失
    return epoch_loss / len(dataloader)

# --- 训练与评估函数 (引入加权损失) ---

def evaluate_epoch(model, dataloader, criterion, loss_weights, target_scaler):
    """执行一个评估轮次（验证集或测试集），计算加权损失和收集结果。"""
    model.eval() # 设置模型为评估模式
    epoch_loss = 0.0 # 记录累计加权损失
    all_attentions = []       # 存储第一个批次的注意力权重 (用于可视化)
    all_preds_scaled = []     # 存储所有标准化预测值
    all_trgs_scaled = []      # 存储所有标准化目标值

    with torch.no_grad(): # 禁用梯度计算
        pbar = tqdm(dataloader, desc="评估中", leave=False)
        for i, (src, trg_scaled) in enumerate(pbar):
            src, trg_scaled = src.to(DEVICE), trg_scaled.to(DEVICE)

            # 前向传播，获取标准化预测和注意力权重
            predictions_scaled, attentions = model(src) # preds: [B, SEQ_LEN], attentions: [B, SEQ_LEN, L]

            # 计算加权损失 (与训练过程相同)
            elementwise_loss = criterion(predictions_scaled, trg_scaled)
            weighted_elementwise_loss = elementwise_loss * loss_weights.unsqueeze(0)
            loss = weighted_elementwise_loss.mean()

            # 累加加权损失
            epoch_loss += loss.item()

            # 存储结果 (移到 CPU 并转为 NumPy)
            all_preds_scaled.append(predictions_scaled.cpu().numpy())
            all_trgs_scaled.append(trg_scaled.cpu().numpy())
            # 仅保存第一个批次的注意力权重用于可视化 (减少内存占用)
            if attentions is not None:
                 all_attentions.append(attentions.cpu().numpy())

    # 计算平均加权损失
    avg_loss = epoch_loss / len(dataloader)

    # 合并所有批次的结果
    final_attentions = np.concatenate(all_attentions, axis=0) if all_attentions else None
    final_preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    final_trgs_scaled = np.concatenate(all_trgs_scaled, axis=0)

    # 返回平均加权损失、第一个批次的注意力、所有标准化预测和目标
    return avg_loss, final_attentions, final_preds_scaled, final_trgs_scaled

best_val_loss = float('inf') # 初始化最佳验证损失为正无穷


# --- 训练循环 ---
# 用于存储最佳模型权重
train_losses, val_losses, learning_rates = [], [], [] # 记录训练历史
print("\n--- 开始训练 ---")

for epoch in range(NUM_EPOCHS):
    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} | 当前学习率: {current_lr:.7f} ---")

    # --- 训练阶段 ---
    # 计算的是加权 MSE 损失
    train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP_GRAD, weighted_loss_factors_tensor)
    train_losses.append(train_loss)

    # --- 评估阶段 (验证集) ---
    # 计算的是加权 MSE 损失，用于模型选择和学习率调度
    val_loss, _, _, _ = evaluate_epoch(model, val_loader, criterion, weighted_loss_factors_tensor, target_scaler)
    val_losses.append(val_loss)

    # 使用验证集上的加权损失来更新学习率调度器
    scheduler.step(val_loss)

    # --- 保存最佳模型 ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # 深拷贝当前模型权重作为最佳权重
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f"  验证损失 (加权 MSE) 提升: {val_loss:.6f}。保存模型权重...")
        # 保存最佳权重到文件
        torch.save(best_model_wts, 'best_model.pth')
    else:
        print(f"  验证损失 (加权 MSE) 未提升 ({val_loss:.6f})，当前最佳: {best_val_loss:.6f}。")

    # 打印当前 epoch 的损失信息
    print(f"  训练损失 (加权 MSE): {train_loss:.6f}")
    print(f"  验证损失 (加权 MSE): {val_loss:.6f}")
print("\n--- 训练完成 ---")

if best_model_wts:
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    print("已加载训练过程中保存的最佳模型权重。")

    # 在测试集上评估模型，获取加权损失和标准化结果
    test_loss_weighted_scaled, test_attentions, test_predictions_scaled, test_targets_scaled = evaluate_epoch(
        model, test_loader, criterion, weighted_loss_factors_tensor, target_scaler
    )
    print(f"测试集损失 (加权标准化 MSE): {test_loss_weighted_scaled:.6f}")

    # --- 计算原始尺度的评估指标 ---
    print("\n--- 计算原始尺度的测试集评估指标 ---")
    # 1. 反标准化预测值和目标值
    test_predictions_original = inverse_standardize_targets(test_predictions_scaled, target_scaler)
    test_targets_original = inverse_standardize_targets(test_targets_scaled, target_scaler)

    # 2. 计算原始尺度上的 MAE (所有参数的平均绝对误差)
    mae_loss_fn = nn.L1Loss() # 使用 L1Loss 计算 MAE
    test_mae_original_overall = mae_loss_fn(torch.tensor(test_predictions_original), torch.tensor(test_targets_original))
    print(f"测试集整体平均绝对误差 (MAE, 原始尺度): {test_mae_original_overall.item():.4f}")

    # 3. (可选) 计算原始尺度上每个参数的 MAE
    mae_per_param = np.mean(np.abs(test_predictions_original - test_targets_original), axis=0)
    print("测试集各参数平均绝对误差 (MAE, 原始尺度):")
    for i, mae_val in enumerate(mae_per_param):
        print(f"  参数 {i+1}: {mae_val:.4f}")

    # 4. (可选) 计算原始尺度上的整体 MSE
    mse_loss_fn = nn.MSELoss() # 使用 MSELoss
    test_mse_original_overall = mse_loss_fn(torch.tensor(test_predictions_original), torch.tensor(test_targets_original))
    print(f"测试集整体均方误差 (MSE, 原始尺度): {test_mse_original_overall.item():.4f}")

else:
    print("未找到有效的模型权重，跳过测试阶段。")
    test_attentions = None
    test_predictions_original = None

if (test_dataset and effective_test_count > 0 and
    test_attentions is not None and test_attentions.shape[0] > 0 and encoder is not None):

    print("\n--- 开始保存所有测试集样本的 128x128 注意力矩阵到单独的 CSV 文件 ---")
    FULL_ATTENTION_OUTPUT_DIR = "full_testset_attention_matrices"
    os.makedirs(FULL_ATTENTION_OUTPUT_DIR, exist_ok=True) # 创建输出文件夹

    feature_h, feature_w = encoder.output_H, encoder.output_W
    L = feature_h * feature_w
    seq_len = OUTPUT_SEQ_LEN

    # 遍历所有测试集样本
    for idx in tqdm(range(effective_test_count), desc=f"保存所有注意力矩阵到 '{FULL_ATTENTION_OUTPUT_DIR}'"):
        try:
            original_filename = test_dataset.filenames[idx]
            clean_filename = os.path.basename(original_filename).replace(".xlsx", "")

            # 获取该样本的注意力权重 [SEQ_LEN, L]
            current_sample_attentions = test_attentions[idx]

            # 检查注意力权重形状是否正确
            if current_sample_attentions.shape != (seq_len, L):
                print(f"警告: 样本 {idx} 的注意力权重形状 {current_sample_attentions.shape} 与预期 ({seq_len}, {L}) 不符，跳过保存。")
                continue

            # 遍历每个预测参数（即每个解码步骤）
            for param_idx in range(seq_len):
                # 获取当前参数的注意力权重并 reshape 为 2D 图
                attn_map_raw = current_sample_attentions[param_idx, :].reshape((feature_h, feature_w))

                # 使用双线性插值将注意力图上采样到原始图像大小
                attn_map_tensor = torch.tensor(attn_map_raw).unsqueeze(0).unsqueeze(0) # [1, 1, H', W']
                upsampled_attn_tensor = nn.functional.interpolate(
                    attn_map_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False
                )
                # 转换为 NumPy 数组 [128, 128]
                upsampled_attn = upsampled_attn_tensor.squeeze().cpu().numpy()

                # 定义 CSV 文件名
                csv_path = os.path.join(FULL_ATTENTION_OUTPUT_DIR, f"{clean_filename}_param{param_idx+1}_attention_128x128.csv")

                # 保存到 CSV 文件
                np.savetxt(csv_path, upsampled_attn, delimiter=',', fmt='%.8f')

        except Exception as e:
            print(f"错误: 保存样本 {idx} ({original_filename}) 的注意力矩阵时发生异常: {e}")
            continue # 继续处理下一个样本

    print(f"所有测试集样本的 128x128 注意力矩阵已成功保存到 '{FULL_ATTENTION_OUTPUT_DIR}' 文件夹中。")
else:
    print("警告: 无法保存所有 128x128 注意力矩阵。原因可能为：测试集为空、评估失败、未找到模型权重或注意力结果。")


if best_model_wts is None and NUM_EPOCHS > 0:
     print("警告: 训练结束但未保存任何最佳模型权重 (可能验证损失从未下降)。将使用最后一轮的模型进行测试。")
     best_model_wts = model.state_dict() # 使用最后一轮的权重
elif NUM_EPOCHS == 0:
     print("警告: NUM_EPOCHS 为 0，未进行训练。")
if NUM_EPOCHS > 0 and train_losses and val_losses and learning_rates:
    print("\n--- 保存训练过程数据到 training_log.csv ---")
    try:
        # 确保所有列表长度一致
        if len(train_losses) == NUM_EPOCHS and len(val_losses) == NUM_EPOCHS and len(learning_rates) == NUM_EPOCHS:
            # 创建 Epoch 列表
            epochs = list(range(1, NUM_EPOCHS + 1))
            # 创建 Pandas DataFrame
            training_log_df = pd.DataFrame({
                'Epoch': epochs,
                'Train Loss (Weighted Std)': train_losses,
                'Validation Loss (Weighted Std)': val_losses,
                'Learning Rate': learning_rates
            })
            # 保存到 CSV 文件
            training_log_df.to_csv('training_log.csv', index=False, float_format='%.8f') # 使用 index=False 不保存 DataFrame 索引
            print("训练过程数据已保存到 training_log.csv")
        else:
            print("警告: 训练历史列表长度不匹配 NUM_EPOCHS，无法保存训练日志。")
            print(f"  len(train_losses)={len(train_losses)}, len(val_losses)={len(val_losses)}, len(learning_rates)={len(learning_rates)}, NUM_EPOCHS={NUM_EPOCHS}")

    except Exception as e:
        print(f"错误: 保存训练过程数据到 CSV 时发生异常: {e}")

# --- (新增) 保存训练过程数据到 CSV ---
if NUM_EPOCHS > 0:
    print("\n--- 绘制并保存损失 (加权标准化) 和学习率曲线 ---")
    fig, ax1 = plt.subplots(dpi=300, figsize=(12, 6))

    # 绘制损失曲线
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Weighted Standardized MSE Loss', color=color) # Y轴标签改为加权损失
    ax1.plot(range(1, NUM_EPOCHS + 1), train_losses, color=color, linestyle='-', label='Training Loss (Weighted Std)')
    ax1.plot(range(1, NUM_EPOCHS + 1), val_losses, color=color, linestyle='--', label='Validation Loss (Weighted Std)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # 绘制学习率曲线 (共享 X 轴)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Learning Rate', color=color)
    ax2.plot(range(1, NUM_EPOCHS + 1), learning_rates, color=color, marker='.', linestyle=':', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log') # 使用对数刻度显示学习率
    ax2.legend(loc='upper right')

    fig.tight_layout() # 调整布局防止重叠
    plt.title('Training/Validation Loss (Weighted Standardized) and Learning Rate') # 标题更新
    plt.savefig('loss_lr_curve_weighted_scaled.png') # 保存文件名更新
    plt.close(fig) # 关闭图形
    print("损失(加权标准化)和学习率曲线图已保存为 loss_lr_curve_weighted_scaled.png")
else:
    print("未进行训练 (NUM_EPOCHS=0)，跳过绘制损失曲线。")

# --- 绘制训练历史曲线 (损失是加权标准化尺度上的) ---
print("\n--- 使用最佳模型在测试集上进行测试 ---")

# --- 可视化测试集中的指定样本 ---
if (test_dataset and effective_test_count > 0 and
    test_attentions is not None and test_predictions_original is not None):

    # 定义要可视化的样本索引 (例如：测试集的前 3 个)
    sample_indices_to_plot = list(range(min(3, effective_test_count))) # 最多可视化3个，且不超过测试集大小

    print(f"\n--- 开始为测试集索引 {sample_indices_to_plot} 生成注意力可视化图 ---")

    for idx in sample_indices_to_plot:
        # 确保索引在有效范围内 (理论上上面已保证)
        if idx < effective_test_count and idx < len(test_attentions) and idx < len(test_predictions_original):
            try:
                # 获取原始文件名
                original_filename = test_dataset.filenames[idx]
                img_path_vis = os.path.join(INPUT_DATA_DIR, original_filename)

                # 加载原始图像数据 (0-1 范围，未标准化) 用于可视化
                original_image_01 = torch.from_numpy(加载并缩放图像数据_01(img_path_vis)).float()

                # 获取该样本的原始尺度预测参数和注意力权重
                predicted_params_original_sample = test_predictions_original[idx]
                attention_weights_sample = test_attentions[idx] # [SEQ_LEN, L]

                # 生成唯一的文件名
                clean_filename = os.path.basename(original_filename).replace(".xlsx", "")
                save_filename = f'attention_vis_sample_{idx}_{clean_filename}.png'

                try:
                    # attention_weights_sample 是 numpy 数组 [SEQ_LEN, L]
                    # L = feature_h * feature_w
                    feature_h, feature_w = encoder.output_H, encoder.output_W
                    L = feature_h * feature_w
                    seq_len = OUTPUT_SEQ_LEN  # 或 attention_weights_sample.shape[0]

                    # 检查获取到的注意力权重形状是否符合预期
                    if attention_weights_sample.shape == (seq_len, L):
                        # 创建列名 (代表展平后的特征图位置，例如: Feature_0, Feature_1, ..., Feature_63 for 8x8)
                        col_names = [f'Feature_{i}' for i in range(L)]

                        # 创建行索引名 (代表参数预测步骤，例如: Param_1_Step, Param_2_Step, ...)
                        row_names = [f'Param_{i + 1}_Step' for i in range(seq_len)]

                        # 创建 Pandas DataFrame
                        attn_df = pd.DataFrame(attention_weights_sample, index=row_names, columns=col_names)

                        # 定义对应的 CSV 文件名
                        save_csv_filename = f'attention_data_sample_{idx}_{clean_filename}.csv'

                        # 保存到 CSV，包含行索引（因为它们有意义），并指定浮点数格式
                        attn_df.to_csv(save_csv_filename, float_format='%.6f')
                        print(f"样本 {idx} 的注意力权重原始数据已保存到: {save_csv_filename}")
                    else:
                        print(
                            f"警告: 样本 {idx} 的注意力权重形状 {attention_weights_sample.shape} 与预期 ({seq_len}, {L}) 不符，跳过保存 CSV。")

                except Exception as e_csv:
                    print(f"错误: 为样本索引 {idx} 保存注意力权重 CSV 时发生错误: {e_csv}")

                # 检查图像是否有效
                if torch.any(original_image_01 != 0) and original_image_01.shape == (1, IMAGE_SIZE, IMAGE_SIZE):
                     # 调用绘图函数
                     plot_attention(image_01=original_image_01,
                                   result_original=predicted_params_original_sample,
                                   attention=attention_weights_sample,
                                   image_size=IMAGE_SIZE,
                                   feature_map_size=(encoder.output_H, encoder.output_W), # 从 encoder 获取 H', W'
                                   save_filename=save_filename)
                else:
                    print(f"无法加载或图像无效，跳过可视化样本索引 {idx} (文件: {original_filename})")
            except Exception as e:
                 print(f"为样本索引 {idx} (文件: {original_filename}) 生成可视化时发生错误: {e}")
        else:
            print(f"样本索引 {idx} 超出有效范围或结果数组长度，跳过可视化。")

else:
    print("无法生成注意力图。原因可能为：测试集为空、评估失败、未找到模型权重、或未收集到注意力/预测结果。")

# --- (新增) 保存测试集详细预测结果与误差到 CSV ---
print("\n--- 保存测试集详细预测结果与误差到 CSV ---")

# 检查所需变量是否存在且有效 (在测试阶段应该已经生成)
if ('test_dataset' in locals() and hasattr(test_dataset, 'filenames') and
    'test_targets_original' in locals() and test_targets_original is not None and
    'test_predictions_original' in locals() and test_predictions_original is not None and
    'effective_test_count' in locals() and effective_test_count > 0 and
    len(test_dataset.filenames) == effective_test_count and
    test_targets_original.shape == (effective_test_count, OUTPUT_SEQ_LEN) and
    test_predictions_original.shape == (effective_test_count, OUTPUT_SEQ_LEN)):

    try:
        # 1. 准备数据
        filenames = test_dataset.filenames                       # 获取测试集文件名列表
        targets = test_targets_original                          # 获取真实参数值 (原始尺度) [N, 7]
        predictions = test_predictions_original                  # 获取预测参数值 (原始尺度) [N, 7]

        # 计算每个参数的绝对误差
        errors = np.abs(predictions - targets)                   # 绝对误差 [N, 7]

        # 计算每个样本的平均绝对误差 (MAE)
        sample_mae = np.mean(errors, axis=1)                     # 每个样本的平均绝对误差 [N]

        # 2. 构建 Pandas DataFrame
        # 初始化字典，用于构建 DataFrame
        data_to_save = {'文件名': filenames}

        # 循环添加真实值、预测值和绝对误差列
        for i in range(OUTPUT_SEQ_LEN):
            param_index = i + 1
            data_to_save[f'真实参数_{param_index}'] = targets[:, i]
            data_to_save[f'预测参数_{param_index}'] = predictions[:, i]
            data_to_save[f'绝对误差_{param_index}'] = errors[:, i]

        # 添加样本整体的平均绝对误差列
        data_to_save['样本平均绝对误差'] = sample_mae

        # 创建 DataFrame
        results_df = pd.DataFrame(data_to_save)

        # 重新排序列顺序，使真实、预测、误差相邻（可选，但更易读）
        column_order = ['文件名']
        for i in range(OUTPUT_SEQ_LEN):
            param_index = i + 1
            column_order.extend([f'真实参数_{param_index}', f'预测参数_{param_index}', f'绝对误差_{param_index}'])
        column_order.append('样本平均绝对误差')
        results_df = results_df[column_order]

        # 3. 保存到 CSV 文件
        csv_filename = 'test_predictions_and_errors.csv'
        # 使用 utf-8-sig 编码确保中文在 Excel 中正确显示，不保存 DataFrame 索引
        results_df.to_csv(csv_filename, index=False, float_format='%.6f', encoding='utf-8-sig')

        print(f"测试集预测结果与误差详细对比已保存到: {csv_filename}")
        print(f"  - CSV 文件包含 {results_df.shape[0]} 行 (每个测试样本) 和 {results_df.shape[1]} 列。")
        print(f"  - 列包括：文件名, 真实参数_1, 预测参数_1, 绝对误差_1, ..., 样本平均绝对误差。")

    except Exception as e:
        print(f"错误: 保存测试集结果与误差到 CSV 时发生异常: {e}")

else:
    # 如果缺少必要的数据，打印警告信息
    print("警告: 无法生成测试集结果与误差的 CSV 文件。")
    print("  可能的原因包括：")
    print("  - 测试集为空 (effective_test_count=0)。")
    print("  - 测试评估阶段未成功运行或被跳过。")
    print("  - 变量 'test_targets_original' 或 'test_predictions_original' 未正确生成或格式错误。")
    # 打印一些调试信息（可选）
    print(f"  - 检查点: effective_test_count={effective_test_count if 'effective_test_count' in locals() else '未定义'}")
    print(f"  - 检查点: len(test_dataset.filenames)={len(test_dataset.filenames) if 'test_dataset' in locals() and hasattr(test_dataset, 'filenames') else '未定义'}")
    print(f"  - 检查点: test_targets_original.shape={test_targets_original.shape if 'test_targets_original' in locals() and test_targets_original is not None else '未定义或为None'}")
    print(f"  - 检查点: test_predictions_original.shape={test_predictions_original.shape if 'test_predictions_original' in locals() and test_predictions_original is not None else '未定义或为None'}")

print("\n--- 脚本执行完毕 ---")