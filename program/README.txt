# ==============================================================================
#                            TF_final     使用说明 (README)
# ==============================================================================
#
# **功能:**
# 本脚本实现了一个基于 CNN 编码器和带有多头注意力机制的 GRU 解码器的 Seq2Seq 模型，
# 用于根据输入的 128x128 图像（Excel 文件表示）预测一个包含 7 个参数的序列。
# 脚本包含了数据加载、预处理（图像缩放、目标值标准化）、模型定义、训练、评估、
# 损失/学习率可视化以及注意力图可视化等功能。
#
# **依赖:**
# - Python 3.x
# - PyTorch (torch, torchvision)
# - Scikit-learn (sklearn)
# - Pandas (pandas)
# - Numpy (numpy)
# - Matplotlib (matplotlib)
# - Tqdm (tqdm)
# - openpyxl (用于读取 .xlsx 文件)
# 建议使用 `pip install torch torchvision torchaudio scikit-learn pandas numpy matplotlib tqdm openpyxl` 安装。
#
# **目录结构:**
# 假设脚本名为 `train_predict.py`：
# /path/to/your/project/
# |-- train_predict.py
# |-- 449-dataset/         <-- 包含图像 .xlsx 文件的目录 (由 INPUT_DATA_DIR 指定)
# |   |-- image1.xlsx
# |   |-- image2.xlsx
# |   |-- ...
# |-- tag.xlsx             <-- 包含图像文件名和对应 7 个参数的 Excel 文件 (由 OUTPUT_PARAM_FILE 指定)
#
# **参数文件 (tag.xlsx) 格式:**
# - 第一列: 图像文件名 (例如 "image1.xlsx")
# - 第二列至第八列: 对应的 7 个目标参数值
# - (其他列将被忽略)
#
# **配置参数 (脚本内):**
# - `INPUT_DATA_DIR`: 图像 .xlsx 文件所在的目录路径。
# - `OUTPUT_PARAM_FILE`: 参数 .xlsx 文件的路径。
# - `IMAGE_SIZE`: 图像的尺寸 (应为 128)。
# - `OUTPUT_SEQ_LEN`: 要预测的参数序列长度 (应为 7)。
# - `BATCH_SIZE`: 训练和评估时的批处理大小。
# - `NUM_EPOCHS`: 训练的总轮数 (增加到 200 以更好地利用数据)。
# - `INITIAL_LEARNING_RATE`: 初始学习率。
# - `CNN_OUTPUT_CHANNELS`: CNN 编码器输出的通道数。
# - `EMBED_DIM`: 解码器中步骤嵌入的维度。
# - `HIDDEN_DIM`: GRU 解码器的隐藏层维度 (需能被 NUM_HEADS 整除)。
# - `NUM_DECODER_LAYERS`: GRU 解码器的层数 (增加到 2 以增强模型容量)。
# - `NUM_HEADS`: 多头注意力机制中的头数 (新增)。
# - `DROPOUT`: Dropout 比率。
# - `CLIP_GRAD`: 梯度裁剪阈值。
# - `LR_SCHEDULER_FACTOR`, `LR_SCHEDULER_PATIENCE`, `LR_SCHEDULER_MIN_LR`: 学习率调度器参数。
# - `WEIGHTED_LOSS_FACTORS`: (新增) 用于加权损失，提高特定参数预测精度。列表长度应等于 `OUTPUT_SEQ_LEN`。
#                            例如 `[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0]` 表示第 4 和第 7 个参数的损失权重为 2。
#
# **运行脚本:**
# ```bash
# python train_predict.py
# ```
#
# **输出:**
# - **控制台输出:** 显示设备信息、数据加载/处理进度、训练/验证损失、学习率变化、测试集评估结果（标准化 MSE、原始尺度 MAE/MSE）。
# - **best_model.pth:** 保存验证集上损失最低的模型权重。
# - **loss_lr_curve_scaled.png:** 训练/验证损失（标准化尺度）和学习率随 Epoch 变化的曲线图。
# - **attention_vis_sample_*.png:** 测试集中指定样本的注意力可视化图（包含原始图像、预测参数和注意力叠加图）。
#
# **注意事项:**
# - 确保 `INPUT_DATA_DIR` 和 `OUTPUT_PARAM_FILE` 路径正确。
# - 确保参数文件 `tag.xlsx` 的格式符合要求，且文件名与图像目录中的文件名能匹配。
# - 图像文件应为 128x128 大小。异常文件将被跳过。
# - 目标参数值不应包含 NaN 或 Inf。
# - `HIDDEN_DIM` 必须能被 `NUM_HEADS` 整除。
# - 调整 `WEIGHTED_LOSS_FACTORS` 来侧重于改进特定参数的预测。
# - 如果训练/验证损失仍然很高或不收敛，可以尝试调整超参数（如学习率、网络结构、Dropout、`NUM_EPOCHS` 等）。
#
# ==============================================================================






























