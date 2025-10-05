import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
# Import StandardScaler for target value standardization
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import math
import sys


# --- Configuration Parameters ---
INPUT_DATA_DIR = "/home/dzw/Desktop/449-dataset/"           # Input image data (.xlsx) directory
OUTPUT_PARAM_FILE = "/home/dzw/Desktop/tag.xlsx"            # Output parameter (.xlsx) file
IMAGE_SIZE = 128                                           # Image size (Height=Width)
OUTPUT_SEQ_LEN = 7                                         # Length of the output parameter sequence
BATCH_SIZE = 32                                            # Batch size
NUM_EPOCHS = 200                                           # Number of epochs (increase for better data utilization)
INITIAL_LEARNING_RATE = 0.0001                             # Initial learning rate
CNN_OUTPUT_CHANNELS = 512                                  # Number of output channels for the CNN encoder
EMBED_DIM = 256                                            # Step embedding dimension
HIDDEN_DIM = 512                                           # Decoder RNN hidden layer dimension (must be divisible by NUM_HEADS)
NUM_DECODER_LAYERS = 1                                     # Number of decoder RNN layers (increase to improve model capacity)
NUM_HEADS = 8                                              # Number of multi-head attention heads (HIDDEN_DIM must be divisible by NUM_HEADS)
DROPOUT = 0.2                                              # Dropout rate (can be adjusted)
CLIP_GRAD = 1.0                                            # Gradient clipping threshold
LR_SCHEDULER_FACTOR = 0.2                                  # Learning rate scheduler decay factor
LR_SCHEDULER_PATIENCE = 10                                 # Learning rate scheduler patience (can be increased)
LR_SCHEDULER_MIN_LR = 1e-7                                 # Minimum learning rate

# --- New: Weighted Loss Configuration ---
# Used to focus on improving the prediction accuracy of specific parameters. The list length must equal OUTPUT_SEQ_LEN.
# Example: Set the loss weights for the 4th (index 3) and 7th (index 6) parameters to 2.0.
WEIGHTED_LOSS_FACTORS = [2.0, 2.0, 5.0, 3.0, 1.0, 1.0, 2.0]
assert len(WEIGHTED_LOSS_FACTORS) == OUTPUT_SEQ_LEN, "Length of the weighted loss factors list must be equal to OUTPUT_SEQ_LEN"
assert HIDDEN_DIM % NUM_HEADS == 0, f"Decoder hidden dimension (HIDDEN_DIM={HIDDEN_DIM}) must be divisible by the number of attention heads (NUM_HEADS={NUM_HEADS})"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {DEVICE} ---")
# (Optional: Matplotlib settings for Chinese characters)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# --- Data Loading and Preprocessing ---
def load_and_scale_image_data_01(filepath):
    """Loads image data from a single Excel file, scales it, and performs checks."""
    try:
        # Check if the file exists and is not empty
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            # print(f"Warning: File {filepath} does not exist or is empty, returning a zero matrix.")
            return np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        # Read the Excel file
        df = pd.read_excel(filepath, header=None)
        # Check if the DataFrame is empty or has an incorrect shape
        if df.empty or df.shape != (IMAGE_SIZE, IMAGE_SIZE):
            # print(f"Warning: File {filepath} is empty or its shape ({df.shape}) is not ({IMAGE_SIZE}, {IMAGE_SIZE}), returning a zero matrix.")
            return np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        # Convert to a NumPy array
        data = df.values.astype(np.float32)
        # Check and clip the pixel value range (15 to 63)
        min_val, max_val = data.min(), data.max()
        if min_val < 15 or max_val > 63:
            # print(f"Note: Pixel value range [{min_val}, {max_val}] in file {filepath} is outside [15, 63]. Clipping values.")
            data = np.clip(data, 15, 63)
        # Scale to the [0, 1] range
        scaled_data = ((data - 15.0) / (63.0 - 15.0))
        # Add a channel dimension (1, H, W)
        return scaled_data[np.newaxis, :, :]
    except Exception as e:
        print(f"Error: An exception occurred while loading or scaling file {filepath}: {e}")
        return np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) # Return a zero matrix on error

# --- Dataset Validation and Matching ---
print("\n--- Starting Dataset Validation and Matching ---")
if not os.path.isdir(INPUT_DATA_DIR): print(f"Error: Input image directory '{INPUT_DATA_DIR}' does not exist!"); sys.exit(1)
if not os.path.isfile(OUTPUT_PARAM_FILE): print(f"Error: Output parameter file '{OUTPUT_PARAM_FILE}' does not exist!"); sys.exit(1)
try:
    # Read the parameter file
    param_df_full = pd.read_excel(OUTPUT_PARAM_FILE)
    # Check if the parameter file has enough columns (filename + OUTPUT_SEQ_LEN parameters)
    if param_df_full.shape[1] < (1 + OUTPUT_SEQ_LEN):
        print(f"Error: Parameter file '{OUTPUT_PARAM_FILE}' has insufficient columns ({param_df_full.shape[1]}), expected at least {1 + OUTPUT_SEQ_LEN}"); sys.exit(1)
except Exception as e: print(f"Error: Failed to read parameter file '{OUTPUT_PARAM_FILE}': {e}"); sys.exit(1)

# Get all filenames from the parameter file (converted to string)
param_filenames = set(param_df_full.iloc[:, 0].astype(str).tolist())
print(f"Found {len(param_filenames)} unique filenames in the parameter file.")

try:
    # Get all .xlsx filenames from the image directory
    actual_files = set(f for f in os.listdir(INPUT_DATA_DIR) if f.lower().endswith('.xlsx'))
    print(f"Found {len(actual_files)} .xlsx files in the image directory '{INPUT_DATA_DIR}'.")
except Exception as e: print(f"Error: Failed to access input directory '{INPUT_DATA_DIR}': {e}"); sys.exit(1)

# Find common filenames between the parameter file and the image directory
common_files = param_filenames.intersection(actual_files)
print(f"Number of successfully matched files: {len(common_files)}")
if not common_files: print("Error: No matching filenames found between the parameter file and the image directory! Please check file naming and paths."); sys.exit(1)

# Filter the parameter DataFrame to keep only rows corresponding to matched files
param_df_filtered = param_df_full[param_df_full.iloc[:, 0].astype(str).isin(common_files)].reset_index(drop=True)
filtered_filenames = param_df_filtered.iloc[:, 0].astype(str).tolist()
all_indices = list(range(len(filtered_filenames))) # Get indices of the filtered data

# --- Dataset Splitting ---
print("\n--- Splitting dataset (80% train, 10% validation, 10% test) ---")
# Split into training set (80%) and a temporary set (20%)
train_indices, temp_indices = train_test_split(all_indices, test_size=0.2, random_state=42, shuffle=True)
# Split the temporary set into validation (50% of 20% -> 10%) and test sets (50% of 20% -> 10%)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42, shuffle=False) # No need to shuffle val/test sets
print(f"Dataset index split complete: Train={len(train_indices)}, Validation={len(val_indices)}, Test={len(test_indices)}")
# Check for empty splits
if not train_indices or not val_indices or not test_indices: print("Error: At least one dataset split is empty! Please increase data volume or adjust split ratios."); sys.exit(1)

# --- Calculate Image Normalization Statistics (on training set images only) ---
print("\n--- Calculating normalization statistics (mean and std) for training set images ---")
pixel_sum_01, pixel_sq_sum_01, num_pixels_01, valid_train_files_count = 0.0, 0.0, 0, 0
# Get the list of filenames for the training set
train_filenames_to_calc = param_df_filtered.iloc[train_indices, 0].astype(str).tolist()
# Iterate through training set files to calculate sum of pixels, sum of squared pixels, and total number of pixels
for filename in tqdm(train_filenames_to_calc, desc="Calculating image mean/std"):
    img_path = os.path.join(INPUT_DATA_DIR, filename)
    img_data_01 = load_and_scale_image_data_01(img_path) # Load and scale to [0, 1]
    # Ensure the image was loaded successfully and is not all zeros
    if img_data_01.shape == (1, IMAGE_SIZE, IMAGE_SIZE) and np.any(img_data_01 != 0):
        pixel_sum_01 += img_data_01.sum()
        pixel_sq_sum_01 += (img_data_01 ** 2).sum()
        num_pixels_01 += img_data_01.size
        valid_train_files_count += 1
# Calculate mean and standard deviation
if num_pixels_01 == 0:
    mean_01, std_01 = 0.5, 0.2 # Use default values if no valid training images are found
    print("Warning: No valid training images found for calculating normalization statistics! Using default mean=0.5, std=0.2.")
else:
    mean_01 = pixel_sum_01 / num_pixels_01
    # Calculate variance, ensuring it's non-negative
    variance_01 = max((pixel_sq_sum_01 / num_pixels_01) - (mean_01 ** 2), 0)
    # Calculate standard deviation, ensuring a minimum value to prevent division by zero
    std_01 = max(np.sqrt(variance_01), 1e-6)
print(f"Image mean: {mean_01:.4f}, std: {std_01:.4f} calculated from {valid_train_files_count} valid training samples.")
# Define the image normalization transform
img_transform = transforms.Compose([transforms.Normalize(mean=[mean_01], std=[std_01])])

# --- Calculate and Apply Target Parameter Standardization ---
print("\n--- Calculating and applying target parameter standardization (StandardScaler) ---")
# 1. Extract target parameters for the training set (columns 1 to 1+OUTPUT_SEQ_LEN)
train_params = param_df_filtered.iloc[train_indices, 1:1+OUTPUT_SEQ_LEN].values.astype(np.float32)

# Check for NaN or Inf in training parameters
if np.isnan(train_params).any() or np.isinf(train_params).any():
    print("Error: The target parameters of the training set contain NaN or Inf values! Please check the original parameter file.")
    # Here, we choose to exit. Alternatively, one could remove or impute rows with NaN/Inf.
    nan_rows = np.isnan(train_params).any(axis=1)
    inf_rows = np.isinf(train_params).any(axis=1)
    problem_indices = np.where(nan_rows | inf_rows)[0]
    print(f"Indices of training samples (relative to train_indices) containing NaN/Inf: {problem_indices}")
    print("It is recommended to clean the original data file or remove these samples from the training set. The script will now exit.")
    sys.exit(1)

# 2. Initialize and fit StandardScaler (using only training data!)
target_scaler = StandardScaler()
target_scaler.fit(train_params)

# 3. (Optional) Print original range and standardized statistics for training set parameters
print(f"Original Min of training set target parameters: {np.min(train_params, axis=0)}")
print(f"Original Max of training set target parameters: {np.max(train_params, axis=0)}")
print(f"Target parameter scaler mean: {target_scaler.mean_}")
print(f"Target parameter scaler scale (std): {target_scaler.scale_}")

# 4. Define a function to apply standardization (will be used in the Dataset)
def standardize_targets(params, scaler):
    """Applies standardization to the input parameters (single sample or batch)."""
    # scaler.transform requires [n_samples, n_features] input
    # If params is 1D [n_features], reshape it to [1, n_features]
    if params.ndim == 1:
        params_reshaped = params.reshape(1, -1)
        return scaler.transform(params_reshaped).flatten() # Return as 1D
    else:
        # If already 2D [n_samples, n_features], transform directly
        return scaler.transform(params)

# 5. Define a function to inverse the standardization (will be used during evaluation)
def inverse_standardize_targets(scaled_params, scaler):
    """Applies inverse standardization to the input standardized parameters (single sample or batch)."""
    if scaled_params.ndim == 1:
        scaled_params_reshaped = scaled_params.reshape(1, -1)
        return scaler.inverse_transform(scaled_params_reshaped).flatten()
    else:
        return scaler.inverse_transform(scaled_params)

# --- Custom Dataset Class (Applying Target Standardization) ---
class ImageParameterDataset(Dataset):
    """
    PyTorch Dataset class for loading images and corresponding parameters.
    It loads all filenames and raw parameters during initialization, and
    loads images and returns standardized parameters in __getitem__.
    """
    def __init__(self, input_dir, param_df, file_indices, img_transform, target_scaler, dataset_name="Dataset"):
        """
        Initializes the dataset.
        Args:
            input_dir (str): Directory of image files.
            param_df (pd.DataFrame): DataFrame containing filenames and parameters (already filtered for matched files).
            file_indices (list): List of indices in param_df that belong to this dataset.
            img_transform (transforms.Compose): Transformations to apply to the images (normalization).
            target_scaler (StandardScaler): The fitted scaler for target parameter standardization.
            dataset_name (str): Name of the dataset (for printing information).
        """
        self.input_dir = input_dir
        self.img_transform = img_transform
        self.target_scaler = target_scaler # Store the target value scaler
        self.dataset_name = dataset_name

        # Get the subset of the DataFrame for this dataset based on the provided indices
        self.param_subset_df = param_df.iloc[file_indices].reset_index(drop=True)
        initial_filenames = self.param_subset_df.iloc[:, 0].astype(str).tolist()
        # Load raw parameter values (not yet standardized)
        initial_parameters_raw = self.param_subset_df.iloc[:, 1:1+OUTPUT_SEQ_LEN].values.astype(np.float32)

        self.filenames = []               # Stores filenames of valid samples
        self.parameters_scaled_list = []  # Stores standardized parameters of valid samples
        skipped_count = 0                 # Counts skipped invalid samples

        # Pre-check data validity, filtering out invalid images or samples with NaN/Inf parameters
        print(f"--- Verifying {self.dataset_name} data validity ---")
        for i, f in enumerate(tqdm(initial_filenames, desc=f"Verifying {self.dataset_name}")):
            filepath = os.path.join(self.input_dir, f)
            img_data_check = load_and_scale_image_data_01(filepath) # Load and check the image
            params_raw = initial_parameters_raw[i]                  # Get the raw parameters

            # Check if the image is valid (correct shape and not all zeros)
            img_valid = img_data_check.shape == (1, IMAGE_SIZE, IMAGE_SIZE) and np.any(img_data_check != 0)
            # Check if the parameters are valid (no NaN or Inf)
            params_valid = not (np.isnan(params_raw).any() or np.isinf(params_raw).any())

            if img_valid and params_valid:
                self.filenames.append(f)
                # Standardize and store parameters for a single sample
                params_scaled = standardize_targets(params_raw, self.target_scaler)
                self.parameters_scaled_list.append(params_scaled)
            else:
                skipped_count += 1
                # (Optional) Print detailed reason for skipping
                # reason = []
                # if not img_valid: reason.append("Invalid image")
                # if not params_valid: reason.append("Params contain NaN/Inf")
                # print(f"Warning ({self.dataset_name}): File {f} skipped due to '{', '.join(reason)}'.")

        # Convert lists to NumPy arrays for more efficient indexing later
        self.parameters_scaled = np.array(self.parameters_scaled_list, dtype=np.float32) if self.parameters_scaled_list else np.empty((0, OUTPUT_SEQ_LEN), dtype=np.float32)
        final_count = len(self.filenames)

        if final_count == 0 and len(file_indices) > 0:
            print(f"Warning: {self.dataset_name} has 0 valid samples after initialization! Please check data quality or filtering logic.")
        elif skipped_count > 0:
             print(f"Note: {self.dataset_name} skipped {skipped_count} invalid samples during initialization. Final valid sample count: {final_count}")
        else:
             print(f"{self.dataset_name} initialization complete. Number of valid samples: {final_count}")


    def __len__(self):
        """Returns the number of valid samples in the dataset."""
        return len(self.filenames)

    def __getitem__(self, idx):
        """Loads a single sample's image and its standardized parameters by index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the filename and image path
        img_name = os.path.join(self.input_dir, self.filenames[idx])
        # Load image data (already scaled to [0, 1])
        image_data_01 = load_and_scale_image_data_01(img_name)
        # Convert to a PyTorch tensor
        image = torch.from_numpy(image_data_01).float()
        # Apply image normalization (subtract mean, divide by std)
        if self.img_transform:
            image = self.img_transform(image)

        # Get the pre-computed standardized parameters for the corresponding index
        params_scaled = self.parameters_scaled[idx]
        params_tensor = torch.from_numpy(params_scaled).float()

        # Add NaN/Inf check (defensive programming, should have been filtered during init)
        if torch.isnan(image).any() or torch.isinf(image).any():
            print(f"Error ({self.dataset_name}): Found image with NaN/Inf in __getitem__ for {self.filenames[idx]} (index {idx})")
            # Return a fixed or zero-value sample, or raise an error
            image = torch.zeros_like(image) # Example: return a zero-value image
        if torch.isnan(params_tensor).any() or torch.isinf(params_tensor).any():
            print(f"Error ({self.dataset_name}): Found parameters with NaN/Inf in __getitem__ for {self.filenames[idx]} (index {idx})")
            params_tensor = torch.zeros_like(params_tensor) # Example: return zero-value parameters

        return image, params_tensor # Return the standardized image and parameters


# --- Create Datasets and DataLoaders (passing target_scaler) ---
print("\n--- Creating dataset instances (with target standardization) ---")
train_dataset = ImageParameterDataset(INPUT_DATA_DIR, param_df_filtered, train_indices, img_transform, target_scaler, dataset_name="Training Set")
val_dataset = ImageParameterDataset(INPUT_DATA_DIR, param_df_filtered, val_indices, img_transform, target_scaler, dataset_name="Validation Set")
test_dataset = ImageParameterDataset(INPUT_DATA_DIR, param_df_filtered, test_indices, img_transform, target_scaler, dataset_name="Test Set")

print("\n--- Effective Dataset Sample Counts ---")
effective_train_count = len(train_dataset)
effective_val_count = len(val_dataset)
effective_test_count = len(test_dataset)
print(f"Effective training samples: {effective_train_count}")
print(f"Effective validation samples: {effective_val_count}")
print(f"Effective test samples: {effective_test_count}")
# Re-check if any dataset instance is empty
if effective_train_count == 0 or effective_val_count == 0 or effective_test_count == 0:
    print("Error: At least one dataset instance has 0 effective samples after creation! Please check data or filtering logic."); sys.exit(1)

print("\n--- Creating DataLoaders ---")
# Determine num_workers based on the operating system
num_workers = 2 if DEVICE.type == 'cuda' else 0 # Use multiprocessing on CUDA, set to 0 on CPU or Windows
print(f"Using num_workers for DataLoaders: {num_workers}")

# Create training DataLoader, enable shuffle and pin_memory (if using GPU)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'), drop_last=False) # drop_last=False keeps the last incomplete batch
# Create validation and test DataLoaders, with shuffle disabled
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
print(f"DataLoaders created: Train batches={len(train_loader)}, Validation batches={len(val_loader)}, Test batches={len(test_loader)}")


# --- Model Architecture ---

class EncoderCNN(nn.Module):
    """ CNN Encoder to extract spatial features from an image """
    def __init__(self, output_channels=CNN_OUTPUT_CHANNELS):
        super().__init__()
        # Define the convolutional network layers
        self.cnn = nn.Sequential(
            # Input: [B, 1, 128, 128]
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2), # Output: [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: [B, 64, 32, 32]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Output: [B, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: [B, 128, 16, 16]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # Added a convolutional layer
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2), # Output: [B, 256, 8, 8]

            nn.Conv2d(256, output_channels, kernel_size=3, stride=1, padding=1), # Output: [B, C, 16, 16] -> changed to [B, C, 16, 16]
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: [B, C, 8, 8] -> C=output_channels
        )
        # Calculate and store the spatial dimensions of the final feature map
        # self.output_H = 16 # Corresponds to 16x16 without the last MaxPool
        # self.output_W = 16
        self.output_H = 8 # Corresponds to 8x8 with the last MaxPool
        self.output_W = 8
        self.num_features_spatial = output_channels * self.output_H * self.output_W # L = H'*W'
        self.output_dim = output_channels # C

    def forward(self, x):
        """
        Forward pass to extract features.
        Args:
            x (torch.Tensor): Input image tensor [B, 1, H, W]
        Returns:
            torch.Tensor: CNN-extracted feature map [B, C, H', W']
        """
        features = self.cnn(x) # Shape: [B, output_channels, H', W']
        return features



# --- Attention Visualization ---
print("\n--- Visualize attention distribution for test set samples (showing original parameter values) ---")

# --- (Modified) Attention visualization function, adds functionality to save 128x128 attention data to CSV ---
def plot_attention(image_01, result_original, attention, image_size=IMAGE_SIZE, feature_map_size=(8, 8), save_filename="attention_visualization.png"):
    """
    Plots the original image, predicted parameters (original scale), and attention map for each decoding step.
    Also saves the upsampled 128x128 attention data for each step to a corresponding CSV file.

    Args:
        image_01 (torch.Tensor): Original image tensor [1, H, W] (scaled to 0-1 range, not normalized).
        result_original (np.ndarray): Array of predicted parameters [SEQ_LEN] (original scale).
        attention (np.ndarray): Array of attention weights [SEQ_LEN, L] (L = feature_map_H * feature_map_W), should be averaged weights.
        image_size (int): Height/width of the original image.
        feature_map_size (tuple): Spatial dimensions of the CNN output feature map (H', W').
        save_filename (str): Base filename for saving the plot (e.g., 'attention_vis_sample_0_image1.png').
                               CSV filenames will be derived from this.
    """
    n_steps = attention.shape[0] # Number of steps in attention weights (should be OUTPUT_SEQ_LEN)
    L = attention.shape[1]       # Number of flattened features
    feature_h, feature_w = feature_map_size # Height and width of the feature map

    if L != feature_h * feature_w:
        print(f"Error: Attention weight length {L} does not match feature map size {feature_h}x{feature_w}={feature_h*feature_w}!")
        return

    # --- Prepare for plotting ---
    fig = plt.figure(dpi=300, figsize=(17, 15)) # Canvas size

    # 1. Plot the original image (grayscale, 0-1 range)
    ax = fig.add_subplot(3, 3, 1) # 1st position in a 3x3 grid
    ax.imshow(image_01.squeeze(0).cpu().numpy(), cmap='gray', vmin=0, vmax=1) # Explicit value range
    ax.set_title("Original Image (0-1 Range)")
    ax.axis('off') # Hide axes

    # 2. Display the predicted parameter values (original scale)
    param_text = "\n".join([f"Parameter {i+1}: {res:.2f}" for i, res in enumerate(result_original)])
    ax = fig.add_subplot(3, 3, 2) # 2nd position in a 3x3 grid
    ax.text(0.1, 0.5, param_text, fontsize=10, va='center') # Display text slightly left of center
    ax.set_title("Predicted Parameters (Original Scale)")
    ax.axis('off')

    # --- Prepare to store upsampled attention data ---
    all_upsampled_maps = [] # To store the 128x128 attention map data for each step
    csv_column_names = []   # To store column names for the CSV

    # 3. Plot the attention map for each decoding step (display up to the first 7 to fit the 3x3 grid)
    plot_limit = min(n_steps, 7)
    print(f"--- Generating and preparing to save 128x128 attention data for {plot_limit} steps for {os.path.basename(save_filename)} ---")
    for i in range(plot_limit): # Loop through each decoding step
        ax = fig.add_subplot(3, 3, i + 3) # Start plotting attention maps from the 3rd subplot

        # Get the average attention weights for the current step and reshape to a 2D map
        attn_map_raw = attention[i, :].reshape((feature_h, feature_w))
        # Upsample the attention map to the original image size using bilinear interpolation
        attn_map_tensor = torch.tensor(attn_map_raw).unsqueeze(0).unsqueeze(0) # [1, 1, H', W']
        # --- This is the 128x128 data we need ---
        upsampled_attn_tensor = nn.functional.interpolate(
            attn_map_tensor, size=(image_size, image_size), mode='bilinear', align_corners=False
        )
        # Convert to a NumPy array for plotting and saving
        upsampled_attn = upsampled_attn_tensor.squeeze().cpu().numpy() # [128, 128]

        # --- Store the upsampled data for the current step (for later CSV saving) ---
        all_upsampled_maps.append(upsampled_attn.flatten()) # Flatten to 1D (16384,)
        csv_column_names.append(f"Param_{i+1}_Attention_Weight") # Add column name

        # --- Continue plotting ---
        # First, draw the background grayscale image
        ax.imshow(image_01.squeeze(0).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        # Then, overlay the attention heatmap (using 'inferno' colormap, alpha controls transparency)
        im = ax.imshow(upsampled_attn, cmap='inferno', alpha=0.7) # Alpha can be adjusted
        ax.set_title(f"Attention Distribution (Parameter {i+1})")
        ax.axis('off')
        # Add a colorbar (legend) to show the range of attention weights
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) # Auto-determine ticks

    # --- Finalize and save the PNG plot ---
    plt.tight_layout() # Adjust subplot layout
    # Extract sample identifier from the PNG filename for the title
    try:
        sample_id_for_title = "_".join(os.path.basename(save_filename).split('_')[2:]) # Extract sample_idx_cleanfilename
        sample_id_for_title = sample_id_for_title.replace('.png','')
    except:
        sample_id_for_title = os.path.basename(save_filename) # Fallback title
    plt.suptitle(f"Attention Visualization for Sample {sample_id_for_title}", fontsize=14)
    fig.subplots_adjust(top=0.92) # Leave space for the main title
    plt.savefig(save_filename) # Save the PNG image
    print(f"Attention visualization image saved as: {save_filename}")
    plt.close(fig) # Close the figure to free up memory

    # --- (New) Save 128x128 attention data to CSV ---
    if all_upsampled_maps:
        try:
            # Define the CSV filename (based on the PNG filename)
            csv_filename = save_filename.replace('attention_vis', 'attention_data_128x128').replace('.png', '.csv')

            # Combine all stored flattened attention maps (each is 16384,) by column
            # all_upsampled_maps is a list of arrays, shape [(16384,), (16384,), ...]
            # np.stack(..., axis=1) stacks them column-wise into [16384, n_steps]
            csv_data_array = np.stack(all_upsampled_maps, axis=1)

            # Create a Pandas DataFrame
            # Rows represent the pixel positions (0 to 16383) of the flattened 128x128 image
            # Columns represent the attention weights for each parameter prediction step
            attn_data_df = pd.DataFrame(csv_data_array, columns=csv_column_names)

            # (Optional) Add pixel location information as an index or columns
            # Create a multi-index (row_number, column_number)
            # row_indices = np.arange(image_size)
            # col_indices = np.arange(image_size)
            # multi_index = pd.MultiIndex.from_product([row_indices, col_indices], names=['image_row', 'image_col'])
            # attn_data_df.index = multi_index
            # Or add two columns
            pixel_indices = np.arange(image_size * image_size)
            row_coords = pixel_indices // image_size
            col_coords = pixel_indices % image_size
            attn_data_df.insert(0, 'image_col', col_coords)
            attn_data_df.insert(0, 'image_row', row_coords)

            # Save to CSV file, without the default Pandas numeric index
            attn_data_df.to_csv(csv_filename, index=False, float_format='%.8f', encoding='utf-8-sig') # Use utf-8-sig to ensure proper display in Excel
            print(f"Corresponding 128x128 attention distribution data saved to: {csv_filename}")
            print(f"  - The CSV file contains {attn_data_df.shape[0]} rows (one for each pixel) and {len(csv_column_names)+2} columns.")
            print(f"  - The first two columns are 'image_row' and 'image_col' (0-indexed) for the pixel in the 128x128 image.")
            print(f"  - Subsequent columns ('Param_X_Attention_Weight') correspond to the heatmap data (after upsampling) overlaid on the subplots in the PNG image.")

        except Exception as e_csv_save:
            print(f"Error: An exception occurred while saving 128x128 attention data to CSV ({csv_filename}): {e_csv_save}")
    else:
        print("Warning: No upsampled attention map data was collected, cannot save CSV.")

# --- Ensure there are no changes where plot_attention is called ---
# For example, in the visualization loop around line 768 of the original script, the call should remain the same:
# plot_attention(image_01=original_image_01,
#                result_original=predicted_params_original_sample,
#                attention=attention_weights_sample,
#                image_size=IMAGE_SIZE,
#                feature_map_size=(encoder.output_H, encoder.output_W),
#                save_filename=save_filename) # Continue passing save_filename
class DecoderRNN(nn.Module):
    """ GRU Decoder with Multi-Head Attention """
    def __init__(self, embed_dim, decoder_dim, encoder_dim, output_dim, num_layers, num_heads, dropout):
        """
        Args:
            embed_dim (int): Step embedding dimension.
            decoder_dim (int): GRU hidden dimension (must be divisible by num_heads).
            encoder_dim (int): Encoder output feature dimension (CNN channel count C).
            output_dim (int): Output dimension for each timestep (1 for predicting a single parameter).
            num_layers (int): Number of GRU layers.
            num_heads (int): Number of heads for the multi-head attention mechanism.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.output_dim = output_dim # Usually 1
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # --- Decoder Components ---
        # 1. Step Embedding Layer: Creates a learnable embedding vector for each output step (0 to OUTPUT_SEQ_LEN-1)
        self.step_embedding = nn.Embedding(OUTPUT_SEQ_LEN, embed_dim)

        # 2. Multi-Head Attention (MHA)
        #    Query: from the decoder's hidden state (decoder_dim)
        #    Key/Value: from the encoder's output (encoder_dim)
        #    embed_dim (MHA internal dimension) is typically set to the query's dimension, i.e., decoder_dim
        #    batch_first=False is the default for PyTorch MHA, requiring handling of input/output dimension order
        self.attention = nn.MultiheadAttention(embed_dim=decoder_dim, # MHA internal and output dimension
                                               num_heads=num_heads,
                                               kdim=encoder_dim,      # Key dimension
                                               vdim=encoder_dim,      # Value dimension
                                               dropout=dropout)       # Dropout within the attention mechanism
                                                   # Important: Input/output format is (SeqLen, Batch, Dim)

        # 3. GRU Layer
        #    Input dimension = step embedding dimension (embed_dim) + context vector dimension (from MHA, which is decoder_dim)
        self.rnn = nn.GRU(input_size=embed_dim + decoder_dim,
                          hidden_size=decoder_dim,
                          num_layers=num_layers,
                          batch_first=True,       # GRU input/output uses (Batch, SeqLen, Feature) format
                          dropout=dropout if num_layers > 1 else 0) # Apply dropout only between GRU layers if multi-layered

        # 4. Output Linear Layer: Maps the GRU output (decoder_dim) to the final parameter value (output_dim=1)
        self.fc_out = nn.Linear(decoder_dim, output_dim)

        # 5. Dropout Layer (for the embedding layer output)
        self.dropout_layer = nn.Dropout(p=dropout)

        # 6. (Optional but recommended) Linear layer to generate the initial hidden state from the Encoder output
        #    Input is the pooled Encoder features [B, encoder_dim]
        #    Output is [num_layers, B, decoder_dim] (requires reshape and permute)
        self.init_h = nn.Linear(encoder_dim, num_layers * decoder_dim) # Outputs a flattened hidden state
        self.tanh = nn.Tanh() # An activation function can be added

    def create_initial_hidden_state(self, encoder_out_pooled):
        """
        Uses the pooled Encoder features to initialize the Decoder's hidden state.
        Args:
            encoder_out_pooled (torch.Tensor): Pooled Encoder features [B, encoder_dim]
        Returns:
            torch.Tensor: Initial hidden state [num_layers, B, decoder_dim]
        """
        # [B, encoder_dim] -> [B, num_layers * decoder_dim]
        h0_flat = self.init_h(encoder_out_pooled)
        h0_flat = self.tanh(h0_flat) # Apply activation function
        # Reshape to [B, num_layers, decoder_dim]
        h0 = h0_flat.view(-1, self.num_layers, self.decoder_dim)
        # Permute dimensions to match GRU requirements: [num_layers, B, decoder_dim]
        h0 = h0.permute(1, 0, 2).contiguous()
        return h0

    def forward(self, step_index, decoder_hidden, encoder_out_reshaped):
        """
        Performs one timestep of the decoder.
        Args:
            step_index (torch.Tensor): Index of the current timestep [B], (values from 0 to OUTPUT_SEQ_LEN-1)
            decoder_hidden (torch.Tensor): Hidden state from the previous timestep [num_layers, B, decoder_dim]
            encoder_out_reshaped (torch.Tensor): Flattened and permuted Encoder output [L, B, encoder_dim]
                                                (L = H'*W', note the dimension order for MHA)
        Returns:
            torch.Tensor: Predicted value for the current timestep [B, output_dim] (output_dim=1)
            torch.Tensor: Hidden state for the current timestep [num_layers, B, decoder_dim]
            torch.Tensor: Averaged attention weights for the current timestep [B, L] (L=H'*W')
        """
        # --- 1. Get the embedding vector for the current step ---
        # step_index: [B] -> embedded: [B, embed_dim]
        embedded = self.step_embedding(step_index)
        embedded = self.dropout_layer(embedded) # Apply dropout

        # --- 2. Calculate multi-head attention ---
        # a) Prepare Query: Use the top layer's hidden state of the RNN [B, decoder_dim].
        #    MHA needs (TargetSeqLen, Batch, QueryDim), where TargetSeqLen=1 here.
        query = decoder_hidden[-1].unsqueeze(0) # [1, B, decoder_dim]

        # b) Key and Value come from encoder_out_reshaped [L, B, encoder_dim].
        #    MHA will automatically handle kdim and vdim.
        # c) Call MHA
        #    attn_output: [TargetSeqLen=1, B, embed_dim=decoder_dim] (context vector)
        #    attn_weights: [B, TargetSeqLen=1, SourceSeqLen=L] (raw attention weights)
        context, attn_weights = self.attention(query=query,
                                               key=encoder_out_reshaped,
                                               value=encoder_out_reshaped,
                                               need_weights=True) # Explicitly request weights

        # d) Process the output
        context = context.squeeze(0) # [B, decoder_dim] - remove the TargetSeqLen dimension
        # Calculate average attention weights (averaged across heads, and remove TargetSeqLen dimension)
        # attn_weights [B, 1, L] -> alpha [B, L]
        # For PyTorch MHA, attn_weights output is (N, L, S) which is (Batch, TargetSeq, SourceSeq)
        # Here TargetSeq=1, SourceSeq=L. So squeezing dim 1 to get (B, L) is correct.
        alpha = attn_weights.squeeze(1)

        # --- 3. Prepare GRU input ---
        # Concatenate the step embedding and the context vector
        # embedded: [B, embed_dim], context: [B, decoder_dim] -> rnn_input: [B, embed_dim + decoder_dim]
        rnn_input = torch.cat((embedded, context), dim=1)
        # GRU requires input shape of [B, SeqLen=1, Feature]
        rnn_input = rnn_input.unsqueeze(1) # [B, 1, embed_dim + decoder_dim]

        # --- 4. Pass through the GRU layer ---
        # decoder_hidden is already [num_layers, B, decoder_dim]
        # output: [B, 1, decoder_dim], hidden: [num_layers, B, decoder_dim]
        # Note that GRU uses batch_first=True
        output, hidden = self.rnn(rnn_input, decoder_hidden)

        # --- 5. Predict the parameter value ---
        # output: [B, 1, decoder_dim] -> squeeze(1): [B, decoder_dim]
        # fc_out: [B, decoder_dim] -> [B, output_dim=1]
        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, alpha

# Note: A separate Attention class is no longer needed; we use nn.MultiheadAttention.

class Seq2Seq(nn.Module):
    """ Seq2Seq model combining the Encoder and Decoder """
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src_img):
        """
        Forward pass of the model. No longer requires target_params or teacher_forcing_ratio.
        Args:
            src_img (torch.Tensor): Input image batch [B, 1, H, W]
        Returns:
            torch.Tensor: Predicted parameter sequence [B, OUTPUT_SEQ_LEN] (standardized scale)
            torch.Tensor: Averaged attention weights for each step [B, OUTPUT_SEQ_LEN, L] (L=H'*W')
        """
        batch_size = src_img.shape[0]
        target_len = OUTPUT_SEQ_LEN # Number of parameters to generate
        encoder_H = self.encoder.output_H
        encoder_W = self.encoder.output_W
        L = encoder_H * encoder_W # Number of flattened features

        # Create tensors to store decoder outputs and attention weights
        # Note: decoder output dimension is 1
        outputs = torch.zeros(batch_size, target_len, self.decoder.output_dim).to(self.device)
        # Stores the attention weights (averaged) for each step
        attentions = torch.zeros(batch_size, target_len, L).to(self.device)

        # --- 1. Extract image features through the Encoder ---
        # encoder_out: [B, C, H', W'] where C=encoder_dim
        encoder_out = self.encoder(src_img)
        encoder_dim = encoder_out.shape[1]

        # --- 2. Prepare Encoder output for the Decoder ---
        # a) Flatten spatial dimensions and permute to match MHA's Key/Value input requirements (L, B, C)
        #    [B, C, H', W'] -> [B, C, L] -> [B, L, C] -> [L, B, C]
        encoder_out_reshaped = encoder_out.view(batch_size, encoder_dim, L)
        encoder_out_reshaped = encoder_out_reshaped.permute(2, 0, 1).contiguous() # [L, B, C]

        # b) Apply global average pooling to the feature map to generate the initial hidden state
        #    [B, C, H', W'] -> [B, C, 1, 1] -> [B, C]
        # pool = nn.AdaptiveAvgPool2d((1, 1)) # Can be done inside the Encoder or here
        # encoder_out_pooled = pool(encoder_out).squeeze(-1).squeeze(-1) # [B, C]
        # Or simply use mean
        encoder_out_pooled = torch.mean(encoder_out, dim=[2, 3]) # [B, C]


        # --- 3. Initialize the Decoder's hidden state ---
        # hidden: [num_layers, B, decoder_dim]
        hidden = self.decoder.create_initial_hidden_state(encoder_out_pooled)

        # --- 4. Decoder loop to generate the output sequence ---
        # The input for the first timestep is the step index 0
        step_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for t in range(target_len): # Loop OUTPUT_SEQ_LEN times
            # Call the Decoder for a single decoding step
            # output: [B, 1], hidden: [num_layers, B, dec_dim], alpha: [B, L]
            # Note that the permuted encoder_out_reshaped is passed
            output, hidden, alpha = self.decoder(step_idx, hidden, encoder_out_reshaped)

            # Store the prediction and attention weight for the current timestep
            outputs[:, t, :] = output # Store in [B, target_len, 1]
            attentions[:, t, :] = alpha # Store in [B, target_len, L]

            # Prepare the input index for the next timestep (1, 2, ..., target_len-1)
            next_step_idx_val = t + 1
            # Ensure the index does not exceed the maximum value (OUTPUT_SEQ_LEN - 1)
            # (The hidden state of the last GRU step will still be calculated, but the corresponding output won't use the t+1 index)
            step_idx = torch.full((batch_size,), fill_value=next_step_idx_val, dtype=torch.long, device=self.device)


        # The shape of outputs is [B, target_len, 1], remove the last dimension
        outputs = outputs.squeeze(-1) # [B, target_len]
        # The shape of attentions is [B, target_len, L]

        return outputs, attentions


print("\n--- Initializing Model, Loss Function, Optimizer, and Scheduler ---")


# --- Instantiate Model, Loss Function, Optimizer, Scheduler ---
# Instantiate Encoder and Decoder
encoder = EncoderCNN(output_channels=CNN_OUTPUT_CHANNELS).to(DEVICE)
decoder = DecoderRNN(embed_dim=EMBED_DIM,
                     decoder_dim=HIDDEN_DIM,
                     encoder_dim=CNN_OUTPUT_CHANNELS,
                     output_dim=1, # Predict one value per timestep
                     num_layers=NUM_DECODER_LAYERS,
                     num_heads=NUM_HEADS,
                     dropout=DROPOUT).to(DEVICE)
# Instantiate the Seq2Seq model
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
print(f'Number of trainable parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

# Print model parameter count
optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)

# Define the optimizer
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',        # Monitor validation loss (minimize)
                                           factor=LR_SCHEDULER_FACTOR,    # Learning rate decay factor
                                           patience=LR_SCHEDULER_PATIENCE,# Number of epochs to wait for improvement
                                           verbose=True,                  # Print learning rate change info
                                           min_lr=LR_SCHEDULER_MIN_LR)    # Minimum learning rate

# Define the learning rate scheduler (reduces LR when validation loss plateaus)
criterion = nn.MSELoss(reduction='none')

# Define the loss function (base MSE, weighting will be handled in the training loop)
# reduction='none' retains the loss for each sample and parameter, facilitating weighting
weighted_loss_factors_tensor = torch.tensor(WEIGHTED_LOSS_FACTORS, dtype=torch.float32, device=DEVICE)

# Convert weighting factors to a Tensor and move to the device
def train_epoch(model, dataloader, optimizer, criterion, clip, loss_weights):
    """Executes one training epoch using weighted loss."""
    model.train() # Set the model to training mode
    epoch_loss = 0.0 # Records the cumulative (weighted) loss for the entire epoch

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for src, trg_scaled in pbar: # src: [B, 1, H, W], trg_scaled: [B, SEQ_LEN] (standardized targets)
        src, trg_scaled = src.to(DEVICE), trg_scaled.to(DEVICE)

        optimizer.zero_grad() # Clear gradients

        # Forward pass to get predictions on the standardized scale
        predictions_scaled, _ = model(src) # predictions_scaled: [B, SEQ_LEN]

        # Calculate the raw MSE loss (element-wise)
        # criterion(preds, targets) -> [B, SEQ_LEN]
        elementwise_loss = criterion(predictions_scaled, trg_scaled)

        # Apply weighting factors
        # loss_weights: [SEQ_LEN] -> unsqueeze(0): [1, SEQ_LEN]
        # elementwise_loss: [B, SEQ_LEN]
        # Element-wise multiplication using broadcasting
        weighted_elementwise_loss = elementwise_loss * loss_weights.unsqueeze(0)

        # Calculate the mean of the weighted loss (as the final loss value for backpropagation)
        loss = weighted_elementwise_loss.mean()

        # Backpropagation
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Update model parameters
        optimizer.step()

        # Accumulate the weighted loss value for the current batch
        epoch_loss += loss.item()

        # Update the tqdm progress bar suffix to show the current batch's weighted loss
        pbar.set_postfix(weighted_loss=f"{loss.item():.6f}")

    # Return the average weighted loss
    return epoch_loss / len(dataloader)

# --- Training and Evaluation Functions (with weighted loss) ---

def evaluate_epoch(model, dataloader, criterion, loss_weights, target_scaler):
    """Executes one evaluation epoch (for validation or test set), calculates weighted loss, and collects results."""
    model.eval() # Set the model to evaluation mode
    epoch_loss = 0.0 # Records cumulative weighted loss
    all_attentions = []       # Stores attention weights from the first batch (for visualization)
    all_preds_scaled = []     # Stores all standardized predictions
    all_trgs_scaled = []      # Stores all standardized targets

    with torch.no_grad(): # Disable gradient calculation
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for i, (src, trg_scaled) in enumerate(pbar):
            src, trg_scaled = src.to(DEVICE), trg_scaled.to(DEVICE)

            # Forward pass to get standardized predictions and attention weights
            predictions_scaled, attentions = model(src) # preds: [B, SEQ_LEN], attentions: [B, SEQ_LEN, L]

            # Calculate weighted loss (same as in training)
            elementwise_loss = criterion(predictions_scaled, trg_scaled)
            weighted_elementwise_loss = elementwise_loss * loss_weights.unsqueeze(0)
            loss = weighted_elementwise_loss.mean()

            # Accumulate weighted loss
            epoch_loss += loss.item()

            # Store results (move to CPU and convert to NumPy)
            all_preds_scaled.append(predictions_scaled.cpu().numpy())
            all_trgs_scaled.append(trg_scaled.cpu().numpy())
            # Save only the attention weights of the first batch for visualization (to reduce memory usage)
            if attentions is not None:
                 all_attentions.append(attentions.cpu().numpy())

    # Calculate average weighted loss
    avg_loss = epoch_loss / len(dataloader)

    # Concatenate results from all batches
    final_attentions = np.concatenate(all_attentions, axis=0) if all_attentions else None
    final_preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    final_trgs_scaled = np.concatenate(all_trgs_scaled, axis=0)

    # Return average weighted loss, attention of the first batch, all standardized predictions and targets
    return avg_loss, final_attentions, final_preds_scaled, final_trgs_scaled

best_val_loss = float('inf') # Initialize best validation loss to positive infinity


# --- Training Loop ---
best_model_wts = None
# For storing the best model weights
train_losses, val_losses, learning_rates = [], [], [] # Record training history
print("\n--- Starting Training ---")

for epoch in range(NUM_EPOCHS):
    # Get the current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} | Current Learning Rate: {current_lr:.7f} ---")

    # --- Training Phase ---
    # The calculated loss is the weighted MSE
    train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP_GRAD, weighted_loss_factors_tensor)
    train_losses.append(train_loss)

    # --- Evaluation Phase (Validation Set) ---
    # The calculated loss is the weighted MSE, used for model selection and learning rate scheduling
    val_loss, _, _, _ = evaluate_epoch(model, val_loader, criterion, weighted_loss_factors_tensor, target_scaler)
    val_losses.append(val_loss)

    # Use the weighted loss on the validation set to update the learning rate scheduler
    scheduler.step(val_loss)

    # --- Save the Best Model ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Deep copy the current model weights as the best weights
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f"  Validation loss (weighted MSE) improved: {val_loss:.6f}. Saving model weights...")
        # Save the best weights to a file
        torch.save(best_model_wts, 'best_model.pth')
    else:
        print(f"  Validation loss (weighted MSE) did not improve ({val_loss:.6f}), current best: {best_val_loss:.6f}.")

    # Print loss information for the current epoch
    print(f"  Training loss (weighted MSE): {train_loss:.6f}")
    print(f"  Validation loss (weighted MSE): {val_loss:.6f}")
print("\n--- Training Complete ---")

if best_model_wts:
    # Load the best model weights
    model.load_state_dict(best_model_wts)
    print("Loaded the best model weights saved during training.")

    # Evaluate the model on the test set to get weighted loss and standardized results
    test_loss_weighted_scaled, test_attentions, test_predictions_scaled, test_targets_scaled = evaluate_epoch(
        model, test_loader, criterion, weighted_loss_factors_tensor, target_scaler
    )
    print(f"Test set loss (weighted standardized MSE): {test_loss_weighted_scaled:.6f}")

    # --- Calculate Evaluation Metrics on the Original Scale ---
    print("\n--- Calculating test set evaluation metrics on the original scale ---")
    # 1. Inverse standardize the predicted values and target values
    test_predictions_original = inverse_standardize_targets(test_predictions_scaled, target_scaler)
    test_targets_original = inverse_standardize_targets(test_targets_scaled, target_scaler)

    # 2. Calculate MAE on the original scale (Mean Absolute Error for all parameters)
    mae_loss_fn = nn.L1Loss() # Use L1Loss to calculate MAE
    test_mae_original_overall = mae_loss_fn(torch.tensor(test_predictions_original), torch.tensor(test_targets_original))
    print(f"Overall Test Set Mean Absolute Error (MAE, original scale): {test_mae_original_overall.item():.4f}")

    # 3. (Optional) Calculate MAE for each parameter on the original scale
    mae_per_param = np.mean(np.abs(test_predictions_original - test_targets_original), axis=0)
    print("Test Set Mean Absolute Error per parameter (MAE, original scale):")
    for i, mae_val in enumerate(mae_per_param):
        print(f"  Parameter {i+1}: {mae_val:.4f}")

    # 4. (Optional) Calculate overall MSE on the original scale
    mse_loss_fn = nn.MSELoss() # Use MSELoss
    test_mse_original_overall = mse_loss_fn(torch.tensor(test_predictions_original), torch.tensor(test_targets_original))
    print(f"Overall Test Set Mean Squared Error (MSE, original scale): {test_mse_original_overall.item():.4f}")

else:
    print("No valid model weights found, skipping the testing phase.")
    test_attentions = None
    test_predictions_original = None

if (test_dataset and effective_test_count > 0 and
    test_attentions is not None and test_attentions.shape[0] > 0 and encoder is not None):

    print("\n--- Saving all 128x128 attention matrices for the test set to individual CSV files ---")
    FULL_ATTENTION_OUTPUT_DIR = "full_testset_attention_matrices"
    os.makedirs(FULL_ATTENTION_OUTPUT_DIR, exist_ok=True) # Create the output folder

    feature_h, feature_w = encoder.output_H, encoder.output_W
    L = feature_h * feature_w
    seq_len = OUTPUT_SEQ_LEN

    # Iterate through all test set samples
    for idx in tqdm(range(effective_test_count), desc=f"Saving all attention matrices to '{FULL_ATTENTION_OUTPUT_DIR}'"):
        try:
            original_filename = test_dataset.filenames[idx]
            clean_filename = os.path.basename(original_filename).replace(".xlsx", "")

            # Get the attention weights for this sample [SEQ_LEN, L]
            current_sample_attentions = test_attentions[idx]

            # Check if the attention weights shape is correct
            if current_sample_attentions.shape != (seq_len, L):
                print(f"Warning: Attention weights shape for sample {idx} is {current_sample_attentions.shape}, expected ({seq_len}, {L}). Skipping save.")
                continue

            # Iterate through each predicted parameter (i.e., each decoding step)
            for param_idx in range(seq_len):
                # Get the attention weights for the current parameter and reshape to a 2D map
                attn_map_raw = current_sample_attentions[param_idx, :].reshape((feature_h, feature_w))

                # Upsample the attention map to the original image size using bilinear interpolation
                attn_map_tensor = torch.tensor(attn_map_raw).unsqueeze(0).unsqueeze(0) # [1, 1, H', W']
                upsampled_attn_tensor = nn.functional.interpolate(
                    attn_map_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False
                )
                # Convert to a NumPy array [128, 128]
                upsampled_attn = upsampled_attn_tensor.squeeze().cpu().numpy()

                # Define the CSV filename
                csv_path = os.path.join(FULL_ATTENTION_OUTPUT_DIR, f"{clean_filename}_param{param_idx+1}_attention_128x128.csv")

                # Save to a CSV file
                np.savetxt(csv_path, upsampled_attn, delimiter=',', fmt='%.8f')

        except Exception as e:
            print(f"Error: An exception occurred while saving attention matrix for sample {idx} ({original_filename}): {e}")
            continue # Continue to the next sample

    print(f"All 128x128 attention matrices for the test set have been successfully saved to the '{FULL_ATTENTION_OUTPUT_DIR}' folder.")
else:
    print("Warning: Cannot save all 128x128 attention matrices. Reason might be: test set is empty, evaluation failed, model weights not found, or attention results are missing.")


if best_model_wts is None and NUM_EPOCHS > 0:
     print("Warning: Training finished but no best model weights were saved (perhaps validation loss never decreased). Using the model from the last epoch for testing.")
     best_model_wts = model.state_dict() # Use weights from the last epoch
elif NUM_EPOCHS == 0:
     print("Warning: NUM_EPOCHS is 0, no training was performed.")
if NUM_EPOCHS > 0 and train_losses and val_losses and learning_rates:
    print("\n--- Saving training process data to training_log.csv ---")
    try:
        # Ensure all lists have the same length
        if len(train_losses) == NUM_EPOCHS and len(val_losses) == NUM_EPOCHS and len(learning_rates) == NUM_EPOCHS:
            # Create Epoch list
            epochs = list(range(1, NUM_EPOCHS + 1))
            # Create a Pandas DataFrame
            training_log_df = pd.DataFrame({
                'Epoch': epochs,
                'Train Loss (Weighted Std)': train_losses,
                'Validation Loss (Weighted Std)': val_losses,
                'Learning Rate': learning_rates
            })
            # Save to a CSV file
            training_log_df.to_csv('training_log.csv', index=False, float_format='%.8f') # Use index=False to not save the DataFrame index
            print("Training process data has been saved to training_log.csv")
        else:
            print("Warning: Length of training history lists do not match NUM_EPOCHS, cannot save training log.")
            print(f"  len(train_losses)={len(train_losses)}, len(val_losses)={len(val_losses)}, len(learning_rates)={len(learning_rates)}, NUM_EPOCHS={NUM_EPOCHS}")

    except Exception as e:
        print(f"Error: An exception occurred while saving training process data to CSV: {e}")

# --- (New) Save training process data to CSV ---
if NUM_EPOCHS > 0:
    print("\n--- Plotting and saving loss (weighted standardized) and learning rate curves ---")
    fig, ax1 = plt.subplots(dpi=300, figsize=(12, 6))

    # Plot loss curves
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Weighted Standardized MSE Loss', color=color) # Y-axis label updated for weighted loss
    ax1.plot(range(1, NUM_EPOCHS + 1), train_losses, color=color, linestyle='-', label='Training Loss (Weighted Std)')
    ax1.plot(range(1, NUM_EPOCHS + 1), val_losses, color=color, linestyle='--', label='Validation Loss (Weighted Std)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot learning rate curve (sharing X-axis)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Learning Rate', color=color)
    ax2.plot(range(1, NUM_EPOCHS + 1), learning_rates, color=color, marker='.', linestyle=':', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log') # Use a logarithmic scale for the learning rate
    ax2.legend(loc='upper right')

    fig.tight_layout() # Adjust layout to prevent overlap
    plt.title('Training/Validation Loss (Weighted Standardized) and Learning Rate') # Title updated
    plt.savefig('loss_lr_curve_weighted_scaled.png') # Filename updated
    plt.close(fig) # Close the figure
    print("Loss (weighted standardized) and learning rate curve plot saved as loss_lr_curve_weighted_scaled.png")
else:
    print("No training was performed (NUM_EPOCHS=0), skipping plotting of loss curves.")

# --- Plot training history curves (loss is on the weighted standardized scale) ---
print("\n--- Testing on the test set with the best model ---")

# --- Visualize specified samples from the test set ---
if (test_dataset and effective_test_count > 0 and
    test_attentions is not None and test_predictions_original is not None):

    # Define the indices of samples to visualize (e.g., the first 3 from the test set)
    sample_indices_to_plot = list(range(min(3, effective_test_count))) # Visualize at most 3, not exceeding the test set size

    print(f"\n--- Starting to generate attention visualization plots for test set indices {sample_indices_to_plot} ---")

    for idx in sample_indices_to_plot:
        # Ensure the index is within a valid range (theoretically guaranteed above)
        if idx < effective_test_count and idx < len(test_attentions) and idx < len(test_predictions_original):
            try:
                # Get the original filename
                original_filename = test_dataset.filenames[idx]
                img_path_vis = os.path.join(INPUT_DATA_DIR, original_filename)

                # Load original image data (0-1 range, not standardized) for visualization
                original_image_01 = torch.from_numpy(load_and_scale_image_data_01(img_path_vis)).float()

                # Get the predicted parameters (original scale) and attention weights for this sample
                predicted_params_original_sample = test_predictions_original[idx]
                attention_weights_sample = test_attentions[idx] # [SEQ_LEN, L]

                # Generate a unique filename
                clean_filename = os.path.basename(original_filename).replace(".xlsx", "")
                save_filename = f'attention_vis_sample_{idx}_{clean_filename}.png'

                try:
                    # attention_weights_sample is a numpy array [SEQ_LEN, L]
                    # L = feature_h * feature_w
                    feature_h, feature_w = encoder.output_H, encoder.output_W
                    L = feature_h * feature_w
                    seq_len = OUTPUT_SEQ_LEN  # or attention_weights_sample.shape[0]

                    # Check if the fetched attention weights shape matches the expectation
                    if attention_weights_sample.shape == (seq_len, L):
                        # Create column names (representing flattened feature map positions, e.g., Feature_0, Feature_1, ..., Feature_63 for 8x8)
                        col_names = [f'Feature_{i}' for i in range(L)]

                        # Create row index names (representing parameter prediction steps, e.g., Param_1_Step, Param_2_Step, ...)
                        row_names = [f'Param_{i + 1}_Step' for i in range(seq_len)]

                        # Create a Pandas DataFrame
                        attn_df = pd.DataFrame(attention_weights_sample, index=row_names, columns=col_names)

                        # Define the corresponding CSV filename
                        save_csv_filename = f'attention_data_sample_{idx}_{clean_filename}.csv'

                        # Save to CSV, including row indices (as they are meaningful), and specify float format
                        attn_df.to_csv(save_csv_filename, float_format='%.6f')
                        print(f"Attention weight raw data for sample {idx} saved to: {save_csv_filename}")
                    else:
                        print(
                            f"Warning: Attention weights shape for sample {idx} is {attention_weights_sample.shape}, expected ({seq_len}, {L}). Skipping CSV save.")

                except Exception as e_csv:
                    print(f"Error: An error occurred while saving the attention weights CSV for sample index {idx}: {e_csv}")

                # Check if the image is valid
                if torch.any(original_image_01 != 0) and original_image_01.shape == (1, IMAGE_SIZE, IMAGE_SIZE):
                     # Call the plotting function
                     plot_attention(image_01=original_image_01,
                                   result_original=predicted_params_original_sample,
                                   attention=attention_weights_sample,
                                   image_size=IMAGE_SIZE,
                                   feature_map_size=(encoder.output_H, encoder.output_W), # Get H', W' from the encoder
                                   save_filename=save_filename)
                else:
                    print(f"Cannot load or image is invalid, skipping visualization for sample index {idx} (file: {original_filename})")
            except Exception as e:
                 print(f"An error occurred while generating visualization for sample index {idx} (file: {original_filename}): {e}")
        else:
            print(f"Sample index {idx} is out of the valid range or result array length, skipping visualization.")

else:
    print("Cannot generate attention plots. Reason might be: test set is empty, evaluation failed, model weights not found, or attention/prediction results were not collected.")

# --- (New) Save detailed test set prediction results and errors to CSV ---
print("\n--- Saving detailed test set prediction results and errors to CSV ---")

# Check if the required variables exist and are valid (should have been generated in the testing phase)
if ('test_dataset' in locals() and hasattr(test_dataset, 'filenames') and
    'test_targets_original' in locals() and test_targets_original is not None and
    'test_predictions_original' in locals() and test_predictions_original is not None and
    'effective_test_count' in locals() and effective_test_count > 0 and
    len(test_dataset.filenames) == effective_test_count and
    test_targets_original.shape == (effective_test_count, OUTPUT_SEQ_LEN) and
    test_predictions_original.shape == (effective_test_count, OUTPUT_SEQ_LEN)):

    try:
        # 1. Prepare the data
        filenames = test_dataset.filenames                       # Get the list of test set filenames
        targets = test_targets_original                          # Get true parameter values (original scale) [N, 7]
        predictions = test_predictions_original                  # Get predicted parameter values (original scale) [N, 7]

        # Calculate absolute error for each parameter
        errors = np.abs(predictions - targets)                   # Absolute error [N, 7]

        # Calculate the mean absolute error (MAE) for each sample
        sample_mae = np.mean(errors, axis=1)                     # MAE for each sample [N]

        # 2. Build a Pandas DataFrame
        # Initialize a dictionary to build the DataFrame
        data_to_save = {'Filename': filenames}

        # Loop to add columns for true values, predicted values, and absolute errors
        for i in range(OUTPUT_SEQ_LEN):
            param_index = i + 1
            data_to_save[f'True_Param_{param_index}'] = targets[:, i]
            data_to_save[f'Pred_Param_{param_index}'] = predictions[:, i]
            data_to_save[f'Abs_Error_{param_index}'] = errors[:, i]

        # Add a column for the overall mean absolute error of the sample
        data_to_save['Sample_MAE'] = sample_mae

        # Create the DataFrame
        results_df = pd.DataFrame(data_to_save)

        # (Optional) Reorder columns to make true, prediction, and error adjacent (more readable)
        column_order = ['Filename']
        for i in range(OUTPUT_SEQ_LEN):
            param_index = i + 1
            column_order.extend([f'True_Param_{param_index}', f'Pred_Param_{param_index}', f'Abs_Error_{param_index}'])
        column_order.append('Sample_MAE')
        results_df = results_df[column_order]

        # 3. Save to a CSV file
        csv_filename = 'test_predictions_and_errors.csv'
        # Use utf-8-sig encoding to ensure proper display in Excel, and do not save the DataFrame index
        results_df.to_csv(csv_filename, index=False, float_format='%.6f', encoding='utf-8-sig')

        print(f"Detailed comparison of test set prediction results and errors saved to: {csv_filename}")
        print(f"  - The CSV file contains {results_df.shape[0]} rows (one for each test sample) and {results_df.shape[1]} columns.")
        print(f"  - Columns include: Filename, True_Param_1, Pred_Param_1, Abs_Error_1, ..., Sample_MAE.")

    except Exception as e:
        print(f"Error: An exception occurred while saving test set results and errors to CSV: {e}")

else:
    # If necessary data is missing, print a warning
    print("Warning: Cannot generate the CSV file for test set results and errors.")
    print("  Possible reasons include:")
    print("  - The test set is empty (effective_test_count=0).")
    print("  - The test evaluation phase did not run successfully or was skipped.")
    print("  - The variables 'test_targets_original' or 'test_predictions_original' were not generated correctly or have an incorrect format.")
    # (Optional) Print some debug information
    print(f"  - Checkpoint: effective_test_count={effective_test_count if 'effective_test_count' in locals() else 'Undefined'}")
    print(f"  - Checkpoint: len(test_dataset.filenames)={len(test_dataset.filenames) if 'test_dataset' in locals() and hasattr(test_dataset, 'filenames') else 'Undefined'}")
    print(f"  - Checkpoint: test_targets_original.shape={test_targets_original.shape if 'test_targets_original' in locals() and test_targets_original is not None else 'Undefined or None'}")
    print(f"  - Checkpoint: test_predictions_original.shape={test_predictions_original.shape if 'test_predictions_original' in locals() and test_predictions_original is not None else 'Undefined or None'}")

print("\n--- Script execution finished ---")