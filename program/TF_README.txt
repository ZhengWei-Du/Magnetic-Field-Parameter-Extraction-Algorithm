# ==============================================================================
#
#                            TF_final     Instructions (README)
#
# ==============================================================================
#
# **Functionality:**
# This script implements a Seq2Seq model based on a CNN encoder and a GRU decoder
# with a multi-head attention mechanism. It is used to predict a sequence of 7 parameters
# based on an input 128x128 image (represented by an Excel file).
# The script includes functionalities for data loading, preprocessing (image scaling,
# target value standardization), model definition, training, evaluation,
# loss/learning rate visualization, and attention map visualization.
#
# **Dependencies:**
# - Python 3.x
# - PyTorch (torch, torchvision)
# - Scikit-learn (sklearn)
# - Pandas (pandas)
# - Numpy (numpy)
# - Matplotlib (matplotlib)
# - Tqdm (tqdm)
# - openpyxl (for reading .xlsx files)
# It is recommended to install them using `pip install torch torchvision torchaudio scikit-learn pandas numpy matplotlib tqdm openpyxl`.
#
# **Directory Structure:**
# Assuming the script is named `train_predict.py`:
# /path/to/your/project/
# |-- train_predict.py
# |-- 449-dataset/         <-- Directory containing image .xlsx files (specified by INPUT_DATA_DIR)
# |   |-- image1.xlsx
# |   |-- image2.xlsx
# |   |-- ...
# |-- tag.xlsx             <-- Excel file containing image filenames and their corresponding 7 parameters (specified by OUTPUT_PARAM_FILE)
#
# **Parameter File (tag.xlsx) Format:**
# - First column: Image filename (e.g., "image1.xlsx")
# - Second to eighth columns: The corresponding 7 target parameter values
# - (Other columns will be ignored)
#
# **Configuration Parameters (In-script):**
# - `INPUT_DATA_DIR`: Path to the directory containing image .xlsx files.
# - `OUTPUT_PARAM_FILE`: Path to the parameter .xlsx file.
# - `IMAGE_SIZE`: The size of the image (should be 128).
# - `OUTPUT_SEQ_LEN`: The length of the parameter sequence to predict (should be 7).
# - `BATCH_SIZE`: Batch size for training and evaluation.
# - `NUM_EPOCHS`: Total number of training epochs (increased to 200 for better data utilization).
# - `INITIAL_LEARNING_RATE`: Initial learning rate.
# - `CNN_OUTPUT_CHANNELS`: Number of output channels for the CNN encoder.
# - `EMBED_DIM`: Dimension of the step embedding in the decoder.
# - `HIDDEN_DIM`: Hidden dimension of the GRU decoder (must be divisible by NUM_HEADS).
# - `NUM_DECODER_LAYERS`: Number of layers in the GRU decoder (increased to 2 to enhance model capacity).
# - `NUM_HEADS`: Number of heads in the multi-head attention mechanism (new).
# - `DROPOUT`: Dropout rate.
# - `CLIP_GRAD`: Gradient clipping threshold.
# - `LR_SCHEDULER_FACTOR`, `LR_SCHEDULER_PATIENCE`, `LR_SCHEDULER_MIN_LR`: Parameters for the learning rate scheduler.
# - `WEIGHTED_LOSS_FACTORS`: (New) Used for weighted loss to improve the prediction accuracy of specific parameters. The list length must be equal to `OUTPUT_SEQ_LEN`.
#                            For example, `[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0]` means the loss weights for the 4th and 7th parameters are 2.
#
# **Running the Script:**
# ```bash
# python train_predict.py
# ```
#
# **Output:**
# - **Console Output:** Displays device information, data loading/processing progress, training/validation loss, learning rate changes, and test set evaluation results (standardized MSE, original-scale MAE/MSE).
# - **best_model.pth:** Saves the model weights with the lowest validation loss.
# - **loss_lr_curve_scaled.png:** A plot showing the training/validation loss (on the standardized scale) and learning rate curves over epochs.
# - **attention_vis_sample_*.png:** Attention visualization plots for specified samples from the test set (including the original image, predicted parameters, and attention overlay).
#
# **Notes:**
# - Ensure the paths for `INPUT_DATA_DIR` and `OUTPUT_PARAM_FILE` are correct.
# - Ensure the format of the parameter file `tag.xlsx` is correct and that its filenames can be matched with files in the image directory.
# - Image files should be 128x128. Anomalous files will be skipped.
# - Target parameter values should not contain NaN or Inf.
# - `HIDDEN_DIM` must be divisible by `NUM_HEADS`.
# - Adjust `WEIGHTED_LOSS_FACTORS` to focus on improving the prediction of specific parameters.
# - If the training/validation loss remains high or does not converge, try adjusting hyperparameters (e.g., learning rate, network architecture, Dropout, `NUM_EPOCHS`, etc.).
#
# ==============================================================================