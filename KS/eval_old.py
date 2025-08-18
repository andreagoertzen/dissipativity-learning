import numpy as np
import torch
import model  # Your model.py file
import os
# Assuming your Dataset and data loading function are in a 'utils.py' file
from utils import load_multi_traj_data, run_model_visualization

def main():
    """
    Main function to load a trained model and run evaluation and visualization.
    """
    # --- 1. Configuration ---
    # Set the device to run the model on (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set the tag to match your training run
    tag = '0814_1657_project'
    
    # --- IMPORTANT: Set these paths correctly ---
    # Path to the saved model file
    model_path = f'{tag}/model_epoch_best.pt' 
    # Path to the dataset
    data_path = 'Data/KS_data_batched_l100.53_grid512_M8_T500.0_dt0.01_amp5.0/data.npz'
    # Directory to save the output figures
    figs_folder = f'eval_results'
    figs_folder = os.path.join(tag, figs_folder)

    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder)

    # --- 2. Load Data ---
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    
    # Use the same function to split data, then select the validation set
    _, val_dataset = load_multi_traj_data(data)
    print(f"Loaded validation dataset with {len(val_dataset)} samples.")

    # Prepare the full validation set for visualization
    # The model expects a tuple of (branch_inputs, trunk_input)
    x_val = (val_dataset.branch_inputs.to(device), val_dataset.trunk_input.to(device))
    y_val = val_dataset.targets.to(device)

    # --- 3. Initialize and Load Model ---
    # Infer model dimensions from the dataset
    m = val_dataset.branch_inputs.shape[1]  # Trajectory dimension
    n = 1
    
    # We need to know if the saved model was trained with the projection layer
    # Set this to match the training configuration of the saved model
    project = True 
    
    print(f"Initializing model with m={m}, n={n}, project={project}...")
    # Create an instance of the model architecture
    eval_model = model.DeepONet(m, n, project=project).to(device)
    
    # Load the saved weights into the model instance
    print(f"Loading saved model weights from {model_path}...")
    eval_model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set the model to evaluation mode
    # This is important as it disables layers like Dropout or BatchNorm if they exist
    eval_model.eval()

    # --- 4. Run Visualization ---
    print(f"Running model visualization... Figures will be saved in '{figs_folder}'")
    with torch.no_grad(): # Disable gradient calculations for inference
        run_model_visualization(
            model=eval_model,
            x_test=x_val,
            y_test=y_val,
            s=m,  # s is the spatial dimension for plotting
            device=device,
            figs_dir=figs_folder
        )
    
    print("\nEvaluation complete.")

if __name__ == '__main__':
    main()