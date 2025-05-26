import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models import CompletionNetwork
from DemDataset import custom_collate_fn  # Import your custom collate function
import torchvision.transforms as transforms
from tqdm import tqdm
import time

def load_model(model_path, device):
    """
    Load the trained completion network model.
    
    Args:
        model_path: Path to the saved model weights
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Initialize the model with the same parameters as during training
    model = CompletionNetwork(input_channels=1).to(device)
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # Set to evaluation mode
    model.eval()
    
    return model

def prepare_data(local_inputs, local_masks, global_inputs, global_masks, device):
    """
    Prepare data for inference by moving tensors to the specified device.
    
    Args:
        local_inputs: Local input tensors
        local_masks: Local mask tensors
        global_inputs: Global input tensors
        global_masks: Global mask tensors
        device: Device to move tensors to
        
    Returns:
        Tuple of tensors moved to the specified device
    """
    # Move tensors to device with highest precision
    local_inputs = local_inputs.to(device, dtype=torch.float32)
    local_masks = local_masks.to(device, dtype=torch.int)
    global_inputs = global_inputs.to(device, dtype=torch.float32)
    global_masks = global_masks.to(device, dtype=torch.int)
    
    return local_inputs, local_masks, global_inputs, global_masks

def merge_local_to_global(local_data, global_data, global_mask, metadata, kernel_size=33):
    """
    Merge local patch data into global data based on metadata.
    
    Args:
        local_data: The local patch data
        global_data: The global data
        global_mask: The global mask
        metadata: Metadata containing position information
        kernel_size: The size of the local patch
        
    Returns:
        Tuple of (merged_data, merged_mask)
    """
    merged_data = global_data.clone()
    merged_mask = global_mask.clone()
    
    pos = []
    for i in range(len(metadata)):
        centerx = metadata[i][0]
        centery = metadata[i][1]
        xbegain = metadata[i][2]
        ybegain = metadata[i][3]
        
        # Calculate the top-left corner of the kernel_size x kernel_size window
        minx = int(centerx - xbegain) - kernel_size // 2
        maxx = int(centerx - xbegain) + kernel_size // 2 + 1
        miny = int(centery - ybegain) - kernel_size // 2
        maxy = int(centery - ybegain) + kernel_size // 2 + 1
        
        # Embed the local data into the global data
        merged_data[i, :, minx:maxx, miny:maxy] = local_data[i]
        merged_mask[i, :, minx:maxx, miny:maxy] = 1
        pos.append([minx, maxx, miny, maxy])
    
    return merged_data, merged_mask, pos

def visualize_dem(tensor, title="DEM Visualization", save_path=None):
    """
    Visualize a DEM tensor with a colormap.
    
    Args:
        tensor: The tensor to visualize
        title: Title for the visualization
        save_path: Path to save the visualization image (optional)
    """
    # Convert tensor to numpy array
    if tensor.dim() > 2:
        # If tensor has more than 2 dimensions, get the first channel
        arr = tensor[0].cpu().detach().numpy()
    else:
        arr = tensor.cpu().detach().numpy()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Use colormap for visualization
    plt.imshow(arr, cmap='jet')
    plt.colorbar(label='Elevation')
    plt.title(title)
    
    # Save if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close()
    
    return arr

def perform_dem_completion(model, input_batch, device, output_dir):
    """
    Perform DEM completion on a batch of data.
    
    Args:
        model: The trained completion network model
        input_batch: A tuple containing the input batch data
        device: Device to run inference on
        output_dir: Directory to save output visualizations
        
    Returns:
        Tuple of completed DEMs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Unpack and prepare batch data
    (local_inputs, local_masks, local_targets, 
     global_inputs, global_masks, global_targets, metadata) = input_batch
    
    local_inputs, local_masks, global_inputs, global_masks = prepare_data(
        local_inputs, local_masks, global_inputs, global_masks, device
    )
    
    # Perform inference
    with torch.no_grad():
        begain = time.time()
        outputs = model(local_inputs, local_masks, global_inputs, global_masks)
        end = time.time()
        print(f"Inference time: {end - begain:.2f} seconds")
        
        # Merge local output to global
        completed_dems, completed_masks, _ = merge_local_to_global(
            outputs, global_inputs, global_masks, metadata
        )
    
    # Visualize and save results
    results = []
    for i in range(len(completed_dems)):
        # Original masked input
        visualize_dem(
            local_inputs[i], 
            title=f"Input DEM with Mask (Sample {i})",
            save_path=os.path.join(output_dir, f"input_dem_{i}.png")
        )
        
        # Mask visualization
        visualize_dem(
            local_targets[i], 
            title=f"Mask (Sample {i})",
            save_path=os.path.join(output_dir, f"local_target_{i}.png")
        )
        
        # Local output patch
        visualize_dem(
            outputs[i], 
            title=f"Completed Local Patch (Sample {i})",
            save_path=os.path.join(output_dir, f"local_output_{i}.png")
        )
        
        # Completed global DEM
        completed_array = visualize_dem(
            completed_dems[i], 
            title=f"Completed DEM (Sample {i})",
            save_path=os.path.join(output_dir, f"completed_dem_{i}.png")
        )
        
        # Ground truth if available
        if global_targets is not None:
            global_targets_tensor = global_targets.to(device)
            visualize_dem(
                global_targets_tensor[i], 
                title=f"Ground Truth DEM (Sample {i})",
                save_path=os.path.join(output_dir, f"ground_truth_{i}.png")
            )
        
        results.append(completed_array)
    
    return completed_dems, results

def main():
    # Configuration
    model_path = r"E:\KingCrimson Dataset\Simulate\data0\results13\phase_3\model_cn_best"  # Path to your best model weights
    result_dir = r"E:\KingCrimson Dataset\Simulate\data0\results13\predict1"  # Directory to save results
    
    # Test data paths
    json_dir = r"E:\KingCrimson Dataset\Simulate\data0\testjson"
    array_dir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\array"
    mask_dir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\mask"
    target_dir = (
        r"E:\KingCrimson Dataset\Simulate\data0\groundtruthstatus\statusarray" # 全局target
    )  # Optional, set to None if not available
    
    # Hardware setup
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    
    # Load test dataset
    from DemDataset import DemDataset  # Import your dataset class
    
    print("Loading test data...")
    test_dataset = DemDataset(json_dir, array_dir, mask_dir, target_dir)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=4,  # Use smaller batch size for inference
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    
    # Run inference
    print("Running inference...")
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        batch_output_dir = os.path.join(result_dir, f"batch_{batch_idx}")
        completed_dems, _ = perform_dem_completion(model, batch, device, batch_output_dir)
        
        # You can save the tensor outputs here if needed
        # torch.save(completed_dems.cpu(), os.path.join(batch_output_dir, "completed_dems.pt"))
    
    print(f"Inference complete. Results saved to {result_dir}")

def process_single_dem(model, dem_file, mask_file, device, output_dir):
    """
    Process a single DEM file for completion.
    
    Args:
        model: The trained completion network model
        dem_file: Path to the DEM file
        mask_file: Path to the mask file
        device: Device to run inference on
        output_dir: Directory to save output visualization
        
    Returns:
        Completed DEM as numpy array
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load DEM and mask
    # Adapt this based on your file format (numpy, tiff, etc.)
    input_dem = np.load(dem_file)
    input_mask = np.load(mask_file)
    
    # Convert to tensors and add batch and channel dimensions
    input_dem_tensor = torch.from_numpy(input_dem).float().unsqueeze(0).unsqueeze(0)
    input_mask_tensor = torch.from_numpy(input_mask).int().unsqueeze(0).unsqueeze(0)
    
    # Create dummy metadata for a single global patch
    # Adjust based on your format
    metadata = [(input_dem.shape[0]//2, input_dem.shape[1]//2, 0, 0)]
    
    # Extract local patch for processing
    kernel_size = 33  # Same as training
    centerx, centery = metadata[0][0], metadata[0][1]
    minx = centerx - kernel_size // 2
    maxx = centerx + kernel_size // 2 + 1
    miny = centery - kernel_size // 2
    maxy = centery + kernel_size // 2 + 1
    
    # Extract local patch
    local_dem = input_dem_tensor[:, :, minx:maxx, miny:maxy]
    local_mask = input_mask_tensor[:, :, minx:maxx, miny:maxy]
    
    # Move to device
    local_dem = local_dem.to(device)
    local_mask = local_mask.to(device)
    global_dem = input_dem_tensor.to(device)
    global_mask = input_mask_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        output_local = model(local_dem, local_mask, global_dem, global_mask)
        
        # Merge local output back to global
        completed_dem, _, _ = merge_local_to_global(
            output_local, global_dem, global_mask, metadata
        )
    
    # Visualize results
    visualize_dem(
        input_dem_tensor, 
        title="Input DEM with Mask",
        save_path=os.path.join(output_dir, "input_dem.png")
    )
    
    visualize_dem(
        input_mask_tensor, 
        title="Mask",
        save_path=os.path.join(output_dir, "mask.png")
    )
    
    completed_array = visualize_dem(
        completed_dem[0], 
        title="Completed DEM",
        save_path=os.path.join(output_dir, "completed_dem.png")
    )
    
    # Save the completed DEM as numpy array
    np.save(os.path.join(output_dir, "completed_dem.npy"), completed_array)
    
    return completed_array

if __name__ == "__main__":
    main()
    
    # Alternatively, to process a single DEM:
    """
    model_path = "path/to/your/best/model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    
    dem_file = "path/to/single/dem.npy"
    mask_file = "path/to/single/mask.npy"
    output_dir = "path/to/single/result"
    
    completed_dem = process_single_dem(model, dem_file, mask_file, device, output_dir)
    """