import torch
import torch.nn.functional as F


def extract_local_from_global(global_data, metadata, kernel_size=33):
    """
    Extract local patch from global data based on metadata
    """
    batch_size = metadata.size(0)
    extracted_patches = []

    for i in range(batch_size):
        centerx, centery, xbegain, ybegain, xnum, ynum, id_val = metadata[i].tolist()

        # Calculate local region boundaries
        minx = int(centerx - xbegain) - kernel_size // 2
        maxx = int(centerx - xbegain) + kernel_size // 2 + 1
        miny = int(centery - ybegain) - kernel_size // 2
        maxy = int(centery - ybegain) + kernel_size // 2 + 1

        # Ensure boundaries are within image
        minx = max(0, min(minx, global_data[i].shape[1] - 1))
        maxx = max(0, min(maxx, global_data[i].shape[1]))
        miny = max(0, min(miny, global_data[i].shape[2] - 1))
        maxy = max(0, min(maxy, global_data[i].shape[2]))

        # Extract patch
        patch = global_data[i, :, minx:maxx, miny:maxy]

        # Ensure patch is kernel_size x kernel_size
        if patch.shape[1] != kernel_size or patch.shape[2] != kernel_size:
            patch = F.interpolate(
                patch.unsqueeze(0),
                size=(kernel_size, kernel_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        extracted_patches.append(patch)

    return torch.stack(extracted_patches)


def prepare_global_context(id_arrays, id_masks, metadata, target_size=128):
    """Resize global data to a manageable size and center it around the local patch"""
    batch_size = len(id_arrays)
    processed_arrays = []
    processed_masks = []

    for i in range(batch_size):
        # Get current array and mask
        current_array = id_arrays[i]
        current_mask = id_masks[i]

        # Extract metadata
        centerx, centery, xbegain, ybegain, xnum, ynum, id_val = metadata[i].tolist()

        # Convert to tensor if not already
        if not isinstance(current_array, torch.Tensor):
            current_array = torch.tensor(current_array, dtype=torch.float32)
        if not isinstance(current_mask, torch.Tensor):
            current_mask = torch.tensor(current_mask, dtype=torch.float32)

        # Add channel dimension if needed
        if current_array.dim() == 2:
            current_array = current_array.unsqueeze(0)
        if current_mask.dim() == 2:
            current_mask = current_mask.unsqueeze(0)

        # Resize to target size
        resized_array = F.interpolate(
            current_array.unsqueeze(0),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        resized_mask = F.interpolate(
            current_mask.unsqueeze(0), size=(target_size, target_size), mode="nearest"
        ).squeeze(0)

        processed_arrays.append(resized_array)
        processed_masks.append(resized_mask)

    return torch.stack(processed_arrays), torch.stack(processed_masks)
