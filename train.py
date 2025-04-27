import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adadelta, Adam
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
from PIL import Image

# Import your models
from models import CompletionNetwork, ContextDiscriminator

# Import your custom dataset
from DemDataset import DemDataset

# Import the custom_collate_fn function from your dataset module
from DemDataset import custom_collate_fn


def train():
    # =================================================
    # Configuration settings (replace with your settings)
    # =================================================
    json_dir = r"E:\KingCrimson Dataset\Simulate\data0\json"  # 局部json
    array_dir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\array"  # 全局array
    mask_dir = r"E:\KingCrimson Dataset\Simulate\data0\arraynmask\mask"  # 全局mask
    target_dir = (
        r"E:\KingCrimson Dataset\Simulate\data0\groundtruthstatus\statusarray" # 全局target
    )
    result_dir = r"E:\KingCrimson Dataset\Simulate\data0\results"  # 结果保存路径

    # Training parameters
    steps_1 = 100  # 90000  # Phase 1 training steps
    steps_2 = 100  # 10000  # Phase 2 training steps
    steps_3 = 100  # 400000  # Phase 3 training steps

    snaperiod_1 = 20  # 10000  # How often to save snapshots in phase 1
    snaperiod_2 = 20  # 2000  # How often to save snapshots in phase 2
    snaperiod_3 = 20  # 10000  # How often to save snapshots in phase 3

    batch_size = 16  # Batch size
    num_test_completions = 8  # Number of test completions to generate
    alpha = 4e-4  # Alpha parameter for loss weighting
    validation_split = 0.2  # Percentage of data to use for validation

    # Hardware setup
    if not torch.cuda.is_available():
        raise Exception("At least one GPU must be available.")
    device = torch.device("cuda:0")

    # =================================================
    # Preparation
    # =================================================
    # Create result directories
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for phase in ["phase_1", "phase_2", "phase_3", "validation"]:
        if not os.path.exists(os.path.join(result_dir, phase)):
            os.makedirs(os.path.join(result_dir, phase))

    # Load dataset
    full_dataset = DemDataset(json_dir, array_dir, mask_dir, target_dir)

    # Split dataset into train and validation
    dataset_size = len(full_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # For reproducibility
    )

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    """# Calculate mean pixel value of the dataset
    mpv = np.zeros(shape=(3,))
    print("Computing mean pixel value of training dataset...")
    for batch in tqdm(train_loader):
        # Unpack batch data
        (
            inputs,
            input_masks,
            input_targets,
            id_arrays,
            id_masks,
            id_targets,
            metadata,
        ) = batch

        # Process each item in the batch
        for i in range(len(inputs)):
            # Get the mean of input_targets which are the ground truth local patches
            mpv += torch.mean(input_targets[i], dim=[0, 1]).cpu().numpy()

    mpv /= len(train_loader)
    mpv = torch.tensor(mpv.reshape(1, 1, 1), dtype=torch.float32).to(device)"""
    alpha = torch.tensor(alpha, dtype=torch.float32).to(device)

    # =================================================
    # Setup Models
    # =================================================
    # Initialize the Completion Network - adjust input channels as needed
    model_cn = CompletionNetwork(input_channels=1).to(
        device
    )  # 1 for elevation + 1 for mask

    # Initialize the Context Discriminator
    model_cd = ContextDiscriminator(
        local_input_channels=1,  # For elevation data
        local_input_size=33,  # The local patch size from your dataset
        global_input_channels=1,  # For elevation data
        global_input_size=600,  # The global image size from your dataset
    ).to(device)

    # Setup optimizers
    opt_cn = Adadelta(model_cn.parameters())
    opt_cd = Adadelta(model_cd.parameters())

    # Loss functions
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    # Helper function to prepare a batch for model input
    def prepare_batch_for_model(batch_data, device):
        (
            inputs,
            input_masks,
            input_targets,
            id_arrays,
            id_masks,
            id_targets,
            metadata,
        ) = batch_data

        batch_local_inputs = []
        batch_local_masks = []
        batch_local_targets = []
        batch_global_inputs = []
        batch_global_masks = []
        batch_global_targets = []

        for i in range(len(inputs)):
            # Prepare local inputs (with mask info)
            local_input = inputs[i].unsqueeze(0).to(device)  # Add channel dimension
            local_mask = input_masks[i].unsqueeze(0).to(device)  # Add channel dimension
            local_target = (
                input_targets[i].unsqueeze(0).to(device)
            )  # Add channel dimension

            # Prepare global inputs
            global_input = id_arrays[i].unsqueeze(0).to(device)  # Add channel dimension
            global_mask = id_masks[i].unsqueeze(0).to(device)  # Add channel dimension
            global_target = (
                id_targets[i].unsqueeze(0).to(device)
            )  # Add channel dimension

            # Combine input and mask for model input
            local_combined = torch.cat([local_input, local_mask], dim=0)
            global_combined = torch.cat([global_input, global_mask], dim=0)

            batch_local_inputs.append(local_combined)
            batch_local_masks.append(local_mask)
            batch_local_targets.append(local_target)
            batch_global_inputs.append(global_combined)
            batch_global_masks.append(global_mask)
            batch_global_targets.append(global_target)

        # Stack into batches
        batch_local_inputs = torch.stack(batch_local_inputs).to(device)
        batch_local_masks = torch.stack(batch_local_masks).to(device)
        batch_local_targets = torch.stack(batch_local_targets).to(device)
        batch_global_inputs = torch.stack(batch_global_inputs).to(device)
        batch_global_masks = torch.stack(batch_global_masks).to(device)
        batch_global_targets = torch.stack(batch_global_targets).to(device)

        return (
            batch_local_inputs,
            batch_local_masks,
            batch_local_targets,
            batch_global_inputs,
            batch_global_masks,
            batch_global_targets,
        )

    def prepare_batch_for_model2(batch_data, device):
        (
            inputs,
            input_masks,
            input_targets,
            id_arrays,
            id_masks,
            id_targets,
            metadata,
        ) = batch_data

        batch_local_inputs = []
        batch_local_masks = []
        batch_local_targets = []
        batch_global_inputs = []
        batch_global_masks = []
        batch_global_targets = []

        for i in range(len(inputs)):
            # Prepare local inputs (with mask info)
            local_input = inputs[i].unsqueeze(0).to(device)  # Add channel dimension
            local_mask = input_masks[i].unsqueeze(0).to(device)  # Add channel dimension
            local_target = (
                input_targets[i].unsqueeze(0).to(device)
            )  # Add channel dimension

            # Prepare global inputs
            global_input = id_arrays[i].unsqueeze(0).to(device)  # Add channel dimension
            global_mask = id_masks[i].unsqueeze(0).to(device)  # Add channel dimension
            global_target = (
                id_targets[i].unsqueeze(0).to(device)
            )  # Add channel dimension

            # Combine input and mask for model input
            # local_combined = torch.cat([local_input, local_mask], dim=0)
            # global_combined = torch.cat([global_input, global_mask], dim=0)

            batch_local_inputs.append(local_input)
            batch_local_masks.append(local_mask)
            batch_local_targets.append(local_target)
            batch_global_inputs.append(global_input)
            batch_global_masks.append(global_mask)
            batch_global_targets.append(global_target)

        # Stack into batches
        batch_local_inputs = torch.stack(batch_local_inputs).to(device)
        batch_local_masks = torch.stack(batch_local_masks).to(device)
        batch_local_targets = torch.stack(batch_local_targets).to(device)
        batch_global_inputs = torch.stack(batch_global_inputs).to(device)
        batch_global_masks = torch.stack(batch_global_masks).to(device)
        batch_global_targets = torch.stack(batch_global_targets).to(device)

        return (
            batch_local_inputs,
            batch_local_masks,
            batch_local_targets,
            batch_global_inputs,
            batch_global_masks,
            batch_global_targets,
            metadata,
        )

    # Helper function to evaluate model on validation set
    def validate(model_cn, val_loader, device):
        model_cn.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                """# Prepare batch data
                (
                    batch_local_inputs,
                    batch_local_masks,
                    batch_local_targets,
                    batch_global_inputs,
                    batch_global_masks,
                    batch_global_targets,
                    meatadata,
                ) = prepare_batch_for_model2(batch, device)"""
                (
                    batch_local_inputs,
                    batch_local_masks,
                    batch_local_targets,
                    batch_global_inputs,
                    batch_global_masks,
                    batch_global_targets,
                    meatadata,
                ) = batch
                batch_local_inputs = batch_local_inputs.to(device)
                batch_local_masks = batch_local_masks.to(device)
                batch_local_targets = batch_local_targets.to(device)
                batch_global_inputs = batch_global_inputs.to(device)
                batch_global_masks = batch_global_masks.to(device)

                # Forward pass
                outputs = model_cn(
                    batch_local_inputs,
                    batch_local_masks,
                    batch_global_inputs,
                    batch_global_masks,
                )

                # Compute loss
                loss = mse_loss(outputs, batch_local_targets)
                val_loss += loss.item()

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        model_cn.train()

        return avg_val_loss

    def completion_network_loss(input, output, mask):
        """print(
            f"shape of input: {input.shape}, shape of output: {output.shape}, shape of mask: {mask.shape}"
        )"""
        return mse_loss(output * mask, input * mask)

    def merge_local_to_global(
        local_data, global_data, global_mask, metadata, kernel_size=33
    ):

        for i in range(len(metadata)):
            centerx = metadata[i][0]
            centery = metadata[i][1]
            xbegain = metadata[i][2]
            ybegain = metadata[i][3]

            # Calculate the top-left corner of the 33x33 window
            [minx, maxx] = [
                int(centerx - xbegain) - kernel_size // 2,
                int(centerx - xbegain) + kernel_size // 2 + 1,
            ]
            [miny, maxy] = [
                int(centery - ybegain) - kernel_size // 2,
                int(centery - ybegain) + kernel_size // 2 + 1,
            ]

            # Embed the local input into the global input
            merged_data = global_data.clone()
            merged_mask = global_mask.clone()
            merged_data[i, :, minx:maxx, miny:maxy] = local_data[i]
            merged_mask[i, :, minx:maxx, miny:maxy] = 1
        return merged_data, merged_mask

    # =================================================
    # Training Phase 1: Train Completion Network only
    # =================================================
    print("Starting Phase 1 training...")
    model_cn.train()
    pbar = tqdm(total=steps_1)

    step = 0
    best_val_loss = float("inf")

    while step < steps_1:
        for batch in train_loader:
            # Prepare batch data
            (
                batch_local_inputs,
                batch_local_masks,
                batch_local_targets,
                batch_global_inputs,
                batch_global_masks,
                batch_global_targets,
                metadata,
            ) = batch  # = prepare_batch_for_model2(batch, device)
            # 将所有张量移动到 GPU
            batch_local_inputs = batch_local_inputs.to(device)
            batch_local_masks = batch_local_masks.to(device)
            batch_local_targets = batch_local_targets.to(device)
            batch_global_inputs = batch_global_inputs.to(device)
            batch_global_masks = batch_global_masks.to(device)
            batch_global_targets = batch_global_targets.to(device)

            # Forward pass
            outputs = model_cn(
                batch_local_inputs,
                batch_local_masks,
                batch_global_inputs,
                batch_global_masks,
            )

            # Calculate loss
            loss = completion_network_loss(
                batch_local_targets, outputs, batch_local_masks
            )

            # Backward and optimize
            loss.backward()
            opt_cn.step()
            opt_cn.zero_grad()

            step += 1
            pbar.set_description(f"Phase 1 | train loss: {loss.item():.5f}")
            pbar.update(1)

            # Testing and saving snapshots
            if step % snaperiod_1 == 0:
                # Run validation
                val_loss = validate(model_cn, val_loader, device)
                print(f"Step {step}, Validation Loss: {val_loss:.5f}")

                # Save if best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(
                        result_dir, "phase_1", "model_cn_best"
                    )
                    torch.save(model_cn.state_dict(), best_model_path)

                # Generate some test completions
                model_cn.eval()
                with torch.no_grad():
                    # Get a batch from validation set
                    val_batch = next(iter(val_loader))

                    # Prepare validation batch
                    (
                        val_local_inputs,
                        val_local_masks,
                        val_local_targets,
                        val_global_inputs,
                        val_global_masks,
                        val_global_targets,
                        metadata,
                    ) = val_batch  # prepare_batch_for_model2(val_batch, device)
                    val_local_inputs = val_local_inputs.to(device)
                    val_local_masks = val_local_masks.to(device)
                    val_local_targets = val_local_targets.to(device)
                    val_global_inputs = val_global_inputs.to(device)
                    val_global_masks = val_global_masks.to(device)
                    val_global_targets = val_global_targets.to(device)

                    # Generate completions
                    test_output = model_cn(
                        val_local_inputs,
                        val_local_masks,
                        val_global_inputs,
                        val_global_masks,
                    )

                    completed, completed_mask = merge_local_to_global(
                        test_output, val_global_inputs, val_global_masks, metadata
                    )

                    # Save images - adjust channel handling as needed
                    # Here we're assuming grayscale elevation maps
                    """imgs = torch.cat(
                        (
                            val_local_targets[:num_test_completions].cpu(),
                            input_elevation[:num_test_completions].cpu(),
                            test_output[:num_test_completions].cpu(),
                        ),
                        dim=0,
                    )

                    imgpath = os.path.join(result_dir, "phase_1", f"step{step}.png")
                    save_image(imgs, imgpath, nrow=num_test_completions)"""

                    # Save model
                    model_path = os.path.join(
                        result_dir, "phase_1", f"model_cn_step{step}"
                    )
                    torch.save(model_cn.state_dict(), model_path)

                model_cn.train()

            if step >= steps_1:
                break

    pbar.close()

    # =================================================
    # Training Phase 2: Train Context Discriminator only
    # =================================================
    print("Starting Phase 2 training...")
    pbar = tqdm(total=steps_2)
    step = 0
    best_val_loss_cd = float("inf")

    while step < steps_2:
        for batch in train_loader:
            # Prepare batch data
            (
                batch_local_inputs,
                batch_local_masks,
                batch_local_targets,
                batch_global_inputs,
                batch_global_masks,
                batch_global_targets,
                metadata,
            ) = batch  # prepare_batch_for_model2(batch, device)
            batch_local_inputs = batch_local_inputs.to(device)
            batch_local_masks = batch_local_masks.to(device)
            batch_local_targets = batch_local_targets.to(device)
            batch_global_inputs = batch_global_inputs.to(device)
            batch_global_masks = batch_global_masks.to(device)
            batch_global_targets = batch_global_targets.to(device)

            # Extract elevation channels for discriminator
            # local_input_elev = batch_local_inputs[:, 0:1, :, :]
            # global_input_elev = batch_global_inputs[:, 0:1, :, :]

            # Generate fake samples using completion network
            with torch.no_grad():
                fake_outputs = model_cn(
                    batch_local_inputs,
                    batch_local_masks,
                    batch_global_inputs,
                    batch_global_masks,
                )

            # Embed batch_local_inputs into batch_global_inputs
            fake_global_embedded, fake_global_mask_embedded = merge_local_to_global(
                fake_outputs, batch_global_inputs, batch_global_masks, metadata
            )

            # global的mask如何处理？
            # Train discriminator with fake samples
            fake_labels = torch.zeros(fake_outputs.size(0), 1).to(device)
            fake_predictions = model_cd(
                fake_outputs,
                batch_local_masks,
                batch_global_targets,
                fake_global_mask_embedded,
            )
            loss_fake = bce_loss(fake_predictions, fake_labels)

            real_global_embedded, real_global_mask_embedded = merge_local_to_global(
                batch_local_inputs, batch_global_inputs, batch_global_masks, metadata
            )

            # Train discriminator with real samples
            real_labels = torch.ones(batch_local_targets.size(0), 1).to(device)
            real_predictions = model_cd(
                batch_local_targets,
                batch_local_masks,
                batch_global_targets,
                real_global_mask_embedded,
            )
            loss_real = bce_loss(real_predictions, real_labels)

            # Combined loss
            loss = (loss_fake + loss_real) / 2.0

            # Backward and optimize
            loss.backward()
            opt_cd.step()
            opt_cd.zero_grad()

            step += 1
            pbar.set_description(f"Phase 2 | train loss: {loss.item():.5f}")
            pbar.update(1)

            # Testing and saving snapshots
            if step % snaperiod_2 == 0:
                # Calculate discriminator accuracy
                model_cd.eval()
                correct = 0
                total = 0

                with torch.no_grad():
                    # Check on validation set
                    for val_batch in val_loader:
                        # Prepare validation batch
                        (
                            val_local_inputs,
                            val_local_masks,
                            val_local_targets,
                            val_global_inputs,
                            val_global_masks,
                            val_global_targets,
                            metadata,
                        ) = batch  # prepare_batch_for_model(val_batch, device)
                        val_local_inputs = val_local_inputs.to(device)
                        val_local_masks = val_local_masks.to(device)
                        val_local_targets = val_local_targets.to(device)
                        val_global_inputs = val_global_inputs.to(device)
                        val_global_masks = val_global_masks.to(device)
                        val_global_targets = val_global_targets.to(device)

                        # Extract elevation channels
                        # val_local_elev = val_local_inputs[:, 0:1, :, :]
                        # val_global_elev = val_global_inputs[:, 0:1, :, :]

                        # Generate fake samples
                        fake_outputs = model_cn(
                            val_local_inputs,
                            val_local_masks,
                            val_global_inputs,
                            val_global_masks,
                        )

                        # Embed batch_local_inputs into batch_global_inputs
                        val_fake_global_embedded, val_fake_global_mask_embedded = (
                            merge_local_to_global(
                                fake_outputs,
                                val_global_inputs,
                                val_global_masks,
                                metadata,
                            )
                        )

                        # Get discriminator predictions for fake samples
                        fake_preds = model_cd(
                            fake_outputs,
                            val_local_masks,
                            val_global_targets,
                            val_fake_global_mask_embedded,
                        )

                        # Embed batch_local_inputs into batch_global_inputs
                        val_real_global_embedded, val_real_global_mask_embedded = (
                            merge_local_to_global(
                                val_local_inputs,
                                val_global_inputs,
                                val_global_masks,
                                metadata,
                            )
                        )

                        # Get discriminator predictions for real samples
                        real_preds = model_cd(
                            val_local_targets,
                            val_local_masks,
                            val_global_targets,
                            val_real_global_mask_embedded,
                        )

                        # 这里什么意思？
                        # Calculate accuracy
                        total += fake_preds.size(0) * 2  # both real and fake samples
                        correct += (
                            (fake_preds < 0.5).sum() + (real_preds >= 0.5).sum()
                        ).item()

                accuracy = correct / total if total > 0 else 0
                print(f"Step {step}, Discriminator Accuracy: {accuracy:.4f}")

                # Save model
                model_path = os.path.join(result_dir, "phase_2", f"model_cd_step{step}")
                torch.save(model_cd.state_dict(), model_path)

                # Save best model if accuracy improved
                # 似乎应该比较生成max
                if accuracy > 0.75:  # Using accuracy threshold of 75%
                    best_model_path = os.path.join(
                        result_dir, "phase_2", "model_cd_best"
                    )
                    torch.save(model_cd.state_dict(), best_model_path)

                model_cd.train()

            if step >= steps_2:
                break

    pbar.close()

    # =================================================
    # Training Phase 3: Joint training of both networks
    # =================================================
    print("Starting Phase 3 training...")
    pbar = tqdm(total=steps_3)
    step = 0
    best_val_loss_joint = float("inf")

    while step < steps_3:
        for batch in train_loader:
            # Prepare batch data
            (
                batch_local_inputs,
                batch_local_masks,
                batch_local_targets,
                batch_global_inputs,
                batch_global_masks,
                batch_global_targets,
                metadata,
            ) = batch  # prepare_batch_for_model2(batch, device)
            batch_local_inputs = batch_local_inputs.to(device)
            batch_local_masks = batch_local_masks.to(device)
            batch_local_targets = batch_local_targets.to(device)
            batch_global_inputs = batch_global_inputs.to(device)
            batch_global_masks = batch_global_masks.to(device)
            batch_global_targets = batch_global_targets.to(device)

            # Extract elevation channels for discriminator
            # local_input_elev = batch_local_inputs[:, 0:1, :, :]
            # global_input_elev = batch_global_inputs[:, 0:1, :, :]

            # --------------------------------------
            # 1. Train the discriminator first
            # --------------------------------------
            # Generate fake samples
            with torch.no_grad():
                fake_outputs = model_cn(
                    batch_local_inputs,
                    batch_local_masks,
                    batch_global_inputs,
                    batch_global_masks,
                )

            # Train discriminator with fake samples
            fake_labels = torch.zeros(fake_outputs.size(0), 1).to(device)
            fake_global_embedded, fake_global_mask_embedded = merge_local_to_global(
                fake_outputs,
                batch_global_inputs,
                batch_global_masks,
                metadata,
            )
            fake_predictions = model_cd(
                fake_outputs,
                batch_local_masks,
                batch_global_targets,
                fake_global_mask_embedded,
            )
            loss_cd_fake = bce_loss(fake_predictions, fake_labels)

            # Train discriminator with real samples
            real_labels = torch.ones(batch_local_targets.size(0), 1).to(device)
            real_global_embedded, real_global_mask_embedded = merge_local_to_global(
                batch_local_targets,
                batch_global_inputs,
                batch_global_masks,
                metadata,
            )
            real_predictions = model_cd(
                batch_local_targets,
                batch_local_masks,
                batch_global_targets,
                real_global_mask_embedded,
            )
            loss_cd_real = bce_loss(real_predictions, real_labels)

            # Combined discriminator loss
            loss_cd = (loss_cd_fake + loss_cd_real) * alpha / 2.0

            # Backward and optimize discriminator
            loss_cd.backward()
            opt_cd.step()
            opt_cd.zero_grad()

            # --------------------------------------
            # 2. Then train the generator (completion network)
            # --------------------------------------
            # Generate outputs and calculate reconstruction loss
            fake_outputs = model_cn(
                batch_local_inputs,
                batch_local_masks,
                batch_global_inputs,
                batch_global_masks,
            )
            loss_cn_recon = mse_loss(fake_outputs, batch_local_targets)

            # Calculate adversarial loss (fool the discriminator)
            fake_global_embedded, fake_global_mask_embedded = merge_local_to_global(
                fake_outputs, batch_global_inputs, batch_global_masks, metadata
            )
            fake_predictions = model_cd(
                fake_outputs,
                batch_local_masks,
                batch_global_targets,
                fake_global_mask_embedded,
            )
            loss_cn_adv = bce_loss(
                fake_predictions, real_labels
            )  # Try to make discriminator think it's real

            # Combined generator loss
            loss_cn = loss_cn_recon + alpha * loss_cn_adv

            # Backward and optimize generator
            loss_cn.backward()
            opt_cn.step()
            opt_cn.zero_grad()

            step += 1
            pbar.set_description(
                f"Phase 3 | D loss: {loss_cd.item():.5f}, G loss: {loss_cn.item():.5f}"
            )
            pbar.update(1)

            # Testing and saving snapshots
            if step % snaperiod_3 == 0:
                # Run validation
                model_cn.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for val_batch in val_loader:
                        # Prepare validation batch
                        (
                            val_local_inputs,
                            val_local_masks,
                            val_local_targets,
                            val_global_inputs,
                            val_global_masks,
                            val_global_targets,
                            metadata,
                        ) = batch  # prepare_batch_for_model2(val_batch, device)
                        val_local_inputs = val_local_inputs.to(device)
                        val_local_masks = val_local_masks.to(device)
                        val_local_targets = val_local_targets.to(device)
                        val_global_inputs = val_global_inputs.to(device)
                        val_global_masks = val_global_masks.to(device)
                        val_global_targets = val_global_targets.to(device)

                        # Generate completions
                        val_outputs = model_cn(
                            val_local_inputs,
                            val_local_masks,
                            val_global_inputs,
                            val_global_masks,
                        )

                        # Calculate loss
                        batch_loss = mse_loss(val_outputs, val_local_targets)
                        val_loss += batch_loss.item()

                # Calculate average validation loss
                avg_val_loss = val_loss / len(val_loader)
                print(f"Step {step}, Validation Loss: {avg_val_loss:.5f}")

                # Save if best model
                if avg_val_loss < best_val_loss_joint:
                    best_val_loss_joint = avg_val_loss
                    cn_best_path = os.path.join(result_dir, "phase_3", "model_cn_best")
                    cd_best_path = os.path.join(result_dir, "phase_3", "model_cd_best")
                    torch.save(model_cn.state_dict(), cn_best_path)
                    torch.save(model_cd.state_dict(), cd_best_path)

                # Generate test completions for visualization
                test_batch = next(iter(val_loader))

                # Prepare test batch
                (
                    test_local_inputs,
                    test_local_masks,
                    test_local_targets,
                    test_global_inputs,
                    test_global_masks,
                    test_global_targets,
                    metadata,
                ) = batch  # prepare_batch_for_model2(test_batch, device)
                test_local_inputs = test_local_inputs.to(device)
                test_local_masks = test_local_masks.to(device)
                test_local_targets = test_local_targets.to(device)
                test_global_inputs = test_global_inputs.to(device)
                test_global_masks = test_global_masks.to(device)
                test_global_targets = test_global_targets.to(device)

                # Generate completions
                test_output = model_cn(
                    test_local_inputs,
                    test_local_masks,
                    test_global_inputs,
                    test_global_masks,
                )

                # Extract elevation channel from inputs
                # test_input_elev = test_local_inputs[:, 0:1, :, :]

                # Blend the output with the input based on the mask
                # completed:最终测试生成的图片
                completed, completed_mask = merge_local_to_global(
                    test_output, test_global_inputs, test_global_masks, metadata
                )
                """completed = test_output * test_local_masks + test_input_elev * (
                    1 - test_local_masks
                )"""

                """# Save images
                imgs = torch.cat(
                    (
                        test_local_targets[:num_test_completions].cpu(),
                        test_input_elev[:num_test_completions].cpu(),
                        test_output[:num_test_completions].cpu(),
                        completed[:num_test_completions].cpu(),
                    ),
                    dim=0,
                )

                imgpath = os.path.join(result_dir, "phase_3", f"step{step}.png")
                save_image(imgs, imgpath, nrow=num_test_completions)"""

                # Save both models
                cn_path = os.path.join(result_dir, "phase_3", f"model_cn_step{step}")
                cd_path = os.path.join(result_dir, "phase_3", f"model_cd_step{step}")
                torch.save(model_cn.state_dict(), cn_path)
                torch.save(model_cd.state_dict(), cd_path)

                model_cn.train()
                model_cd.train()

            if step >= steps_3:
                break

    pbar.close()
    print("Training complete!")


if __name__ == "__main__":
    train()
