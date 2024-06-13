import torch
import torch.nn as nn
from utils.preprocessing import preprocess_image, get_patch, image2world

def train(model, train_loader, train_images, e, obs_len, pred_len, batch_size, params, gt_template, device, input_template, optimizer, criterion, dataset_name, homo_mat, preprocessing_fn, seg_mask=False):
    """
    Run training for one epoch
    :param model: torch model
    :param train_loader: torch dataloader
    :param train_images: dict with keys: scene_name value: preprocessed image as torch.Tensor
    :param e: epoch number
    :param params: dict of hyperparameters
    :param gt_template:  precalculated Gaussian heatmap template as torch.Tensor
    :return: train_ADE, train_FDE, train_loss for one epoch
    """
    accum = 0
    accum_iter = 0
    train_loss = 0
    train_ADE = []
    train_FDE = []
    model.train()
    
    if params['grad_accum'] and batch_size > params['batch_size_gpu']:
        assert batch_size % params['batch_size_gpu'] == 0, f"{batch_size} can't divide by {params['batch_size_gpu']}"
        accum = batch_size
        batch_size = params['batch_size_gpu']
    
    # Outer loop, for loop over each scene as scenes have different image size and to do feature extract only once
    for batch, (trajectory, meta, scene) in enumerate(train_loader):
        # Get scene image and apply feature extract
        model.eval()
        scene_image = train_images[scene]
        scene_image = preprocess_image(scene_image, preprocessing_fn, seg_mask=seg_mask).to(device).unsqueeze(0)
        scene_image_static = model.feature_extract(scene_image)
        model.train()
        
        # Inner loop, for each trajectory in the scene
        for i in range(0, len(trajectory), batch_size):
            if params['feature_extract']:
                scene_image = model.feature_extract_trans(scene_image_static)
            else:
                scene_image = scene_image_static
            
            # Create Heatmaps for past and ground-truth future trajectories
            _, _, H, W = scene_image.shape  # image shape
            
            observed = trajectory[i:i+batch_size, :obs_len].reshape(-1, 2).cpu().numpy()
            observed_map = get_patch(input_template, observed, H, W)
            observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])
            
            gt_future = trajectory[i:i+batch_size, obs_len:].to(device)
            gt_future_map = get_patch(gt_template, gt_future.reshape(-1, 2).cpu().numpy(), H, W)
            gt_future_map = torch.stack(gt_future_map).reshape([-1, pred_len, H, W])
            
            # Concatenate heatmap and semantic map
            img_map = scene_image.expand(observed_map.shape[0], -1, -1, -1)  # expand to match heatmap size (scene_image.copy * batch_size)
            feature_input = torch.cat([img_map, observed_map], dim=1)
            
            """
            Forward
            """
            # Calculate features
            features = model.pred_features(feature_input)
            
            # Predict trajectory
            pred_traj_map = model.pred_traj(features)
            
            """
            Backprop
            """
            # Use point probability distribution map to calculate loss (BCEWithLogitsLoss)
            loss = criterion(pred_traj_map, gt_future_map) * params['loss_scale']
            
            # Normalize loss to accumulated batch size
            if accum:
                accum_iter += batch_size
                loss /= (accum / batch_size)
            else:
                # If we don't need gradient accumulation then do zero_grad first, otherwise end.
                optimizer.zero_grad()
            
            loss.backward()
            
            if (not accum) or int(accum_iter % accum) == 0 or ((batch + 1) == len(train_loader) and (i + batch_size) >= len(trajectory)):
                if params['clip_grad']:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['max_norm'], norm_type=2)
                
                optimizer.step()
                
                if accum:
                    optimizer.zero_grad()
            
            with torch.set_grad_enabled(False):
                # Evaluate using Softargmax, not a very exact evaluation but a lot faster than full prediction
                pred_traj = model.softargmax(pred_traj_map)
                pred_goal = model.softargmax(pred_traj_map[:, -1:])
                
                # converts ETH/UCY pixel coordinates back into world-coordinates
                if dataset_name == 'eth':
                    pred_goal = image2world(pred_goal, scene, homo_mat, params['resize'])
                    pred_traj = image2world(pred_traj, scene, homo_mat, params['resize'])
                    gt_future = image2world(gt_future, scene, homo_mat, params['resize'])
                
                ADE = ((((gt_future - pred_traj) / params['resize']) ** 2).sum(dim=2) ** 0.5).mean(dim=1)
                FDE = ((((gt_future[:, -1:] - pred_goal[:, -1:]) / params['resize']) ** 2).sum(dim=2) ** 0.5).mean(dim=1)
                
                train_loss += loss
                
                train_ADE.append(ADE)
                train_FDE.append(FDE)
            
    train_ADE = torch.cat(train_ADE).mean()
    train_FDE = torch.cat(train_FDE).mean()
    
    return train_ADE.item(), train_FDE.item(), train_loss.item()

def val(model, val_loader, val_images, obs_len, pred_len, batch_size, params, gt_template, device, input_template, criterion, preprocessing_fn, dataset_name=None, seg_mask=False):
    model.eval()
    total_loss = 0
    val_loss = 0
    
    with torch.no_grad():
        for trajectory, _, scene in val_loader:
            scene_image = val_images[scene]
            scene_image = preprocess_image(scene_image, preprocessing_fn, seg_mask=seg_mask).to(device).unsqueeze(0)
            scene_image = model.feature_extract(scene_image)
            if params['feature_extract']:
                scene_image = model.feature_extract_trans(scene_image)
            
            for i in range(0, len(trajectory), batch_size):
                _, _, H, W = scene_image.shape
                
                observed = trajectory[i:i+batch_size, :obs_len].reshape(-1, 2).cpu().numpy()
                observed_map = get_patch(input_template, observed, H, W)
                observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])
                
                gt_future = trajectory[i:i + batch_size, obs_len:].to(device)
                gt_future_map = get_patch(gt_template, gt_future.reshape(-1, 2).cpu().numpy(), H, W)
                gt_future_map = torch.stack(gt_future_map).reshape([-1, pred_len, H, W])
                
                img_map = scene_image.expand(observed_map.shape[0], -1, -1, -1)
                
                feature_input = torch.cat([img_map, observed_map], dim=1)
                features = model.pred_features(feature_input)
                
                pred_traj_map = model.pred_traj(features)
                
                # Use point probability distribution map to calculate loss (BCEWithLogitsLoss)
                loss = criterion(pred_traj_map, gt_future_map) * params['loss_scale']
                
                total_loss += loss
        
        val_loss = total_loss.mean().item()
    
    return val_loss
