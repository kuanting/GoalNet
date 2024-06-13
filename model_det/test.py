import torch
import torch.nn as nn
from utils.preprocessing import preprocess_image, get_patch, image2world

def evaluate(model, val_loader, val_images, obs_len, pred_len, batch_size, device, input_template, params, dataset_name=None, homo_mat=None, preprocessing_fn=None, seg_mask=False):
    """
    :param model: torch model
    :param val_loader: torch dataloader
    :param val_images: dict with keys: scene_name value: preprocessed image as torch.Tensor
    :param obs_len: int, observed timesteps
    :param batch_size: int, batch_size
    :param device: torch device
    :param input_template: torch.Tensor, heatmap template
    :param dataset_name: ['sdd','ind','eth','pie','jaad']
    :param params: dict with hyperparameters
    :param homo_mat: dict with homography matrix
    :return: val_ADE, val_FDE for one epoch
    """
    model.eval()
    val_ADE = []
    val_FDE = []
    val_MSE = {'MSE15': [], 'MSE30': [], 'MSE45':[]}
    if params['matrics'] == 'mse' and params['bbox_wh_model'] is not None:
        model_pred_wh = torch.jit.load(params['bbox_wh_model']).to(device)
        model_pred_wh.eval()
    
    with torch.no_grad():
        # outer loop, for loop over each scene as scenes have different image size and to do feature extract only once
        for trajectory, meta, scene in val_loader:
            # Get scene image and apply feature extract
            scene_image = val_images[scene]
            scene_image = preprocess_image(scene_image, preprocessing_fn, seg_mask=seg_mask).to(device).unsqueeze(0)
            scene_image = model.feature_extract(scene_image)
            if params['feature_extract']:
                scene_image = model.feature_extract_trans(scene_image)
            
            for i in range(0, len(trajectory), batch_size):
                # Create Heatmaps for past and ground-truth future trajectories
                _, _, H, W = scene_image.shape
                observed = trajectory[i:i+batch_size, :obs_len, :].reshape(-1, 2).cpu().numpy()
                observed_map = get_patch(input_template, observed, H, W)
                observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])
                
                gt_future = trajectory[i:i+batch_size, obs_len:].to(device)
                img_map = scene_image.expand(observed_map.shape[0], -1, -1, -1)
                
                # Forward pass
                # Calculate features
                feature_input = torch.cat([img_map, observed_map], dim=1)
                features = model.pred_features(feature_input)
                
                # Predict goal and waypoint probability distributions
                pred_traj_map = model.pred_traj(features)
                
                pred_traj = model.softargmax(pred_traj_map)
                pred_goal = model.softargmax(pred_traj_map[:, -1:])
                
                # converts ETH/UCY pixel coordinates back into world-coordinates
                if dataset_name == 'eth':
                    pred_goal = image2world(pred_goal, scene, homo_mat, params['resize'])
                    pred_traj = image2world(pred_traj, scene, homo_mat, params['resize'])
                    gt_future = image2world(gt_future, scene, homo_mat, params['resize'])
                
                if params['matrics'] == 'ade':
                    # ADE / FDE
                    val_ADE.append(((((gt_future - pred_traj) / params['resize']) ** 2).sum(dim=2) ** 0.5).mean(dim=1))
                    val_FDE.append(((((gt_future[:, -1:] - pred_goal[:, -1:]) / params['resize']) ** 2).sum(dim=2) ** 0.5).mean(dim=1))
                elif params['matrics'] == 'mse':
                    # CMSE / CFMSE
                    val_ADE.append((((gt_future - pred_traj) / params['resize']) ** 2).mean(dim=(1,2)))
                    val_FDE.append((((gt_future[:, -1:] - pred_goal[:, -1:]) / params['resize']) ** 2).mean(dim=2))
                
                # MSE15/30/45
                if params['matrics'] == 'mse' and params['bbox_wh_model'] is not None:
                    pred_traj = pred_traj / params['resize'] # [N,45,2]
                    total_len = obs_len + pred_len
                    
                    wh = torch.Tensor(meta[['w', 'h']].to_numpy().astype('float32').reshape(-1, total_len, 2))
                    observed_xy = trajectory[i:i+batch_size, :obs_len, :].to(device) / params['resize']  # [N,15,2]
                    gt_future_xy = trajectory[i:i+batch_size, obs_len:, :].to(device) / params['resize'] # [N,45,2]
                    observed_wh = wh[i:i+batch_size, :obs_len, :].to(device)  # [N,15,2]
                    gt_future_wh = wh[i:i+batch_size, obs_len:, :].to(device) # [N,45,2]
                    
                    # X[N,60], Y[N,60], W[N,15], H[N,15] = [N,150]
                    input_x = torch.cat([observed_xy[:, :, 0], pred_traj[:, :, 0]], dim=1)
                    input_y = torch.cat([observed_xy[:, :, 1], pred_traj[:, :, 1]], dim=1)
                    input_xywh = torch.cat([input_x, input_y, observed_wh[:, :, 0], observed_wh[:, :, 1]], dim=1)
                    
                    # Predict bbox width and height by xywh flatten data
                    pred_wh = model_pred_wh(input_xywh)
                    pred_wh = torch.stack([pred_wh[:, :pred_len], pred_wh[:, pred_len:]], dim=2) # dim = max_dim+1; [N,45+45] -> [N,45,2]
                    
                    def xywh_to_x1y1x2y2(cxcy, bbox_wh):
                        x1y1x2y2 = []
                        for n in range(len(cxcy)):
                            x1y1x2y2.append([])
                            for xy, wh in zip(cxcy[n], bbox_wh[n]):
                                x1, x2 = xy[0] - (wh[0] / 2), xy[0] + (wh[0] / 2)
                                y1, y2 = xy[1] - (wh[1] / 2), xy[1] + (wh[1] / 2)
                                x1y1x2y2[n].append([x1, y1, x2, y2])
                        
                        return torch.Tensor(x1y1x2y2)
                    
                     # [N,45,4]
                    gt_future_x1y1x2y2 = xywh_to_x1y1x2y2(gt_future_xy, gt_future_wh).to(device)
                    future_samples_x1y1x2y2 = xywh_to_x1y1x2y2(pred_traj, pred_wh).to(device)
                    
                    val_MSE['MSE45'].append(((gt_future_x1y1x2y2 - future_samples_x1y1x2y2) ** 2).mean(dim=(1,2)))
                    val_MSE['MSE30'].append(((gt_future_x1y1x2y2[:, :30] - future_samples_x1y1x2y2[:, :30]) ** 2).mean(dim=(1,2)))
                    val_MSE['MSE15'].append(((gt_future_x1y1x2y2[:, :15] - future_samples_x1y1x2y2[:, :15]) ** 2).mean(dim=(1,2)))
        
        if len(val_ADE) > 0:
            val_ADE = torch.cat(val_ADE).mean().item()
            val_FDE = torch.cat(val_FDE).mean().item()
        if len(val_MSE['MSE15']) > 0:
            val_MSE['MSE15'] = torch.cat(val_MSE['MSE15']).mean().item()
            val_MSE['MSE30'] = torch.cat(val_MSE['MSE30']).mean().item()
            val_MSE['MSE45'] = torch.cat(val_MSE['MSE45']).mean().item()
    
    return val_ADE, val_FDE, val_MSE
