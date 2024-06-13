import torch
import torch.nn as nn
from utils.preprocessing import preprocess_image, get_patch, sampling, image2world, torch_multivariate_gaussian_heatmap
from utils.kmeans import kmeans
from utils.kde_nll import compute_kde_nll

def evaluate(model, val_loader, val_images, num_goals, num_traj, obs_len, pred_len, batch_size, device, input_template, params, dataset_name=None, homo_mat=None, preprocessing_fn=None, seg_mask=False, use_TTST=False, use_CWS=False, rel_thresh=0.002, CWS_params=None):
    """
    :param model: torch model
    :param val_loader: torch dataloader
    :param val_images: dict with keys: scene_name value: preprocessed image as torch.Tensor
    :param num_goals: int, number of goals
    :param num_traj: int, number of trajectories per goal
    :param obs_len: int, observed timesteps
    :param batch_size: int, batch_size
    :param device: torch device
    :param input_template: torch.Tensor, heatmap template
    :param use_TTST: bool
    :param use_CWS: bool
    :param rel_thresh: float
    :param CWS_params: dict
    :param dataset_name: ['sdd','ind','eth','pie','jaad']
    :param params: dict with hyperparameters
    :param homo_mat: dict with homography matrix
    :return: val_ADE, val_FDE for one epoch
    """
    # Considering the actual use situation, you may want to use evaluate_byscene
    if params['train_mode'] == 'trajectory':
        return evaluate_bytraj(model, val_loader, val_images, num_goals, num_traj, obs_len, pred_len, device, input_template, params, preprocessing_fn, seg_mask, params, use_TTST, use_CWS, rel_thresh, CWS_params)
    else:
        return evaluate_byscene(model, val_loader, val_images, num_goals, num_traj, obs_len, pred_len, batch_size, device, input_template, params, dataset_name, homo_mat, preprocessing_fn, seg_mask, use_TTST, use_CWS, rel_thresh, CWS_params)

# Note: By this method, your real batch_size for testing may smaller than your batch_size setting
def evaluate_byscene(model, val_loader, val_images, num_goals, num_traj, obs_len, pred_len, batch_size, device, input_template, params, dataset_name=None, homo_mat=None, preprocessing_fn=None, seg_mask=False, use_TTST=False, use_CWS=False, rel_thresh=0.002, CWS_params=None):
    model.eval()
    val_ADE = []
    val_FDE = []
    val_MSE = {'MSE15': [], 'MSE30': [], 'MSE45':[]}
    KDE_NLL = []
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
                pred_waypoint_map = model.pred_goal(features)
                pred_waypoint_map = pred_waypoint_map[:, params['waypoints']]
                
                pred_waypoint_map_sigmoid = pred_waypoint_map / params['temperature']
                pred_waypoint_map_sigmoid = model.sigmoid(pred_waypoint_map_sigmoid)
                
                ################################################ TTST ##################################################
                if use_TTST and num_goals > 1:
                    # TTST Begin
                    # sample a large amount of goals to be clustered
                    goal_samples = sampling(pred_waypoint_map_sigmoid[:, -1:], num_samples=10000, replacement=True, rel_threshold=rel_thresh)
                    goal_samples = goal_samples.permute(2, 0, 1, 3)
                    
                    num_clusters = num_goals - 1
                    goal_samples_softargmax = model.softargmax(pred_waypoint_map[:, -1:])  # first sample is softargmax sample
                    
                    # Iterate through all person/batch_num, as this k-Means implementation doesn't support batched clustering
                    goal_samples_list = []
                    for person in range(goal_samples.shape[1]):
                        goal_sample = goal_samples[:, person, 0]

                        # Actual k-means clustering, Outputs:
                        # cluster_ids_x -  Information to which cluster_idx each point belongs to
                        # cluster_centers - list of centroids, which are our new goal samples
                        cluster_ids_x, cluster_centers = kmeans(X=goal_sample, num_clusters=num_clusters, distance='euclidean', device=device, tqdm_flag=False, tol=0.001, iter_limit=1000)
                        goal_samples_list.append(cluster_centers)
                    
                    goal_samples = torch.stack(goal_samples_list).permute(1, 0, 2).unsqueeze(2)
                    goal_samples = torch.cat([goal_samples_softargmax.unsqueeze(0), goal_samples], dim=0)
                    # TTST End
                
                # Not using TTST
                else:
                    goal_samples = sampling(pred_waypoint_map_sigmoid[:, -1:], num_samples=num_goals)
                    goal_samples = goal_samples.permute(2, 0, 1, 3)
                
                # Predict waypoints:
                # in case len(params['waypoints']) == 1, so only goal is needed (goal counts as one waypoint in this implementation)
                if len(params['waypoints']) == 1:
                    waypoint_samples = goal_samples
                
                ################################################ CWS ###################################################
                # CWS Begin
                if use_CWS and len(params['waypoints']) > 1:
                    sigma_factor = CWS_params['sigma_factor']
                    ratio = CWS_params['ratio']
                    rot = CWS_params['rot']
                    
                    goal_samples = goal_samples.repeat(num_traj, 1, 1, 1)  # repeat K_a times
                    last_observed = trajectory[i:i+batch_size, obs_len-1].to(device)  # [N, 2]
                    waypoint_samples_list = []  # in the end this should be a list of [K, N, # waypoints, 2] waypoint coordinates
                    for g_num, waypoint_samples in enumerate(goal_samples.squeeze(2)):
                        waypoint_list = []  # for each K sample have a separate list
                        waypoint_list.append(waypoint_samples)
                        
                        for waypoint_num in reversed(range(len(params['waypoints'])-1)):
                            distance = last_observed - waypoint_samples
                            gaussian_heatmaps = []
                            traj_idx = g_num // num_goals  # idx of trajectory for the same goal
                            for dist, coordinate in zip(distance, waypoint_samples):  # for each person
                                length_ratio = 1 / (waypoint_num + 2)
                                gauss_mean = coordinate + (dist * length_ratio)  # Get the intermediate point's location using CV model
                                sigma_factor_ = sigma_factor - traj_idx
                                gaussian_heatmaps.append(torch_multivariate_gaussian_heatmap(gauss_mean, H, W, dist, sigma_factor_, ratio, device, rot))
                            gaussian_heatmaps = torch.stack(gaussian_heatmaps)  # [N, H, W]
                            
                            waypoint_map_before = pred_waypoint_map_sigmoid[:, waypoint_num]
                            waypoint_map = waypoint_map_before * gaussian_heatmaps
                            # normalize waypoint map
                            waypoint_map = (waypoint_map.flatten(1) / waypoint_map.flatten(1).sum(-1, keepdim=True)).view_as(waypoint_map)
                            
                            # For first traj samples use softargmax
                            if g_num // num_goals == 0:
                                # Softargmax
                                waypoint_samples = model.softargmax_on_softmax_map(waypoint_map.unsqueeze(0))
                                waypoint_samples = waypoint_samples.squeeze(0)
                            else:
                                waypoint_samples = sampling(waypoint_map.unsqueeze(1), num_samples=1, rel_threshold=0.05)
                                waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
                                waypoint_samples = waypoint_samples.squeeze(2).squeeze(0)
                            waypoint_list.append(waypoint_samples)
                        
                        waypoint_list = waypoint_list[::-1]
                        waypoint_list = torch.stack(waypoint_list).permute(1, 0, 2)  # permute back to [N, # waypoints, 2]
                        waypoint_samples_list.append(waypoint_list)
                    waypoint_samples = torch.stack(waypoint_samples_list)
                # CWS End
                
                # If not using CWS, and we still need to sample waypoints (i.e., not only goal is needed)
                elif not use_CWS and len(params['waypoints']) > 1:
                    waypoint_samples = sampling(pred_waypoint_map_sigmoid[:, :-1], num_samples=num_goals * num_traj)
                    waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
                    goal_samples = goal_samples.repeat(num_traj, 1, 1, 1)  # repeat K_a times
                    waypoint_samples = torch.cat([waypoint_samples, goal_samples], dim=2)
                
                # Interpolate trajectories given goal and waypoints
                future_samples = []
                for waypoint in waypoint_samples:
                    waypoint_map = get_patch(input_template, waypoint.reshape(-1, 2).cpu().numpy(), H, W)
                    waypoint_map = torch.stack(waypoint_map).reshape([-1, len(params['waypoints']), H, W])
                    
                    waypoint_maps_downsampled = [nn.AvgPool2d(kernel_size=2 ** i, stride=2 ** i)(waypoint_map) for i in range(1, len(features))]
                    waypoint_maps_downsampled = [waypoint_map] + waypoint_maps_downsampled
                    
                    traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, waypoint_maps_downsampled)]
                    
                    pred_traj_map = model.pred_traj(traj_input)
                    pred_traj = model.softargmax(pred_traj_map)
                    future_samples.append(pred_traj)
                future_samples = torch.stack(future_samples)
                
                gt_goal = gt_future[:, -1:]
                
                # converts ETH/UCY pixel coordinates back into world-coordinates
                if dataset_name == 'eth':
                    waypoint_samples = image2world(waypoint_samples, scene, homo_mat, params['resize'])
                    pred_traj = image2world(pred_traj, scene, homo_mat, params['resize'])
                    gt_future = image2world(gt_future, scene, homo_mat, params['resize'])
                
                if params['matrics'] == 'ade':
                    # ADE / FDE
                    val_ADE.append(((((gt_future - future_samples) / params['resize']) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
                    val_FDE.append(((((gt_goal - waypoint_samples[:, :, -1:]) / params['resize']) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
                elif params['matrics'] == 'mse':
                    # CMSE / CFMSE
                    val_ADE.append((((gt_future - future_samples) / params['resize']) ** 2).mean(dim=(2,3)).min(dim=0)[0])
                    val_FDE.append((((gt_goal - waypoint_samples[:, :, -1:]) / params['resize']) ** 2).mean(dim=3).min(dim=0)[0])
                
                # MSE15/30/45
                if params['matrics'] == 'mse' and params['bbox_wh_model'] is not None:
                    # [K,N,45,2] -> [N,K,45,2]
                    future_samples = future_samples.permute(1, 0, 2, 3) / params['resize']
                    total_len = obs_len + pred_len
                    
                    wh = torch.Tensor(meta[['w', 'h']].to_numpy().astype('float32').reshape(-1, total_len, 2))
                    observed_xy = trajectory[i:i+batch_size, :obs_len, :].to(device) / params['resize']  # [N,15,2]
                    gt_future_xy = trajectory[i:i+batch_size, obs_len:, :].to(device) / params['resize'] # [N,45,2]
                    observed_wh = wh[i:i+batch_size, :obs_len, :].to(device)  # [N,15,2]
                    gt_future_wh = wh[i:i+batch_size, obs_len:, :].to(device) # [N,45,2]
                    
                    input_xywh = []
                    K = future_samples.shape[1]
                    for j, n_traj in enumerate(future_samples):
                        # X[K,60], Y[K,60], W[K,15], H[K,15] = [K,150]
                        input_x = torch.cat([observed_xy[j:j+1, :, 0].expand(K, -1), n_traj[:, :, 0]], dim=1)
                        input_y = torch.cat([observed_xy[j:j+1, :, 1].expand(K, -1), n_traj[:, :, 1]], dim=1)
                        input_xywh.append(torch.cat([input_x, input_y, observed_wh[j:j+1, :, 0].expand(K, -1), observed_wh[j:j+1, :, 1].expand(K, -1)], dim=1))
                    
                    input_xywh = torch.cat(input_xywh) # [N*K,150]
                    
                    # Predict bbox width and height by xywh flatten data
                    pred_wh = model_pred_wh(input_xywh)
                    pred_wh = torch.stack([pred_wh[:, :pred_len], pred_wh[:, pred_len:]], dim=2) # dim = max_dim+1; [N*K,45+45] -> [N*K,45,2]
                    
                    def xywh_to_x1y1x2y2(cxcy, bbox_wh):
                        x1y1x2y2 = []
                        for n in range(len(cxcy)):
                            x1y1x2y2.append([])
                            for xy, wh in zip(cxcy[n], bbox_wh[n]):
                                x1, x2 = xy[0] - (wh[0] / 2), xy[0] + (wh[0] / 2)
                                y1, y2 = xy[1] - (wh[1] / 2), xy[1] + (wh[1] / 2)
                                x1y1x2y2[n].append([x1, y1, x2, y2])
                        
                        return torch.Tensor(x1y1x2y2)
                    
                    gt_future_x1y1x2y2 = xywh_to_x1y1x2y2(gt_future_xy, gt_future_wh).to(device) # [N,45,4]
                    future_samples_x1y1x2y2 = torch.cat([xywh_to_x1y1x2y2(n_traj, pred_wh[j*K:(j+1)*K]).unsqueeze(0).to(device) for j, n_traj in enumerate(future_samples)]) # [N,K,45,4]
                    future_samples_x1y1x2y2 = future_samples_x1y1x2y2.permute(1, 0, 2, 3) # [K,N,45,4]
                    
                    val_MSE['MSE45'].append(((gt_future_x1y1x2y2 - future_samples_x1y1x2y2) ** 2).mean(dim=(2,3)).min(dim=0)[0])
                    val_MSE['MSE30'].append(((gt_future_x1y1x2y2[:, :30] - future_samples_x1y1x2y2[:, :, :30]) ** 2).mean(dim=(2,3)).min(dim=0)[0])
                    val_MSE['MSE15'].append(((gt_future_x1y1x2y2[:, :15] - future_samples_x1y1x2y2[:, :, :15]) ** 2).mean(dim=(2,3)).min(dim=0)[0])
                    
                    # As BiTraP, NLL should set K = 2000
                    if params['eval_nll']:
                        # [K,N,45,4] -> [N,45,K,4]
                        pred = future_samples_x1y1x2y2.detach().permute(1, 2, 0, 3).cpu().numpy()
                        gt = gt_future_x1y1x2y2.detach().cpu().numpy()
                        for i in range(len(pred)):
                            KDE_NLL.append(torch.tensor(compute_kde_nll(pred[i:i+1], gt[i:i+1])).to(device))
        
        if len(val_ADE) > 0:
            val_ADE = torch.cat(val_ADE).mean().item()
            val_FDE = torch.cat(val_FDE).mean().item()
        if len(val_MSE['MSE15']) > 0:
            val_MSE['MSE15'] = torch.cat(val_MSE['MSE15']).mean().item()
            val_MSE['MSE30'] = torch.cat(val_MSE['MSE30']).mean().item()
            val_MSE['MSE45'] = torch.cat(val_MSE['MSE45']).mean().item()
        if len(KDE_NLL) > 0:
            KDE_NLL = torch.stack(KDE_NLL, dim=0).mean().item()
    
    return val_ADE, val_FDE, val_MSE, KDE_NLL

# Note: By this method, your scene W,H in test set should be same
def evaluate_bytraj(model, val_loader, val_images, num_goals, num_traj, obs_len, device, input_template, params, preprocessing_fn=None, seg_mask=False, use_TTST=False, use_CWS=False, rel_thresh=0.002, CWS_params=None):
    model.eval()
    val_ADE = []
    val_FDE = []
    val_MSE = {'MSE15': [], 'MSE30': [], 'MSE45':[]}
    KDE_NLL = []
    if params['matrics'] == 'mse' and params['bbox_wh_model'] is not None:
        model_pred_wh = torch.jit.load(params['bbox_wh_model']).to(device)
        model_pred_wh.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            gt_future = []
            feature_input = []
            
            # Prepare for bacth test data
            for i in range(len(batch["scene"])):
                scene = batch["scene"][i]
                trajectory = batch["trajectory"][i]
                
                # Get scene image and apply semantic segmentation
                scene_image = val_images[scene]
                scene_image = preprocess_image(scene_image, preprocessing_fn).to(device).unsqueeze(0)
                scene_image = model.feature_extract(scene_image)
                if params['feature_extract']:
                    scene_image = model.feature_extract_trans(scene_image)
                
                # Create Heatmaps for past and ground-truth future trajectories
                _, _, H, W = scene_image.shape  # image shape
                
                observed = trajectory[:, :obs_len].reshape(-1, 2).cpu().numpy()
                observed_map = get_patch(input_template, observed, H, W)
                observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])
                
                gt_future.append(trajectory[:, obs_len:].to(device))
                
                # Concatenate heatmap and semantic map
                feature_input.append(torch.cat([scene_image, observed_map], dim=1))
                
            gt_future = torch.cat(gt_future)
            feature_input = torch.cat(feature_input)
            
            # Forward pass
            # Calculate features
            features = model.pred_features(feature_input)
            
            # Predict goal and waypoint probability distributions
            pred_waypoint_map = model.pred_goal(features)
            pred_waypoint_map = pred_waypoint_map[:, params['waypoints']]
            
            pred_waypoint_map_sigmoid = pred_waypoint_map / params['temperature']
            pred_waypoint_map_sigmoid = model.sigmoid(pred_waypoint_map_sigmoid)
            
            ################################################ TTST ##################################################
            if use_TTST:
                # TTST Begin
                # sample a large amount of goals to be clustered
                goal_samples = sampling(pred_waypoint_map_sigmoid[:, -1:], num_samples=10000, replacement=True, rel_threshold=rel_thresh)
                goal_samples = goal_samples.permute(2, 0, 1, 3)
                
                num_clusters = num_goals - 1
                goal_samples_softargmax = model.softargmax(pred_waypoint_map[:, -1:])  # first sample is softargmax sample
                
                # Iterate through all person/batch_num, as this k-Means implementation doesn't support batched clustering
                goal_samples_list = []
                for person in range(goal_samples.shape[1]):
                    goal_sample = goal_samples[:, person, 0]

                    # Actual k-means clustering, Outputs:
                    # cluster_ids_x -  Information to which cluster_idx each point belongs to
                    # cluster_centers - list of centroids, which are our new goal samples
                    cluster_ids_x, cluster_centers = kmeans(X=goal_sample, num_clusters=num_clusters, distance='euclidean', device=device, tqdm_flag=False, tol=0.001, iter_limit=1000)
                    goal_samples_list.append(cluster_centers)
                
                goal_samples = torch.stack(goal_samples_list).permute(1, 0, 2).unsqueeze(2)
                goal_samples = torch.cat([goal_samples_softargmax.unsqueeze(0), goal_samples], dim=0)
                # TTST End
            
            # Not using TTST
            else:
                goal_samples = sampling(pred_waypoint_map_sigmoid[:, -1:], num_samples=num_goals)
                goal_samples = goal_samples.permute(2, 0, 1, 3)
            
            # Predict waypoints:
            # in case len(params['waypoints']) == 1, so only goal is needed (goal counts as one waypoint in this implementation)
            if len(params['waypoints']) == 1:
                waypoint_samples = goal_samples
            
            ################################################ CWS ###################################################
            # CWS Begin
            if use_CWS and len(params['waypoints']) > 1:
                sigma_factor = CWS_params['sigma_factor']
                ratio = CWS_params['ratio']
                rot = CWS_params['rot']
                
                goal_samples = goal_samples.repeat(num_traj, 1, 1, 1)  # repeat K_a times
                
                last_observed = []
                for i in range(len(batch["scene"])):
                    trajectory = batch["trajectory"][i]
                    last_observed.append(trajectory[:, obs_len-1].to(device))
                last_observed = torch.cat(last_observed)  # [N, 2]
                
                waypoint_samples_list = []  # in the end this should be a list of [K, N, # waypoints, 2] waypoint coordinates
                for g_num, waypoint_samples in enumerate(goal_samples.squeeze(2)):
                    waypoint_list = []  # for each K sample have a separate list
                    waypoint_list.append(waypoint_samples)
                    
                    for waypoint_num in reversed(range(len(params['waypoints'])-1)):
                        distance = last_observed - waypoint_samples
                        gaussian_heatmaps = []
                        traj_idx = g_num // num_goals  # idx of trajectory for the same goal
                        for dist, coordinate in zip(distance, waypoint_samples):  # for each person
                            length_ratio = 1 / (waypoint_num + 2)
                            gauss_mean = coordinate + (dist * length_ratio)  # Get the intermediate point's location using CV model
                            sigma_factor_ = sigma_factor - traj_idx
                            gaussian_heatmaps.append(torch_multivariate_gaussian_heatmap(gauss_mean, H, W, dist, sigma_factor_, ratio, device, rot))
                        gaussian_heatmaps = torch.stack(gaussian_heatmaps)  # [N, H, W]
                        
                        waypoint_map_before = pred_waypoint_map_sigmoid[:, waypoint_num]
                        waypoint_map = waypoint_map_before * gaussian_heatmaps
                        # normalize waypoint map
                        waypoint_map = (waypoint_map.flatten(1) / waypoint_map.flatten(1).sum(-1, keepdim=True)).view_as(waypoint_map)
                        
                        # For first traj samples use softargmax
                        if g_num // num_goals == 0:
                            # Softargmax
                            waypoint_samples = model.softargmax_on_softmax_map(waypoint_map.unsqueeze(0))
                            waypoint_samples = waypoint_samples.squeeze(0)
                        else:
                            waypoint_samples = sampling(waypoint_map.unsqueeze(1), num_samples=1, rel_threshold=0.05)
                            waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
                            waypoint_samples = waypoint_samples.squeeze(2).squeeze(0)
                        waypoint_list.append(waypoint_samples)
                    
                    waypoint_list = waypoint_list[::-1]
                    waypoint_list = torch.stack(waypoint_list).permute(1, 0, 2)  # permute back to [N, # waypoints, 2]
                    waypoint_samples_list.append(waypoint_list)
                waypoint_samples = torch.stack(waypoint_samples_list)
            # CWS End
            
            # If not using CWS, and we still need to sample waypoints (i.e., not only goal is needed)
            elif not use_CWS and len(params['waypoints']) > 1:
                waypoint_samples = sampling(pred_waypoint_map_sigmoid[:, :-1], num_samples=num_goals * num_traj)
                waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
                goal_samples = goal_samples.repeat(num_traj, 1, 1, 1)  # repeat K_a times
                waypoint_samples = torch.cat([waypoint_samples, goal_samples], dim=2)
            
            # Interpolate trajectories given goal and waypoints
            future_samples = []
            for waypoint in waypoint_samples:
                waypoint_map = get_patch(input_template, waypoint.reshape(-1, 2).cpu().numpy(), H, W)
                waypoint_map = torch.stack(waypoint_map).reshape([-1, len(params['waypoints']), H, W])
                
                waypoint_maps_downsampled = [nn.AvgPool2d(kernel_size=2 ** i, stride=2 ** i)(waypoint_map) for i in range(1, len(features))]
                waypoint_maps_downsampled = [waypoint_map] + waypoint_maps_downsampled
                
                traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, waypoint_maps_downsampled)]
                
                pred_traj_map = model.pred_traj(traj_input)
                pred_traj = model.softargmax(pred_traj_map)
                future_samples.append(pred_traj)
            future_samples = torch.stack(future_samples)
            
            gt_goal = gt_future[:, -1:]
            
            if params['matrics'] == 'ade':
                # ADE / FDE
                val_ADE.append(((((gt_future - future_samples) / params['resize']) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
                val_FDE.append(((((gt_goal - waypoint_samples[:, :, -1:]) / params['resize']) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
            elif params['matrics'] == 'mse':
                # CMSE / CFMSE
                val_ADE.append((((gt_future - future_samples) / params['resize']) ** 2).mean(dim=(2,3)).min(dim=0)[0])
                val_FDE.append((((gt_goal - waypoint_samples[:, :, -1:]) / params['resize']) ** 2).mean(dim=3).min(dim=0)[0])
                
            # MSE15/30/45
            if params['matrics'] == 'mse' and params['bbox_wh_model'] is not None:
                # [K,N,45,2] -> [N,K,45,2]
                future_samples = future_samples.permute(1, 0, 2, 3) / params['resize']
                total_len = trajectory.shape[1]
                pred_len = total_len - obs_len
                
                wh = torch.Tensor(meta[['w', 'h']].to_numpy().astype('float32').reshape(-1, total_len, 2))
                observed_xy = trajectory[i:i+batch_size, :obs_len, :].to(device) / params['resize']  # [N,15,2]
                gt_future_xy = trajectory[i:i+batch_size, obs_len:, :].to(device) / params['resize'] # [N,45,2]
                observed_wh = wh[i:i+batch_size, :obs_len, :].to(device)  # [N,15,2]
                gt_future_wh = wh[i:i+batch_size, obs_len:, :].to(device) # [N,45,2]
                
                input_xywh = []
                K = future_samples.shape[1]
                for j, n_traj in enumerate(future_samples):
                    # X[K,60], Y[K,60], W[K,15], H[K,15] = [K,150]
                    input_x = torch.cat([observed_xy[j:j+1, :, 0].expand(K, -1), n_traj[:, :, 0]], dim=1)
                    input_y = torch.cat([observed_xy[j:j+1, :, 1].expand(K, -1), n_traj[:, :, 1]], dim=1)
                    input_xywh.append(torch.cat([input_x, input_y, observed_wh[j:j+1, :, 0].expand(K, -1), observed_wh[j:j+1, :, 1].expand(K, -1)], dim=1))
                
                input_xywh = torch.cat(input_xywh) # [N*K,150]
                
                # Predict bbox width and height by xywh flatten data
                pred_wh = model_pred_wh(input_xywh)
                pred_wh = torch.stack([pred_wh[:, :pred_len], pred_wh[:, pred_len:]], dim=2) # dim = max_dim+1; [N*K,45+45] -> [N*K,45,2]
                
                def xywh_to_x1y1x2y2(cxcy, bbox_wh):
                    x1y1x2y2 = []
                    for n in range(len(cxcy)):
                        x1y1x2y2.append([])
                        for xy, wh in zip(cxcy[n], bbox_wh[n]):
                            x1, x2 = xy[0] - (wh[0] / 2), xy[0] + (wh[0] / 2)
                            y1, y2 = xy[1] - (wh[1] / 2), xy[1] + (wh[1] / 2)
                            x1y1x2y2[n].append([x1, y1, x2, y2])
                    
                    return torch.Tensor(x1y1x2y2)
                
                gt_future_x1y1x2y2 = xywh_to_x1y1x2y2(gt_future_xy, gt_future_wh).to(device) # [N,45,4]
                future_samples_x1y1x2y2 = torch.cat([xywh_to_x1y1x2y2(n_traj, pred_wh[j*K:(j+1)*K]).unsqueeze(0).to(device) for j, n_traj in enumerate(future_samples)]) # [N,K,45,4]
                future_samples_x1y1x2y2 = future_samples_x1y1x2y2.permute(1, 0, 2, 3) # [K,N,45,4]
                
                val_MSE['MSE45'].append(((gt_future_x1y1x2y2 - future_samples_x1y1x2y2) ** 2).mean(dim=(2,3)).min(dim=0)[0])
                val_MSE['MSE30'].append(((gt_future_x1y1x2y2[:, :30] - future_samples_x1y1x2y2[:, :, :30]) ** 2).mean(dim=(2,3)).min(dim=0)[0])
                val_MSE['MSE15'].append(((gt_future_x1y1x2y2[:, :15] - future_samples_x1y1x2y2[:, :, :15]) ** 2).mean(dim=(2,3)).min(dim=0)[0])
                
                # As BiTraP, NLL should set K = 2000
                if params['eval_nll']:
                    # [K,N,45,4] -> [N,45,K,4]
                    pred = future_samples_x1y1x2y2.detach().permute(1, 2, 0, 3).cpu().numpy()
                    gt = gt_future_x1y1x2y2.detach().cpu().numpy()
                    for i in range(len(pred)):
                        KDE_NLL.append(torch.tensor(compute_kde_nll(pred[i:i+1], gt[i:i+1])).to(device))
        
        if len(val_ADE) > 0:
            val_ADE = torch.cat(val_ADE).mean().item()
            val_FDE = torch.cat(val_FDE).mean().item()
        if len(val_MSE['MSE15']) > 0:
            val_MSE['MSE15'] = torch.cat(val_MSE['MSE15']).mean().item()
            val_MSE['MSE30'] = torch.cat(val_MSE['MSE30']).mean().item()
            val_MSE['MSE45'] = torch.cat(val_MSE['MSE45']).mean().item()
        if len(KDE_NLL) > 0:
            KDE_NLL = torch.stack(KDE_NLL, dim=0).mean().item()
    
    return val_ADE, val_FDE, val_MSE, KDE_NLL
