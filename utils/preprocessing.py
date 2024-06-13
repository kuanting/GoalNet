import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import cv2
import multiprocessing
from copy import deepcopy

def gkern(kernlen=31, nsig=4):
    """ creates gaussian kernel with side length l and a sigma of sig """
    ax = np.linspace(-(kernlen - 1) / 2., (kernlen - 1) / 2., kernlen)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(nsig))
    return kernel / np.sum(kernel)

def create_gaussian_heatmap_template(size, kernlen=81, nsig=4, normalize=True):
    """ Create a big gaussian heatmap template to later get patches out """
    template = np.zeros([size, size])
    kernel = gkern(kernlen=kernlen, nsig=nsig)
    m = kernel.shape[0]
    x_low = template.shape[1] // 2 - int(np.floor(m / 2))
    x_up = template.shape[1] // 2 + int(np.ceil(m / 2))
    y_low = template.shape[0] // 2 - int(np.floor(m / 2))
    y_up = template.shape[0] // 2 + int(np.ceil(m / 2))
    template[y_low:y_up, x_low:x_up] = kernel
    if normalize:
        template = template / template.max()
    return template

def create_dist_mat(size, normalize=True):
    """ Create a big distance matrix template to later get patches out """
    middle = size // 2
    dist_mat = np.linalg.norm(np.indices([size, size]) - np.array([middle, middle])[:,None,None], axis=0)
    if normalize:
        dist_mat = dist_mat / dist_mat.max() * 2
    return dist_mat

def get_patch(template, traj, H, W):
    x = np.round(traj[:,0]).astype('int')
    y = np.round(traj[:,1]).astype('int')

    x_low = template.shape[1] // 2 - x
    x_up = template.shape[1] // 2 + W - x
    y_low = template.shape[0] // 2 - y
    y_up = template.shape[0] // 2 + H - y

    patch = [template[y_l:y_u, x_l:x_u] for x_l, x_u, y_l, y_u in zip(x_low, x_up, y_low, y_up)]

    return patch

def torch_multivariate_gaussian_heatmap(coordinates, H, W, dist, sigma_factor, ratio, device, rot=False):
    """
    Create Gaussian Kernel for CWS
    """
    ax = torch.linspace(0, H, H, device=device) - coordinates[1]
    ay = torch.linspace(0, W, W, device=device) - coordinates[0]
    xx, yy = torch.meshgrid([ax, ay], indexing='ij')
    meshgrid = torch.stack([yy, xx], dim=-1)
    radians = torch.atan2(dist[0], dist[1])
    
    c, s = torch.cos(radians), torch.sin(radians)
    R = torch.Tensor([[c, s], [-s, c]]).to(device)
    if rot:
        R = torch.matmul(torch.Tensor([[0, -1], [1, 0]]).to(device), R)
    dist_norm = dist.square().sum(-1).sqrt() + 5  # some small padding to avoid division by zero
    
    conv = torch.Tensor([[dist_norm / sigma_factor / ratio, 0], [0, dist_norm / sigma_factor]]).to(device)
    conv = torch.square(conv)
    T = torch.matmul(R, conv)
    T = torch.matmul(T, R.T)
    
    kernel = (torch.matmul(meshgrid, torch.inverse(T)) * meshgrid).sum(-1)
    kernel = torch.exp(-0.5 * kernel)
    return kernel / kernel.sum()

def preprocess_images_for_segmentation(images, encoder='resnet101', encoder_weights='imagenet', seg_mask=False, classes=6, multiproc=True):
    """ Preprocess image for pretrained semantic segmentation, input is dictionary containing images
    In case input is segmentation map, then it will create one-hot-encoding from discrete values"""
    import segmentation_models_pytorch as smp

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    
    def run_preprocessing_fn(key, im):
        nonlocal images
        if seg_mask:
            im = [(im == v) for v in range(classes)]
            im = np.stack(im, axis=-1)  # .astype('int16')
        else:
            im = preprocessing_fn(im)
        im = im.transpose(2, 0, 1).astype('float32')
        im = torch.Tensor(im)
        images[key] = im
    
    # im.dtype = uint8 -> float64 -> float32
    if multiproc:
        # Use main process cpu with thread, instead of Pool(), due to I/O cost
        with multiprocessing.pool.ThreadPool() as pool:
            imlist = [(key, im) for key, im in images.items()]
            pool.starmap(run_preprocessing_fn, imlist, chunksize=10)
    else:
        for key, im in images.items():
            run_preprocessing_fn(key, im)

def preprocess_image(image, preprocessing_fn, seg_mask=False, classes=6):
    """ Preprocess image for pretrained semantic segmentation, input is image
    In case input is segmentation map, then it will create one-hot-encoding from discrete values"""
    
    # im.dtype = uint8 -> float64 -> float32
    if seg_mask:
        im = [(image == v) for v in range(classes)]
        im = np.stack(im, axis=-1)
    else:
        im = preprocessing_fn(image)
    im = im.transpose(2, 0, 1).astype('float32')
    im = torch.Tensor(im)
    return im

def resize(images, factor, seg_mask=False):
    for key, image in images.items():
        if seg_mask:
            images[key] = cv2.resize(image, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        else:
            images[key] = cv2.resize(image, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

def pad(images, division_factor=32):
    """ Pad image so that it can be divided by division_factor, as many architectures such as UNet needs a specific size
    at it's bottlenet layer"""
    for key, im in images.items():
        if im.ndim == 3:
            H, W, C = im.shape
        else:
            H, W = im.shape
        H_new = int(np.ceil(H / division_factor) * division_factor)
        W_new = int(np.ceil(W / division_factor) * division_factor)
        im = cv2.copyMakeBorder(im, 0, H_new - H, 0, W_new - W, cv2.BORDER_CONSTANT)
        images[key] = im

def sampling(probability_map, num_samples, rel_threshold=None, replacement=False):
    # new view that has shape=[batch*timestep, H*W]
    prob_map = probability_map.contiguous().view(probability_map.size(0) * probability_map.size(1), -1)
    if rel_threshold is not None:
        thresh_values = prob_map.max(dim=1)[0].unsqueeze(1).expand(-1, prob_map.size(1))
        mask = prob_map < thresh_values * rel_threshold
        prob_map = prob_map * (~mask).int()
        prob_map = prob_map / prob_map.sum()

    # samples.shape=[batch*timestep, num_samples]
    samples = torch.multinomial(prob_map, num_samples=num_samples, replacement=replacement)
    # samples.shape=[batch, timestep, num_samples]

    # unravel sampled idx into coordinates of shape [batch, time, sample, 2]
    samples = samples.view(probability_map.size(0), probability_map.size(1), -1)
    idx = samples.unsqueeze(3)
    preds = idx.repeat(1, 1, 1, 2).float()
    preds[:, :, :, 0] = (preds[:, :, :, 0]) % probability_map.size(3)
    preds[:, :, :, 1] = torch.floor((preds[:, :, :, 1]) / probability_map.size(3))

    return preds

def image2world(image_coords, scene, homo_mat, resize):
    """
    Transform trajectories of one scene from image_coordinates to world_coordinates
    :param image_coords: torch.Tensor, shape=[num_person, (optional: num_samples), timesteps, xy]
    :param scene: string indicating current scene, options=['eth', 'hotel', 'student01', 'student03', 'zara1', 'zara2']
    :param homo_mat: dict, key is scene, value is torch.Tensor containing homography matrix (data/eth_ucy/scene_name.H)
    :param resize: float, resize factor
    :return: trajectories in world_coordinates
    """
    traj_image2world = image_coords.clone()
    if traj_image2world.dim() == 4:
        traj_image2world = traj_image2world.reshape(-1, image_coords.shape[2], 2)
    if scene in ['eth', 'hotel']:
        # eth and hotel have different coordinate system than ucy data
        traj_image2world[:, :, [0, 1]] = traj_image2world[:, :, [1, 0]]
    traj_image2world = traj_image2world / resize
    traj_image2world = F.pad(input=traj_image2world, pad=(0, 1, 0, 0), mode='constant', value=1)
    traj_image2world = traj_image2world.reshape(-1, 3)
    traj_image2world = torch.matmul(homo_mat[scene], traj_image2world.T).T
    traj_image2world = traj_image2world / traj_image2world[:, 2:]
    traj_image2world = traj_image2world[:, :2]
    traj_image2world = traj_image2world.view_as(image_coords)
    return traj_image2world

def create_images_dict(data, image_path, image_file='reference.jpg', factor=1, multiproc=True):
    images = {}
    if multiproc:
        with multiprocessing.Pool() as pool:
            scene_list = [(scene, image_path, image_file, factor) for scene in data.sceneId.unique()]
            im_out = pool.starmap(read_and_resize_image, scene_list)
            for i in im_out:
                scene, im = i
                images[scene] = im
    else:
        for scene in data.sceneId.unique():
            _, im = read_and_resize_image(scene, image_path, image_file, factor)
            images[scene] = im
    return images

def read_and_resize_image(scene, image_path, image_file='reference.jpg', factor=1):
    if image_file == 'oracle.png':
        im = cv2.imread(os.path.join(image_path, scene, image_file), 0)
        if factor != 1:
            im = cv2.resize(im, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
    else:
        im = cv2.imread(os.path.join(image_path, scene, image_file))
        if factor != 1:
            im = cv2.resize(im, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    
    return scene, im

def rot(df, image, k=1, factor=1, scene=None):
    '''
    Rotates image and coordinates counter-clockwise by k * 90° within image origin
    :param df: Pandas DataFrame with at least columns 'x' and 'y'
    :param image: PIL Image
    :param k: Number of times to rotate by 90°
    :return: Rotated Dataframe and image
    '''
    xy = df.copy()
    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape
    
    x0 /= factor
    y0 /= factor
    
    xy.loc()[:, 'x'] = xy['x'] - x0 / 2
    xy.loc()[:, 'y'] = xy['y'] - y0 / 2
    c, s = np.cos(-k * np.pi / 2), np.sin(-k * np.pi / 2)
    R = np.array([[c, s], [-s, c]])
    xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
    for i in range(k):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape
    
    x0 /= factor
    y0 /= factor
    
    xy.loc()[:, 'x'] = xy['x'] + x0 / 2
    xy.loc()[:, 'y'] = xy['y'] + y0 / 2
    
    if scene:
        return xy, image, scene
    else:
        return xy, image

def fliplr(df, image, factor=1, scene=None):
    '''
    Flip image and coordinates horizontally
    :param df: Pandas DataFrame with at least columns 'x' and 'y'
    :param image: PIL Image
    :return: Flipped Dataframe and image
    '''
    xy = df.copy()
    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape
    
    x0 /= factor
    y0 /= factor
    
    xy.loc()[:, 'x'] = xy['x'] - x0 / 2
    xy.loc()[:, 'y'] = xy['y'] - y0 / 2
    R = np.array([[-1, 0], [0, 1]])
    xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
    image = cv2.flip(image, 1)
    
    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape
    
    x0 /= factor
    y0 /= factor
    
    xy.loc()[:, 'x'] = xy['x'] + x0 / 2
    xy.loc()[:, 'y'] = xy['y'] + y0 / 2
    
    if scene:
        return xy, image, scene
    else:
        return xy, image

def augment_data(data, images, factor=1, multiproc=True, mode='all', method=['rot', 'flip']):
    '''
    Perform data augmentation
    :param data: Pandas df, needs x,y,metaId,sceneId columns
    :param image: dict, images
    :param multiproc: bool, use multi thread
    :param mode: ['all', 'split'], str, if split it will return split_df by same W,H
    :param method: ['rot', 'flip'], list, choose augmentation method to use
    :return: (list, dict), Dataframe and image
    '''
    rot_data1, rot_data2 = [], []
    splitlist = []
    
    if len(method) == 0:
        print('Warning: Data augmentation is not used')
    
    # Rotate by 90°, 180°, 270°
    if 'rot' in method:
        data_ = data.copy()  # data without rotation, used so rotated data can be appended to original df
        k2rot = {1: '_rot90', 2: '_rot180', 3: '_rot270'}
        
        for k in k2rot.keys():
            metaId_max = data['metaId'].max()
            if multiproc:
                def create_list(scene):
                    return (data_[data_.sceneId == scene], images[scene], k, factor, scene)
                
                def data_append(data_rot, im, scene):
                    nonlocal rot_data1, rot_data2, images
                    rot_angle = k2rot[k]
                    images[scene + rot_angle] = im
                    data_rot['sceneId'] = scene + rot_angle
                    data_rot['metaId'] = data_rot['metaId'] + (metaId_max * k) + 1
                    # Note: Don't do pd.concat here
                    if mode == 'all' or k == 2:
                        rot_data2.append(data_rot)
                    else:
                        rot_data1.append(data_rot)
                
                # Use main process cpu with thread, instead of Pool(), due to I/O cost
                with multiprocessing.pool.ThreadPool() as pool:
                    scene_list = [[scene] for scene in data_.sceneId.unique()]
                    scene_list = pool.starmap(create_list, scene_list, chunksize=10)
                    rot_out = pool.starmap(rot, scene_list, chunksize=10)
                    pool.starmap(data_append, rot_out, chunksize=10)
            else:
                for scene in data_.sceneId.unique():
                    data_rot, im = rot(data_[data_.sceneId == scene], images[scene], k, factor)
                    rot_angle = k2rot[k]
                    images[scene + rot_angle] = im
                    
                    data_rot['sceneId'] = scene + rot_angle
                    data_rot['metaId'] = data_rot['metaId'] + (metaId_max * k) + 1
                    # Don't do pd.concat to simulate append in loop, it will be very slow
                    if mode == 'all' or k == 2:
                        rot_data2.append(data_rot)
                    else:
                        rot_data1.append(data_rot)
        
        data = pd.concat([data, pd.concat(rot_data2, ignore_index=True)], ignore_index=True)
        if mode == 'split':
            rot_data = pd.concat(rot_data1, ignore_index=True)
            splitlist = [data, rot_data]
        else:
            splitlist = [data]
        
        # Clear list
        rot_data1, rot_data2 = [], []
    else:
        splitlist = [data]
    
    if 'flip' in method:
        for idx, d in enumerate(splitlist):
            metaId_max = d['metaId'].max()
            if multiproc:
                def create_list(scene):
                    return (d[d.sceneId == scene], images[scene], factor, scene)
                
                def data_append(data_flip, im_flip, scene):
                    nonlocal rot_data1, rot_data2, images
                    data_flip['sceneId'] = data_flip['sceneId'] + '_fliplr'
                    data_flip['metaId'] = data_flip['metaId'] + (metaId_max * (idx+1)) + 1
                    images[scene + '_fliplr'] = im_flip
                    # Note: Don't do pd.concat here
                    if idx == 0:
                        rot_data2.append(data_flip)
                    else:
                        rot_data1.append(data_flip)
                
                # Use main process cpu with thread, instead of Pool(), due to I/O cost
                with multiprocessing.pool.ThreadPool() as pool:
                    scene_list = [[scene] for scene in d.sceneId.unique()]
                    scene_list = pool.starmap(create_list, scene_list, chunksize=10)
                    flip_out = pool.starmap(fliplr, scene_list, chunksize=10)
                    pool.starmap(data_append, flip_out, chunksize=10)
            else:
                for scene in d.sceneId.unique():
                    data_flip, im_flip = fliplr(d[d.sceneId == scene], images[scene], factor)
                    data_flip['sceneId'] = data_flip['sceneId'] + '_fliplr'
                    data_flip['metaId'] = data_flip['metaId'] + (metaId_max * (idx+1)) + 1
                    # Don't do pd.concat to simulate append in loop, it will be very slow
                    if idx == 0:
                        rot_data2.append(data_flip)
                    else:
                        rot_data1.append(data_flip)
                    images[scene + '_fliplr'] = im_flip
        
        data = pd.concat([data, pd.concat(rot_data2, ignore_index=True)], ignore_index=True)
        
        if mode == 'split':
            rot_data = pd.concat([rot_data, pd.concat(rot_data1, ignore_index=True)], ignore_index=True)
            splitlist = [data, rot_data]
        else:
            splitlist = [data]
            
    return splitlist, images
