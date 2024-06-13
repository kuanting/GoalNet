import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import segmentation_models_pytorch as smp
from utils.modules import ASPP, LayerNorm2d, Attention_gate
from utils.SpatialSoftArgmax2d import SpatialSoftArgmax2d
from utils.preprocessing import augment_data, create_images_dict, create_gaussian_heatmap_template, create_dist_mat, pad, resize
from utils.dataloader import SceneDataset, scene_collate
from model_det.train import train, val
from model_det.test import evaluate

"""
modules                            | parameters |  flops   |
------------------------------------------------------------
feature_extract (down, ConvNeXt-T) |   27.819M  | 12.314G  |
feature_extract (up)               |    3.528M  | 10.456G  |
GoalNet (traj-encoder)             |    M  |  G  |
GoalNet (traj-decoder)             |    M  |  G  |
bbox_wh                            |    0.174M  |  0.174M  |
------------------------------------------------------------
Total                              |   M  | G  |
"""
# GoalNet (Det), for deterministic trajectory prediction
class TrajEncoder(nn.Module):
    def __init__(self, in_channels, channels=(32, 32, 64, 64, 64)):
        """
        Traj. encoder model
        :param in_channels: int, feature_classes + obs_len
        :param channels: list, hidden layer channels
        """
        super().__init__()
        self.stages = nn.ModuleList()
        
        # First block
        self.stages.append(nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1),
            LayerNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        ))
        
        # Subsequent blocks, each starting with Maxpool
        for i in range(len(channels)-1):
            self.stages.append(nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                LayerNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, stride=1, padding=1),
                LayerNorm2d(channels[i+1]),
                nn.ReLU(inplace=True)
            ))
        
        # Last Maxpool layer before passing the features into decoder
        self.stages.append(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        
        # init_weights after defined all layers
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Saves the feature maps Tensor of each layer into a list, as we will later need them again for the decoder
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

class TrajDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, output_len):
        """
        Traj. decoder models
        :param encoder_channels: list, encoder channels, used for skip connections
        :param decoder_channels: list, decoder channels
        :param output_len: int, pred_len
        :param traj: False or int, if False -> Goal and waypoint predictor, if int -> number of waypoints
        """
        super().__init__()
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        center_channels = encoder_channels[0]
        
        # The center layer (the layer with the smallest feature map size)
        self.center = nn.Sequential(
            ASPP(center_channels, center_channels*2),
            LayerNorm2d(center_channels*2),
            nn.ReLU(inplace=True)
        )
        
        # Determine the upsample channel dimensions
        upsample_channels_in = [center_channels*2] + decoder_channels[:-1]
        
        # Upsampling consists of bilinear upsampling + 3x3 Conv, here the 3x3 Conv is defined
        self.upsample_conv = [
            nn.Conv2d(in_channels_, out_channels_, kernel_size=3, stride=1, padding=1)
            for in_channels_, out_channels_ in zip(upsample_channels_in, decoder_channels)]
        self.upsample_conv = nn.ModuleList(self.upsample_conv)
        
        # Attention gate
        self.attention = [
            Attention_gate(F_g, F_l, (F_g + F_l) // 2)
            for F_g, F_l in zip(decoder_channels, encoder_channels)]
        self.attention = nn.ModuleList(self.attention)
        
        # Determine the input and output channel dimensions of each layer in the decoder
        # As we concat the encoded feature and decoded features we have to sum both dims
        in_channels = [enc + dec for enc, dec in zip(encoder_channels, decoder_channels)]
        
        self.decoder = [nn.Sequential(
            nn.Conv2d(in_channels_, out_channels_, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(out_channels_),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels_, out_channels_, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(out_channels_),
            nn.ReLU(inplace=True))
            for in_channels_, out_channels_ in zip(in_channels, decoder_channels)]
        self.decoder = nn.ModuleList(self.decoder)
        
        # Final 1x1 Conv prediction to get our heatmap logits (before softmax)
        self.predictor = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=output_len, kernel_size=1, stride=1, padding=0)
    
        # init_weights after defined all layers
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        # Takes in the list of feature maps from the encoder. Trajectory predictor in addition the goal and waypoint heatmaps
        features = features[::-1]  # reverse the order of encoded features, as the decoder starts from the smallest image
        center_feature = features[0]
        x = self.center(center_feature)
        for feature, module, upsample_conv, attention in zip(features[1:], self.decoder, self.upsample_conv, self.attention):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # bilinear interpolation for upsampling
            x = upsample_conv(x)                # 3x3 conv for upsampling
            feature = attention(x, feature)     # Attention gate
            x = torch.cat([x, feature], dim=1)  # concat encoder and decoder features
            x = module(x)                       # conv_block
        x = self.predictor(x)                   # last predictor layer
        return x

class FeatureExtract_trans(nn.Module):
    def __init__(self, out_channels=6, convnext_channels=[768, 384, 192, 96]):
        """
        ConvNeXt channels definition
        https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py#L160
        """
        super().__init__()
        self.upsample_conv = nn.ModuleList()
        self.scale = [2,2,2,4] # ConvNeXt: 4*2*2*2
        last_channel = convnext_channels[-1] // 2
        channels = convnext_channels + [last_channel]
        assert (last_channel > out_channels)
        
        for i in range(len(channels)-1):
            self.upsample_conv.append(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1)
            )
        
        self.head = nn.Conv2d(last_channel, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        for factor, upsample_conv in zip(self.scale, self.upsample_conv):
            x = F.interpolate(x, scale_factor=factor, mode='bilinear', align_corners=False)
            x = upsample_conv(x)
        x = self.head(x)
        x = F.softmax(x, dim=1)
        return x

class GoalNetTorch(nn.Module):
    def __init__(self, obs_len, pred_len, segmentation_model, feature_extract=False, feature_classes=6, encoder_channels=[], decoder_channels=[]):
        """
        Complete GoalNet Architecture
        :param obs_len: int, observed timesteps
        :param pred_len: int, predicted timesteps
        :param segmentation_model: str, filepath to pretrained segmentation model
        :param feature_classes: int, number of feature dims
        :param encoder_channels: list, encoder channel structure
        :param decoder_channels: list, decoder channel structure
        """
        super().__init__()
        
        if feature_extract:
            # Note: If your input shape isn't (W,H = even, even), please check the shape after do feature_extract
            from torchvision.models import convnext_tiny
            self.feature_extract_ = convnext_tiny(weights='IMAGENET1K_V1')
            self.feature_extract_.avgpool = nn.Identity()
            self.feature_extract_.classifier = nn.Identity()
            self.feature_extract_trans = FeatureExtract_trans(feature_classes)
        elif segmentation_model is not None:
            #self.feature_extract_ = smp.Unet(encoder_name="resnet101", encoder_weights=None, in_channels=3, classes=feature_classes, activation='softmax')
            self.feature_extract_ = torch.load(segmentation_model)
        else:
            self.feature_extract_ = nn.Identity()
        
        self.encoder = TrajEncoder(in_channels=feature_classes + obs_len, channels=encoder_channels)
        self.decoder = TrajDecoder(encoder_channels, decoder_channels, output_len=pred_len)
        self.softargmax_ = SpatialSoftArgmax2d(normalized_coordinates=False)
    
    def feature_extract(self, image):
        return self.feature_extract_(image)
    
    # Forward pass for trajectory decoder
    def pred_traj(self, features):
        traj = self.decoder(features)
        return traj
    
    # Forward pass for feature encoder, returns list of feature maps
    def pred_features(self, x):
        features = self.encoder(x)
        return features
    
    # Softmax for Image data as in dim=NxCxHxW, returns softmax image shape=NxCxHxW
    def softmax(self, x):
        return nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)
    
    # Softargmax for Image data as in dim=NxCxHxW, returns 2D coordinates=Nx2
    def softargmax(self, output):
        # [N,C,H,W] -> [N,C,2]
        return self.softargmax_(output)
    
    def sigmoid(self, output):
        return torch.sigmoid(output)
    
class GoalNet:
    def __init__(self, obs_len, pred_len, params):
        """
        :param obs_len: observed timesteps
        :param pred_len: predicted timesteps
        :param params: dictionary with hyperparameters
        """
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.division_factor = 2 ** len(params['encoder_channels'])

        self.model = GoalNetTorch(obs_len=obs_len,
                               pred_len=pred_len,
                               segmentation_model=params['segmentation_model'],
                               feature_extract=params['feature_extract'],
                               feature_classes=params['feature_classes'],
                               encoder_channels=params['encoder_channels'],
                               decoder_channels=params['decoder_channels'])
        # function to normalize images
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet101', 'imagenet')
    
    def train(self, train_data, test_data, params, train_image_path, test_image_path, experiment_name, batch_size=8, device=None, dataset_name=None, val_data=None, val_image_path=None):
        """
        Train function
        :param train_data: pd.df, train data
        :param val_data: pd.df, val data
        :param test_data: pd.df, test data
        :param params: dictionary with training hyperparameters
        :param train_image_path: str, filepath to train images
        :param val_image_path: str, filepath to val images
        :param test_image_path: str, filepath to test images
        :param experiment_name: str, arbitrary name to name weights file
        :param batch_size: int, batch size
        :param device: torch.device, if None -> 'cuda' if torch.cuda.is_available() else 'cpu'
        :return:
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        obs_len = self.obs_len
        pred_len = self.pred_len
        total_len = pred_len + obs_len
        
        print('Preprocess data')
        dataset_name = dataset_name.lower()
        image_file_name = 'reference.jpg'
        splitby = 'sceneId'
        mode = 'all'
        collate_fn = scene_collate
        loader_batch_size = 1
        
        if params['matrics'] == 'mse':
            params['bbox_wh_model'] = None # Don't eval bbox MSE during training
            avg_matrics, f_matrics = "CMSE", "CFMSE"
        else:
            avg_matrics, f_matrics = "ADE", "FDE"
        
        # ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
        if dataset_name == 'eth':
            image_file_name = 'oracle.png'
            self.homo_mat = {}
            for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
                self.homo_mat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt')).to(device)
            seg_mask = True
        else:
            self.homo_mat = None
            seg_mask = False
        
        # Load train images
        train_images = create_images_dict(train_data, image_path=train_image_path, image_file=image_file_name, factor=params['resize'])
        
        # Augment train data and images & Initialize dataloaders
        # Use augment_data will take 4*2x memory usage (0, 90, 180, 270 * h_flip)
        train_loader = []
        df_train, train_images = augment_data(train_data, train_images, factor=params['resize'], mode=mode, method=params['data_augmentation'])
        for df in df_train:
            train_dataset = SceneDataset(df, resize=params['resize'], total_len=total_len, splitby=splitby)
            loader = DataLoader(train_dataset, batch_size=loader_batch_size, collate_fn=collate_fn, shuffle=True)
            train_loader.append(loader)
        
        # Pad images, make sure that image shape is divisible by 32, for structure like UNet
        pad(train_images, division_factor=self.division_factor)  
        
        # Load test scene images
        test_images = create_images_dict(test_data, image_path=test_image_path, image_file=image_file_name, factor=params['resize'])
        
        test_dataset = SceneDataset(test_data, resize=params['resize'], total_len=total_len, splitby=splitby)
        test_loader = DataLoader(test_dataset, batch_size=loader_batch_size, collate_fn=collate_fn)
        
        # Pad images, make sure that image shape is divisible by 32, for structure like UNet
        pad(test_images, division_factor=self.division_factor)
        
        # For ReduceLROnPlateau lr_scheduler
        if val_data is not None:
            # Load val scene images
            val_images = create_images_dict(val_data, image_path=val_image_path, image_file=image_file_name, factor=params['resize'])
            
            val_dataset = SceneDataset(val_data, resize=params['resize'], total_len=total_len, splitby=splitby)
            val_loader = DataLoader(val_dataset, batch_size=loader_batch_size, collate_fn=collate_fn)
            
            # Pad images, make sure that image shape is divisible by 32, for structure like UNet
            pad(val_images, division_factor=self.division_factor)
        
        model = self.model.to(device)
        
        # Freeze feature extract model
        for param in model.feature_extract_.parameters():
            param.requires_grad = False
        
        #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params["learning_rate"])
        # Optimizer will still update freeze weight without auto_grad, but the results will be slightly better
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        criterion = nn.BCEWithLogitsLoss()
        if val_data is not None:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=3, min_lr=1e-10, verbose=1)
        
        # Create template
        size = int(4200 * params['resize'])
        
        input_template = create_dist_mat(size=size)
        input_template = torch.Tensor(input_template).to(device)
        
        gt_template = create_gaussian_heatmap_template(size=size, kernlen=params['kernlen'], nsig=params['nsig'], normalize=False)
        gt_template = torch.Tensor(gt_template).to(device)
        
        best_test_ADE = 99999999999999
        
        self.train_ADE = []
        self.train_FDE = []
        self.val_ADE = []
        self.val_FDE = []
        
        print('Start training')
        for e in tqdm(range(params['num_epochs']), desc='Epoch'):
            for i in range(len(train_loader)):
                train_ADE, train_FDE, train_loss = train(model, train_loader[i], train_images, e, obs_len, pred_len, 
                                                        batch_size, params, gt_template, device, input_template, 
                                                        optimizer, criterion, dataset_name, self.homo_mat, self.preprocessing_fn, seg_mask)
                self.train_ADE.append(train_ADE)
                self.train_FDE.append(train_FDE)
            
            # update lr
            if val_data is not None:
                val_loss = val(model, val_loader, val_images, obs_len, pred_len, batch_size, params, gt_template, device, input_template, criterion, self.preprocessing_fn, dataset_name, seg_mask)
                lr_scheduler.step(val_loss)
            
            val_ADE, val_FDE, _ = evaluate(model, test_loader, test_images, obs_len, pred_len, batch_size, device, input_template, 
                                            params, dataset_name, self.homo_mat, self.preprocessing_fn, seg_mask)
            self.val_ADE.append(val_ADE)
            self.val_FDE.append(val_FDE)
            print(f'\nEpoch {e}: \nVal {avg_matrics}: {val_ADE} \nVal {f_matrics}: {val_FDE}')
            
            if val_ADE < best_test_ADE:
                print(f'Best Epoch {e}: \nVal {avg_matrics}: {val_ADE} \nVal {f_matrics}: {val_FDE}')
                torch.save(model.state_dict(), f'pretrained_models/{experiment_name}_{e}_weights.pt')
                best_test_ADE = val_ADE
    
    def evaluate(self, data, params, image_path, batch_size=8, rounds=1, device=None, dataset_name=None):
        """
        Val function
        :param data: pd.df, val data
        :param params: dictionary with training hyperparameters
        :param image_path: str, filepath to val images
        :param batch_size: int, batch size
        :param rounds: int, number of epochs to evaluate
        :param device: torch.device, if None -> 'cuda' if torch.cuda.is_available() else 'cpu'
        :return:
        """
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        obs_len = self.obs_len
        pred_len = self.pred_len
        total_len = pred_len + obs_len
        
        print('Preprocess data')
        dataset_name = dataset_name.lower()
        image_file_name = 'reference.jpg'
        splitby = 'sceneId'
        collate_fn = scene_collate
        loader_batch_size = 1
        
        # ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
        if dataset_name == 'eth':
            image_file_name = 'oracle.png'
            self.homo_mat = {}
            for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
                self.homo_mat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt')).to(device)
            seg_mask = True
        else:
            self.homo_mat = None
            seg_mask = False
                
        test_images = create_images_dict(data, image_path=image_path, image_file=image_file_name, factor=params['resize'])
        
        test_dataset = SceneDataset(data, resize=params['resize'], total_len=total_len, splitby=splitby)
        test_loader = DataLoader(test_dataset, batch_size=loader_batch_size, collate_fn=collate_fn)
        
        # Pad images, make sure that image shape is divisible by 32, for structure like UNet
        pad(test_images, division_factor=self.division_factor)
        
        model = self.model.to(device)
        
        # Create template
        size = int(4200 * params['resize'])
        
        input_template = torch.Tensor(create_dist_mat(size=size)).to(device)
        
        self.eval_ADE = []
        self.eval_FDE = []
        self.eval_MSE = {'MSE15': [], 'MSE30': [], 'MSE45':[]}
        
        print('Start testing')
        for e in tqdm(range(rounds), desc='Round'):
            test_ADE, test_FDE, test_MSE = evaluate(model, test_loader, test_images, obs_len, pred_len, batch_size, device, input_template,
                                                    params, dataset_name, self.homo_mat, self.preprocessing_fn, seg_mask)
            if params['matrics'] == 'ade':
                print(f'Round {e}: \nTest ADE: {test_ADE} \nTest FDE: {test_FDE}')
            elif params['matrics'] == 'mse':
                print(f'Round {e}: \nTest CMSE: {test_ADE} \nTest CFMSE: {test_FDE}')
                print(f'Test MSE(0.5): {test_MSE["MSE15"]} \nTest MSE(1.0): {test_MSE["MSE30"]} \nTest MSE(1.5): {test_MSE["MSE45"]}')
                self.eval_MSE["MSE15"].append(test_MSE["MSE15"])
                self.eval_MSE["MSE30"].append(test_MSE["MSE30"])
                self.eval_MSE["MSE45"].append(test_MSE["MSE45"])
            
            self.eval_ADE.append(test_ADE)
            self.eval_FDE.append(test_FDE)
        
        msg = f'\n\nAverage performance over {rounds} rounds:'
        if params['matrics'] == 'ade':
            msg += f' \nTest ADE: {sum(self.eval_ADE) / len(self.eval_ADE)} \nTest FDE: {sum(self.eval_FDE) / len(self.eval_FDE)}'
        elif params['matrics'] == 'mse':
            msg += f' \nTest CMSE: {sum(self.eval_ADE) / len(self.eval_ADE)} \nTest CFMSE: {sum(self.eval_FDE) / len(self.eval_FDE)}'
            if len(self.eval_MSE["MSE15"]) > 0:
                msg += f' \nTest MSE(0.5): {sum(self.eval_MSE["MSE15"]) / len(self.eval_MSE["MSE15"])}'
                msg += f' \nTest MSE(1.0): {sum(self.eval_MSE["MSE30"]) / len(self.eval_MSE["MSE30"])}'
                msg += f' \nTest MSE(1.5): {sum(self.eval_MSE["MSE45"]) / len(self.eval_MSE["MSE45"])}'
        print(msg)
    
    def calc_flop(self, params, device=None):
        """
        Calculate model params and flops
        https://github.com/facebookresearch/fvcore
        :param device: torch.device, if None -> 'cuda' if torch.cuda.is_available() else 'cpu'
        """
        import fvcore
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        obs_len = self.obs_len
        model = self.model.to(device)
        
        # Freeze feature extract model
        for param in model.feature_extract_.parameters():
            param.requires_grad = False
        
        # 1920*1080 / 4, pad 32 = 480*288
        inputs = torch.rand(1, 3, 480, 288).to(device)
        flops = FlopCountAnalysis(model.feature_extract_, inputs)
        print('feature_extract (down):', flop_count_table(flops))
        
        inputs = model.feature_extract(inputs)
        flops = FlopCountAnalysis(model.feature_extract_trans, inputs)
        print('feature_extract (up):', flop_count_table(flops))
        
        inputs = torch.rand(1, obs_len+params['feature_classes'], 480, 288).to(device)
        flops = FlopCountAnalysis(model.encoder, inputs)
        print('GoalNet (traj-encoder):', flop_count_table(flops))
        
        inputs = model.encoder(inputs)
        flops = FlopCountAnalysis(model.traj_decoder, inputs)
        print('GoalNet (traj-decoder):', flop_count_table(flops))
        
        total_params = sum(param.numel() for param in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f'Trainable params: {trainable_params} in total {total_params}')
    
    def load(self, path):
        print(self.model.load_state_dict(torch.load(path)))
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
