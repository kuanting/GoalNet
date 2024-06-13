import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import segmentation_models_pytorch as smp
from utils.modules import ASPP, LayerNorm2d, Attention_gate
from utils.SpatialSoftArgmax2d import SpatialSoftArgmax2d, spatial_expectation2d
from utils.preprocessing import augment_data, create_images_dict, create_gaussian_heatmap_template, create_dist_mat, pad, resize
from utils.dataloader import SceneDataset, scene_collate, meta_collate
from model.test import evaluate
from utils.visualize import painter1, painter2

"""
modules                            | parameters |  flops   |
------------------------------------------------------------
feature_extract (down, ConvNeXt-T) |   27.819M  | 12.314G  |
feature_extract (up)               |    3.528M  | 10.456G  |
GoalNet (encoder)                  |    0.287M  |  2.422G  |
GoalNet (goal-decoder)             |    0.915M  |  9.167G  |
GoalNet (traj-decoder)             |    0.929M  |  9.232G  |
bbox_wh                            |    0.174M  |  0.174M  |
------------------------------------------------------------
Total                              |   33.652M  | 43.591G  |
"""
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
        
        # Subsequent blocks, each starting with 2x2 conv_block(stride=2)
        for i in range(len(channels)-1):
            self.stages.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i], kernel_size=2, stride=2, padding=0),
                LayerNorm2d(channels[i]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                LayerNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, stride=1, padding=1),
                LayerNorm2d(channels[i+1]),
                nn.ReLU(inplace=True)
            ))
        
        # Last 2x2 conv_block(stride=2) layer before passing the features into decoder
        self.stages.append(nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], kernel_size=2, stride=2, padding=0),
            LayerNorm2d(channels[-1]),
            nn.ReLU(inplace=True)
        ))
        
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
    def __init__(self, encoder_channels, decoder_channels, output_len, traj=False):
        """
        Traj. decoder models
        :param encoder_channels: list, encoder channels, used for skip connections
        :param decoder_channels: list, decoder channels
        :param output_len: int, pred_len
        :param traj: False or int, if False -> Goal and waypoint predictor, if int -> number of waypoints
        """
        super().__init__()
        
        # The trajectory decoder takes in addition the conditioned goal and waypoints as an additional image channel
        if traj:
            encoder_channels = [channel+traj for channel in encoder_channels]
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
        
        # init_weights after defined all layers
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        for factor, upsample_conv in zip(self.scale, self.upsample_conv):
            x = F.interpolate(x, scale_factor=factor, mode='bilinear', align_corners=False)
            x = upsample_conv(x)
        x = self.head(x)
        x = F.softmax(x, dim=1)
        return x

class GoalNetTorch(nn.Module):
    def __init__(self, obs_len, pred_len, segmentation_model, feature_extract=False, feature_classes=6, encoder_channels=[], decoder_channels=[], waypoints=1):
        """
        Complete GoalNet Architecture
        :param obs_len: int, observed timesteps
        :param pred_len: int, predicted timesteps
        :param segmentation_model: str, filepath to pretrained segmentation model
        :param feature_classes: int, number of feature dims
        :param encoder_channels: list, encoder channel structure
        :param decoder_channels: list, decoder channel structure
        :param waypoints: int, number of waypoints
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
        
        self.goal_decoder = TrajDecoder(encoder_channels, decoder_channels, output_len=pred_len)
        self.traj_decoder = TrajDecoder(encoder_channels, decoder_channels, output_len=pred_len, traj=waypoints)
        
        self.softargmax_ = SpatialSoftArgmax2d(normalized_coordinates=False)
    
    def feature_extract(self, image):
        return self.feature_extract_(image)
    
    # Forward pass for goal decoder
    def pred_goal(self, features):
        goal = self.goal_decoder(features)
        return goal
    
    # Forward pass for trajectory decoder
    def pred_traj(self, features):
        traj = self.traj_decoder(features)
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
    
    def softargmax_on_softmax_map(self, x):
        # As input a batched image where softmax is already performed (not logits)
        # [N,C,H,W] -> [N,C,2]
        return spatial_expectation2d(x, normalized_coordinates=False)

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
                               decoder_channels=params['decoder_channels'],
                               waypoints=len(params['waypoints']))
        # function to normalize images
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet101', 'imagenet')
        
    def evaluate(self, data, params, image_path, batch_size=8, num_goals=20, num_traj=1, rounds=1, device=None, dataset_name=None):
        """
        Val function
        :param data: pd.df, val data
        :param params: dictionary with training hyperparameters
        :param image_path: str, filepath to val images
        :param batch_size: int, batch size
        :param num_goals: int, number of goals per trajectory (K_e)
        :param num_traj: int, number of trajectory per goal (K_a)
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
        
        # Data split method
        if params['train_mode'] == 'scene':
            splitby = 'sceneId'
            collate_fn = scene_collate
            loader_batch_size = 1
        elif params['train_mode'] == 'trajectory':
            splitby = 'metaId'
            collate_fn = meta_collate
            loader_batch_size = batch_size
        else:
            raise ValueError(f'{params["train_mode"]} test mode is not supported')
        
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
            test_ADE, test_FDE, test_MSE, future_samples, bbox = evaluate(model, test_loader, test_images, num_goals, num_traj, obs_len, pred_len, batch_size, device, input_template,
                                                    params, dataset_name, self.homo_mat, self.preprocessing_fn, seg_mask, use_TTST=params['use_TTST'],
                                                    use_CWS=params['use_CWS'], rel_thresh=params['rel_threshold'], CWS_params=params['CWS_params'])
            painter1(image_path + '/' + data.iloc[0]['sceneId'] + '/reference.jpg', data, future_samples)
            painter2(image_path + '/' + data.iloc[0]['sceneId'] + '/reference.jpg', data, bbox)
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
        print('GoalNet (encoder):', flop_count_table(flops))
        
        inputs = model.encoder(inputs)
        flops = FlopCountAnalysis(model.goal_decoder, inputs)
        print('GoalNet (goal-decoder):', flop_count_table(flops))
        
        inputs = [torch.rand(x.shape[0], x.shape[1]+1, x.shape[2], x.shape[3]).to(device) for x in inputs]
        flops = FlopCountAnalysis(model.traj_decoder, inputs)
        print('GoalNet (traj-decoder):', flop_count_table(flops))
        
        total_params = sum(param.numel() for param in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f'Trainable params: {trainable_params} in total {total_params}')
    
    def load(self, path):
        print(self.model.load_state_dict(torch.load(path)))
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
