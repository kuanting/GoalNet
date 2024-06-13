import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class SceneDataset(Dataset):
    def __init__(self, data, resize, total_len, splitby='sceneId'):
        """ Dataset that contains the trajectories of one scene as one element in the list. It doesn't contain the
        images to save memory.
        :params data (pd.DataFrame): Contains all trajectories
        :params resize (float): image resize factor, to also resize the trajectories to fit image scale
        :params total_len (int): total time steps, i.e. obs_len + pred_len
        """
        
        self.trajectories, self.meta, self.scene_list = self.split_trajectories(data, total_len, splitby=splitby)
        self.trajectories = self.trajectories * resize
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        meta = self.meta[idx]
        scene = self.scene_list[idx]
        return trajectory, meta, scene
    
    def split_trajectories(self, data, total_len, splitby='sceneId'):
        trajectories = []
        meta = []
        scene_list = []
        for meta_id, meta_df in tqdm(data.groupby(splitby, as_index=False), desc='Prepare Dataset'):
            trajectories.append(meta_df[['x', 'y']].to_numpy().astype('float32').reshape(-1, total_len, 2))
            meta.append(meta_df)
            scene_list.append(meta_df.iloc()[0:1].sceneId.item())
        
        return np.array(trajectories, dtype=object), meta, scene_list

def scene_collate(batch):
    _batch = batch[0]
    # Note: If trajectories.shape[0] > 1, then no astype is OK, else not
    # trajectories, meta, scene
    return torch.Tensor(_batch[0].astype('float32')), _batch[1], _batch[2]

def meta_collate(batch):
    scene_data = { "trajectory" : [], "meta" : [], "scene" : []}
    for _batch in batch:
        # Note: If trajectories.shape[0] > 1, then no astype is OK, else not
        # trajectories, meta, scene
        scene_data["trajectory"].append(torch.Tensor(_batch[0].astype('float32')))
        scene_data["meta"].append(_batch[1])
        scene_data["scene"].append(_batch[2])
    
    return scene_data