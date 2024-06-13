import os
import shutil
import pandas as pd

from JAAD_origin import JAAD

data_path = f"D:\JAADDataset\JAAD-JAAD_2.0"
vid_path = f"D:\JAADDataset\JAAD_clips"
obs_len = 15
pred_len = 45

def run(split):
    traj_data_opts = {
        'fstride': 1,
        'sample_type': 'all',
        'height_rng': [0, float('inf')],
        'squarify_ratio': 0,
        'data_split_type': 'default',  # kfold, random, default
        'seq_type': 'trajectory',
        'min_track_size': 61,
        'random_params': {'ratios': None, 'val_data': True, 'regen_data': True},
        'kfold_params': {'num_folds': 5, 'fold': 1}
    }

    traj_model_opts = {
        'track_overlap': 0.5,
        'observe_length': obs_len,
        'predict_length': pred_len,
        'type': ['bbox', 'center']
    }
    imdb = JAAD(data_path=data_path)
    
    beh_seq = imdb.generate_data_trajectory_sequence(split, **traj_data_opts)
    data = get_traj_data(beh_seq, **traj_model_opts)
    #print(beh_seq.keys())
    #print(data.keys())
    
    return data

def get_traj_tracks(dataset, data_types, observe_length, predict_length, overlap):
    #  Calculates the overlap in terms of number of frames
    seq_length = observe_length + predict_length
    overlap_stride = observe_length if overlap == 0 else int((1 - overlap) * observe_length)
    overlap_stride = 1 if overlap_stride < 1 else overlap_stride

    #  Check the validity of keys selected by user as data type
    d = {}
    for dt in data_types:
        try:
            d[dt] = dataset[dt]
        except:# KeyError:
            raise KeyError('Wrong data type is selected %s' % dt)
    
    d['image'] = dataset['image']
    #  Sample tracks from sequneces
    for k in d.keys():
        tracks = []
        for track in d[k]:
            for i in range(0, len(track) - seq_length + 1, overlap_stride):
                tracks.append(track[i:i + seq_length])
        d[k] = tracks
    
    return d

def get_traj_data(data, **model_opts):
    opts = {
        'track_overlap': 0.5,
        'observe_length': 15,
        'predict_length': 45,
        'type': ['bbox', 'center']
    }
    
    for key, value in model_opts.items():
        assert key in opts.keys(), 'wrong data parameter %s' % key
        opts[key] = value

    observe_length = opts['observe_length']
    predict_length = opts['predict_length']
    data_types = opts['type']
    data_tracks = get_traj_tracks(data, data_types, observe_length, opts['predict_length'], opts['track_overlap'])
    return data_tracks

def get_spilt(split):
    dataspilt = run(split)
    data_list = {}
    param = ['frame', 'trackId', 'x', 'y', 'w', 'h', 'sceneId', 'metaId']
    for i in param:
        data_list[i] = []
    
    for i in range(len(dataspilt['image'])):
        for j in range(obs_len + pred_len):
            temp = []
            frame = os.path.basename(dataspilt['image'][i][obs_len-1]).replace('.png', '') # (choose now as ref. frame)
            vid_num = os.path.basename(os.path.dirname(dataspilt['image'][i][obs_len-1]))   
            bbox = dataspilt['bbox'][i][j]
            data_list[param[0]].append(frame) # frame
            data_list[param[1]].append(i) # trackId (only use for rename)
            data_list[param[2]].append(dataspilt['center'][i][j][0]) # x (center)
            data_list[param[3]].append(dataspilt['center'][i][j][1]) # y (center)
            data_list[param[4]].append(bbox[2] - bbox[0]) # w (x1 = x - w/2, x2 = x + w/2)
            data_list[param[5]].append(bbox[3] - bbox[1]) # h (y1 = y - h/2, y2 = y + h/2)
            data_list[param[6]].append(f"{vid_num}-{frame}") # sceneId
            data_list[param[7]].append(i) # metaId
     
    return pd.DataFrame(data_list)

def extract_img(pkl_file):
    df = pd.read_pickle(pkl_file)
    vid = {}
    for name in df['sceneId'].unique():
        vid_num, frame = name.split('-')
        
        if not vid_num in vid:
            vid[vid_num] = []
        
        vid[vid_num].append(frame)
    
    for i in vid.keys():
        os.mkdir("./frame")
        os.system("ffmpeg -i {} -start_number 0 -qscale:v 2 {}/%05d.jpg".format(os.path.join(vid_path, f"{i}.mp4"), "frame"))
        
        sets = pkl_file.replace('_pie.pkl', '')
        if not os.path.exists(f"./{sets}"):
            os.mkdir(f"./{sets}")
        
        for index in vid[i]:
            if not os.path.exists(f"./{sets}/{i}-{index}"):
                os.mkdir(f"./{sets}/{i}-{index}")
                shutil.copy(f"./frame/{index}.jpg", f"./{sets}/{i}-{index}/reference.jpg")
        
        # Delete video extract temp img
        if os.path.exists("./frame"):
            shutil.rmtree("./frame", ignore_errors=True)

get_spilt('train').to_pickle("train_jaad.pkl")
get_spilt('val').to_pickle("val_jaad.pkl")
get_spilt('test').to_pickle("test_jaad.pkl")
extract_img("train_jaad.pkl")
extract_img("val_jaad.pkl")
extract_img("test_jaad.pkl")
