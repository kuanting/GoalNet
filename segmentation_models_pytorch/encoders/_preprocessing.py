import numpy as np

def preprocess_input(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):
    if input_space == "BGR":
        x = x[..., ::-1].copy()
    
    # for inplace operation
    x = x.astype('float64')
    
    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x /= 255.0

    if mean is not None:
        mean = np.array(mean)
        x -= mean

    if std is not None:
        std = np.array(std)
        x /= std

    return x
