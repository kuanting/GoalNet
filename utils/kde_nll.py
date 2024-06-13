import numpy as np
from scipy.stats import gaussian_kde

def compute_kde_nll(pred_traj, target_traj):
    '''
    pred_traj: (batch, T, K, 2/4)
    '''
    kde_ll = 0.
    log_pdf_lower_bound = -20
    batch_size, T, _, _ = pred_traj.shape
    
    for batch_num in range(batch_size):
        for timestep in range(T):
            try:
                kde = gaussian_kde(pred_traj[batch_num, timestep, :, ].T)
                pdf = np.clip(kde.logpdf(target_traj[batch_num, timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf / (T * batch_size)
            except np.linalg.LinAlgError:
                kde_ll = np.nan
    return -kde_ll
