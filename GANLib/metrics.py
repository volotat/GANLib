import numpy as np

def magic_distance(set_real, set_pred, p = 1000):
    set_pred_ = np.expand_dims(set_pred, axis=-1)
    set_real_ = np.expand_dims(set_real, axis=-1)
    dists = np.linalg.norm(set_pred_ - set_real_, axis = -1) ** (1/p)
    dists = dists.reshape((dists.shape[0], -1))
    
    result = (np.mean(dists, axis = -1) / np.amax(dists, axis = -1)) ** p
    return result