import numpy as np
import keras

#Distance defined as (1 - average of probabilities data points from one set appears in other set)
def magic_distance(set_real, set_pred, p = 1000):
    set_pred_ = np.expand_dims(set_pred, axis=-1)
    set_real_ = np.expand_dims(set_real, axis=-1)
    dists = np.linalg.norm(set_pred_ - set_real_, axis = -1) ** (1/p)
    dists = dists.reshape((dists.shape[0], -1))
    
    result = (np.mean(dists, axis = -1) / np.amax(dists, axis = -1)) ** p
    return result
    
    

#need to initiate this model somehow at the start of training    
inception_model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=True) 
print('inception_model init')   
def inception_score(set_real, set_pred, splits=10):
    images = set_pred.repeat(10, axis = 1).repeat(10, axis = 2) [:,:299,:299]
    preds = inception_model.predict(images)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
      
    return scores