import numpy as np
import random
import torch
import os
from sklearn.metrics.pairwise import rbf_kernel
from numpy.linalg import norm as npnorm



def seed_all(seed):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def rbf_kernel_estimator(
        test_Z: np.ndarray,
        recal_Z: np.ndarray,
        recal_epsilon:np.ndarray,
        quants,
        wid = 1E-1,
):
    
    assert isinstance(test_Z, np.ndarray)
    assert isinstance(recal_Z, np.ndarray)
    assert isinstance(recal_epsilon, np.ndarray)
    
    
    assert len(test_Z.shape) == 2
    assert len(recal_Z.shape) == 2
    assert len(recal_epsilon.shape) == 1

    # quants should be a rising list
    assert npnorm(np.sort(quants) - quants) < 1E-6


    assert len(quants.shape) == 1
    
    indices = np.argsort(recal_epsilon)
    
    sorted_epsi = recal_epsilon[indices]

    sorted_recal_Z = recal_Z[indices]
       
    dist_mat = rbf_kernel(test_Z/wid, sorted_recal_Z/wid, gamma = 1)
     
    summation_matform = np.triu(np.ones((len(recal_Z), len(recal_Z))))
 
    aggregated_dist_mat = np.matmul(dist_mat, summation_matform)

    empirical_quantiles = aggregated_dist_mat / aggregated_dist_mat[:, -1:]


    quantiles_unsquze = empirical_quantiles.reshape(empirical_quantiles.shape + (-1,))

    tf_mat = quantiles_unsquze <= quants
    

    harvest_ids = np.clip(np.transpose(tf_mat.sum(axis=1), (1, 0)), a_max = len(recal_Z)-1, a_min = -1)

    return sorted_epsi[harvest_ids]          # shape (len(quants), len(test_Z))



def naive_kernel_estimator(
        test_Z: np.ndarray,
        recal_Z: np.ndarray,
        recal_epsilon:np.ndarray,
        quants,
        wid = 1E-1,
):
    
    assert isinstance(test_Z, np.ndarray)
    assert isinstance(recal_Z, np.ndarray)
    assert isinstance(recal_epsilon, np.ndarray)
    
    
    assert len(test_Z.shape) == 2
    assert len(recal_Z.shape) == 2
    assert len(recal_epsilon.shape) == 1

    # quants should be a rising list
    assert npnorm(np.sort(quants) - quants) < 1E-6


    assert len(quants.shape) == 1
    
    indices = np.argsort(recal_epsilon)
    
    sorted_epsi = recal_epsilon[indices]

    sorted_recal_Z = recal_Z[indices]
       
    dist_mat = (-np.log(rbf_kernel(test_Z/wid, sorted_recal_Z/wid, gamma = 1)) > 1).astype(float)
     
    summation_matform = np.triu(np.ones((len(recal_Z), len(recal_Z))))
 
    aggregated_dist_mat = np.matmul(dist_mat, summation_matform)

    empirical_quantiles = aggregated_dist_mat / aggregated_dist_mat[:, -1:]


    quantiles_unsquze = empirical_quantiles.reshape(empirical_quantiles.shape + (-1,))

    tf_mat = quantiles_unsquze <= quants
    

    harvest_ids = np.clip(np.transpose(tf_mat.sum(axis=1), (1, 0)), a_max = len(recal_Z)-1, a_min = -1)

    return sorted_epsi[harvest_ids]          # shape (len(quants), len(test_Z))

