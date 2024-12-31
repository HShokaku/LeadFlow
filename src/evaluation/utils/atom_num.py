"""Utils for sampling size of a molecule of a given protein pocket."""

import numpy as np
from scipy import spatial as sc_spatial

from src.evaluation.utils.atom_num_config import CONFIG


def get_space_size(pocket_3d_pos):
    aa_dist = sc_spatial.distance.pdist(pocket_3d_pos, metric='euclidean')
    aa_dist = np.sort(aa_dist)[::-1]
    # 获取10个最远点的中位数
    # 如果用每个原子的重原子，那这其实受ref_ligand获取口袋的大小影响
    # 如果用refligand的中心位置再去取10A，会不会好一些，那就完全变成了一个球形的口袋，大家都一样了
    # 现在这个方法挺鲁棒的
    return np.median(aa_dist[:10])


def _get_bin_idx(space_size):
    bounds = CONFIG['bounds']
    for i in range(len(bounds)):
        if bounds[i] > space_size:
            return i
    return len(bounds)


def sample_atom_num(space_size):
    bin_idx = _get_bin_idx(space_size)
    num_atom_list, prob_list = CONFIG['bins'][bin_idx]
    return np.random.choice(num_atom_list, p=prob_list)
