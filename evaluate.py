#reference: https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/evaluate_rerank.py
import numpy as np
from utils.ranking import cmc, mean_ap
from utils.rerank import re_ranking
#######################################################################

def pairwise_distance(mat1, mat2):
    m = mat1.shape[0]  # query number
    n = mat2.shape[0]  # gallery number
    x = np.repeat(np.sum(np.square(mat1), axis=1, keepdims=True), n, axis=1)  # mxn
    y = np.repeat(np.sum(np.square(mat2), axis=1, keepdims=True), m, axis=1)  # nxm
    y = np.transpose(y)  # mxn
    return x + y - 2 * np.dot(mat1, mat2.T)

######################################################################
def evaluate(query_features, query_labels, query_cams,  gallery_features, gallery_labels,
             gallery_cams):
    #query_feature: array, NxD
    #query_cam: array, 1xN
    #query_label: array, 1xN
    #gallery_feature: array, MxD
    #gallery_cam：array, 1xM
    #gallery_label array, 1xM
    distmat = pairwise_distance(query_features, gallery_features)

    print('Applying person re-ranking ...')
    distmat_qq = pairwise_distance(query_features, query_features)
    distmat_gg = pairwise_distance(gallery_features, gallery_features)
    distmat = re_ranking(distmat, distmat_qq, distmat_gg)
    # Compute mean AP
    mAP = mean_ap(distmat, query_labels, gallery_labels, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_labels, gallery_labels, query_cams,
                            gallery_cams, **params)
                  for name, params in cmc_configs.items()}
    cmc_topk = (1, 5, 10)
    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))

