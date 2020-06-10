#reference: https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/evaluate_rerank.py
import numpy as np
from ranking import cmc, mean_ap

#######################################################################

def compute_cmc_mAP(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index)).astype(np.int)
    if good_index.shape[0] == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    # a = [1,2,3,4,5]
    # b = [1,3]
    # np.in1d(a,b,invert=True) >>> [False, True, False, True, True]
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


# Evaluate
def eval(score, ql, qc, gl, gc, dist_type):
    if dist_type == 'cosine':
        index = np.argsort(score)  # from small to large
        index = index[::-1]  # from large to small
    else:
        index = np.argsort(score)  # from small to large

    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)
    # find items in query_index, but not in camera_index, which means same id and different camera
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)  #
    # find items in both query_index and camera_index, which means same id and same camera
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_cmc_mAP(index, good_index, junk_index)
    return CMC_tmp


######################################################################
def evaluate(query_feature, query_label, query_cam,  gallery_feature, gallery_label,
             gallery_cam, dist_type):
    #query_feature: array, NxD
    #query_cam: array, 1xN
    #query_label: array, 1xN
    #gallery_feature: array, MxD
    #gallery_cam：array, 1xM
    #gallery_label array, 1xM
    CMC = np.zeros(len(gallery_label)).astype(np.int)
    ap = 0.0
    if dist_type == 'cosine':
        scores = np.dot(query_feature, gallery_feature.T)
    else:
        m = query_feature.shape[0] # query number
        n = gallery_feature.shape[0] # gallery number
        x = np.repeat(np.sum(np.square(query_feature), axis=1, keepdims=True), n, axis=1) #mxn
        y = np.repeat(np.sum(np.square(gallery_feature), axis=1, keepdims=True), m, axis=1) #nxm
        y = np.transpose(y) #mxn
        scores = x+y-2*np.dot(query_feature, gallery_feature.T)
    # Compute mean AP
    mAP = mean_ap(scores, query_label, gallery_label, query_cam, gallery_cam)
    print('Mean AP: {:4.1%}'.format(mAP))


    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(scores, query_label, gallery_label, query_cam,
                            gallery_cam, **params)
                  for name, params in cmc_configs.items()}
    cmc_topk = (1, 5, 10)
    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))

    # print(query_label)
    '''
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = eval(scores[i], query_label[i], query_cam[i], gallery_label,
                                   gallery_cam, dist_type)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        # print(i, CMC_tmp[0])

    CMC = CMC / len(query_label)  # average CMC
    print('Rank@1:%f\n Rank@5:%f\n Rank@10:%f\n mAP:%f' % (CMC[0], CMC[4], CMC[9],
                                                           ap / len(query_label)))
    '''


