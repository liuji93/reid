

This repository is the implementation of Person ReID by 
using [MobileNetV2](https://arxiv.org/abs/1801.04381) on 
[Market1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf) dataset. 
The implementation is based on keras (tensorflow backend).
## Requirements
tensorflow==2.2.0 <br/>
tensorflow_addons<br/>
scikit-learn<br/>
matplotlib<br/>
numpy<br/>
imgaug<br/>
cv2<br/>


## Training

To train the baseline model (MoblieNetV2 backbone and Crossentropy+Triplet loss), run this command:
```train
python train_triplet.py --learning_rate 0.01
```
To train the improved baseline model, run this command:
```train
python train_triplet_seimihard.py --USE_Semihard True --USE_BNNeck True 
```


## Evaluation

To evaluate the baseline model on Market1501, run:
```eval
python test_triplet.py
```
To evaluate the improved baseline model on Market1501, run:
```eval
python test_triplet_semihard.py
```
## Results

We achieve the following performance on Market1501:

| Model name           |        mAP      | Rank-1 Accuracy|
| ------------------   |---------------- | -------------- |
|      baseline        |        --%      |      --%       |
|   improved baseline  |      40.5%      |      61.7%     |

