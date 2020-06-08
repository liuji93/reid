import numpy as np
import cv2, random
from sklearn.utils import shuffle as shuffle_tuple
import imgaug as ia
from imgaug import augmenters as iaa
from tensorflow.python.keras.backend import set_session
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import normalize
import tensorflow as tf
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        # crop images by -5% to 10% of their height/width
        iaa.Cutout(nb_iterations=1, size=0.2, squared=False),  # random erasing to enhance robust
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.85, 1.15), "y": (0.85, 1.5)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, # translate by -20 to +20 percent (per axis)
            rotate=(-15, 15), # rotate by -45 to +45 degrees
            shear=(-5, 5), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 3),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(1, 5)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(1, 5)), # blur image using local medians with kernel sizes between 2 and 7
                ]),

                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.01, 0.03), per_channel=0.2),
                ]),
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)

                iaa.ContrastNormalization((0.3, 1.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
            ],
            random_order=True
        )
    ],
    random_order=True
)


def load_img_batch(img_batch_list, label_batch_list, num_classes, img_width, img_height):
    N = len(img_batch_list)
    X_batch = np.zeros((N, img_height, img_width, 3))
    Y_batch = np.zeros((N, num_classes))
    for i, img_path in enumerate(img_batch_list):
        img = cv2.imread(img_path)
        #img = cv2.resize(img,(img_height, img_width))
        img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        X_batch[i,:,:,:] = img
        if label_batch_list is not None:
            Y_batch[i,label_batch_list[i]] = 1
    if label_batch_list is not None:
        return X_batch, Y_batch
    else:
        return X_batch



def generator_batch(img_path_list, img_label_list, num_classes,
                    img_width, img_height, batch_size=32, shuffle=False, aug=False):
    assert len(img_path_list) == len(img_label_list), 'number of image should be equal to that of label'
    N = len(img_path_list)

    if shuffle:
        img_path_list, img_label_list = shuffle_tuple(img_path_list, img_label_list)

    batch_index = 0 # indicates batch_size

    while True:
        current_index = (batch_index*batch_size) % N #the first index for each batch per epoch
        if N >= (current_index+batch_size): # judge whether the current end index is over the train num
            current_batch_size = batch_size
            batch_index += 1 # indicates the next batch_size
        else:
            current_batch_size = N - current_index
            batch_index = 0

        X_batch, Y_batch = load_img_batch(img_path_list[current_index:current_index+current_batch_size],
                                          img_label_list[current_index:current_index + current_batch_size],
                                          num_classes,
                                          img_width,
                                          img_height)
        # data augmentation
        if aug:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq.augment_images(X_batch)
        # normalization
        X_batch = X_batch / 255.
        X_batch = (X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        yield (X_batch, Y_batch)


def generator_batch_triplet(img_path_list, img_label_list, num_classes,
                    img_width, img_height, batch_size=32, shuffle=False, aug=False):
    assert len(img_path_list) == len(img_label_list), 'number of image should be equal to that of label'
    if shuffle:
        img_path_list, img_label_list = shuffle_tuple(img_path_list, img_label_list)

    dic = {}
    for img_label, img_path in zip(img_label_list, img_path_list):
        dic.setdefault(img_label, []).append(img_path)

    #N = len(img_path_list)
    while True:
        person_ids_anchor_list = [k for k in dic.keys() if len(dic[k]) >= 2]
        anchor_ids_sampled = random.sample(person_ids_anchor_list, k = batch_size)
        negative_ids_2b_sampled = set(dic.keys()) - set(anchor_ids_sampled)
        anchor_positive_list = [tuple(random.sample(dic[anchor_id], k=2)) \
                                for anchor_id in anchor_ids_sampled]
        anchor_img_path_list, positive_img_path_list = zip(*anchor_positive_list)
        negative_ids_sampled = random.sample(negative_ids_2b_sampled, k = batch_size)
        negative_img_path_list = [random.sample(dic[negative_id], k=1)[0] \
                                  for negative_id in negative_ids_sampled]
        #print('anchor_img_path_list', anchor_img_path_list)
        anchor_X_batch, anchor_Y_batch = load_img_batch(list(anchor_img_path_list), anchor_ids_sampled,
                                         num_classes, img_width, img_height)
        positive_X_batch, _ = load_img_batch(list(positive_img_path_list), anchor_ids_sampled,
                                         num_classes, img_width, img_height)
        negative_X_batch, _ = load_img_batch(negative_img_path_list, negative_ids_sampled,
                                         num_classes, img_width, img_height)
        # data augmentation
        if aug:
            anchor_X_batch = anchor_X_batch.astype(np.uint8)
            anchor_X_batch = seq.augment_images(anchor_X_batch)

            positive_X_batch = positive_X_batch.astype(np.uint8)
            positive_X_batch = seq.augment_images(positive_X_batch)

            negative_X_batch = negative_X_batch.astype(np.uint8)
            negative_X_batch = seq.augment_images(negative_X_batch)
        # normalization
        anchor_X_batch = anchor_X_batch / 255.
        anchor_X_batch = (anchor_X_batch-np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
        positive_X_batch = positive_X_batch / 255.
        positive_X_batch = (positive_X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        negative_X_batch = negative_X_batch / 255.
        negative_X_batch = (negative_X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        yield ([anchor_X_batch, positive_X_batch, negative_X_batch],  [anchor_Y_batch, anchor_Y_batch])


def generator_batch_triplet_hard(img_path_list, img_label_list, num_classes,
                    img_width, img_height, batch_size, num_instances, shuffle=False, aug=False):

    assert len(img_path_list) == len(img_label_list), 'number of image should be equal to that of label'
    if shuffle:
        img_path_list, img_label_list = shuffle_tuple(img_path_list, img_label_list)

    dic = {}
    for img_label, img_path in zip(img_label_list, img_path_list):
        dic.setdefault(img_label, []).append(img_path)
    assert batch_size % num_instances == 0, "there should be K images per person in a batch"
    P = int(batch_size//num_instances)
    person_ids_list = [k for k in dic.keys() if len(dic[k]) >= num_instances]

    while True:
        person_ids_sampled = random.sample(person_ids_list, k=P)
        img_path_sampled = []
        img_ids_sampled = []
        for person_id in person_ids_sampled:
            img_path_sampled += random.sample(dic[person_id], k=num_instances)
            for _ in range(num_instances):
                img_ids_sampled.append(person_id)
        X_batch = np.zeros((batch_size, img_height, img_width, 3))
        Y_batch = np.zeros((batch_size, num_classes))

        for i, img_path in enumerate(img_path_sampled):
            img = cv2.imread(img_path)
            # img = cv2.resize(img,(img_height, img_width))
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
            X_batch[i, :, :, :] = img
            if img_ids_sampled is not None:
                Y_batch[i, img_ids_sampled[i]] = 1
        if aug:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq.augment_images(X_batch)
        X_batch = X_batch / 255.
        X_batch = (X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        y_batch = np.array(img_ids_sampled).astype(np.int32)
        yield ([X_batch, y_batch], [Y_batch, y_batch, y_batch])


def generator_batch_test(img_path_list, img_width, img_height, batch_size=32, shuffle=False):
    N = len(img_path_list)

    if shuffle:
        img_path_list = shuffle_tuple(img_path_list)

    batch_index = 0 # indicates batch_size

    while True:
        current_index = (batch_index*batch_size) % N #the first index for each batch per epoch
        if N >= (current_index+batch_size): # judge whether the current end index is over the train num
            current_batch_size = batch_size
            batch_index += 1 # indicates the next batch_size
        else:
            current_batch_size = N - current_index
            batch_index = 0
        img_batch_list = img_path_list[current_index:current_index + current_batch_size]

        X_batch = np.zeros((current_batch_size, img_height, img_width, 3))
        for i, img_path in enumerate(img_batch_list):
            img = cv2.imread(img_path)
            if img.shape[:2] != (img_height, img_width):
                img = cv2.resize(img, (img_height, img_width))
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
            X_batch[i, :, :, :] = img
        # normalization
        X_batch = X_batch / 255.
        X_batch = (X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        yield X_batch


def cross_entropy_label_smoothing(y_true, y_pred):
    label_smoothing = 0.2
    return categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)


def triplet_loss(y_gt, y_pred):
    margin = 0.5
    #y_pred = K.l2_normalize(y_pred, axis=1)
    #print('y_pred shape: ', y_pred.shape)
    assert y_pred.shape[1] % 3 == 0, 'concatenating error'
    dim_num = int(y_pred.shape[1]//3)
    anchor = K.l2_normalize(y_pred[:, :dim_num], axis=1)
    positive = K.l2_normalize(y_pred[:, dim_num:2*dim_num], axis=1)
    negative = K.l2_normalize(y_pred[:, 2*dim_num:3*dim_num], axis=1)
    pos_anchor_dist = K.sum(K.square(positive-anchor), axis=1)
    neg_anchor_dist = K.sum(K.square(negative-anchor), axis=1)
    basic_loss = pos_anchor_dist - neg_anchor_dist + margin
    loss = K.maximum(basic_loss, 0.0)
    return loss


def scheduler(epoch, lr):
    if epoch < 80:
        return lr
    elif epoch == 80:
        return lr*0.1
    elif epoch < 90:
        return lr
    elif epoch == 90:
        return lr*0.1
    else:
        return lr










