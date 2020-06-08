import os
import numpy as np
import matplotlib
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Activation, Input, Concatenate, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_addons as tfa
from utils import generator_batch_triplet_hard


opt_ways = {
    'sgd':SGD,
    'adam':Adam
}


def cross_entropy_label_smoothing(y_true, y_pred):
    label_smoothing = 0.2
    return categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)


def tripletSemihardloss(y_true, y_pred, margin=0.3):
    loss = tfa.losses.TripletSemiHardLoss(margin)
    return loss(y_true, y_pred)


def triplet_hard_loss(y_true, y_pred, margin = 0.3):
    label_shape = K.int_shape(y_true)
    print('label shape: ', label_shape)
    batch_size = label_shape[0]
    print('batch size: ', batch_size)
    print('y_true dim number', K.ndim(y_true))
    assert K.ndim(y_true) == 2, 'triplet hard loss label should be one-dim tensor'
    #anchor = K.l2_normalize(y_pred, axis=1)
    anchor = y_pred
    similarity_matrix = K.dot(anchor, K.transpose(anchor))
    similarity_matrix = K.clip(similarity_matrix, 0.0, 1.0)
    row2mat = K.repeat_elements(y_true, rep=batch_size, axis=1)
    col2mat = K.repeat_elements(K.transpose(y_true), rep=batch_size, axis=0)
    gt_matrix = K.cast(K.equal(row2mat, col2mat), 'float')
    positive_ind = K.argmin(similarity_matrix + 99. * (1 - gt_matrix), axis=1)
    negative_ind = K.argmax(similarity_matrix - 99 * gt_matrix, axis=1)
    positive = K.gather(anchor, positive_ind)
    negative = K.gather(anchor, negative_ind)
    pos_anchor_dist = K.sum(K.square(positive-anchor), axis=1)
    neg_anchor_dist = K.sum(K.square(negative-anchor), axis=1)
    basic_loss = pos_anchor_dist - neg_anchor_dist + margin
    loss = K.maximum(basic_loss, 0.0)
    return loss


def lr_decay_basic(epoch, initial_lrate):
    decay_epochs = [80, 90]
    if epoch in decay_epochs:
        new_lrate = 0.1*initial_lrate
        return new_lrate
    else:
        return initial_lrate


def lr_decay_warmup(epoch, initial_lrate):
    if epoch <= 11:
        return 0.00035*0.1*(epoch+1)
    if epoch >= 12 and epoch <= 41:
        return 0.00035
    if epoch >= 42 and epoch <= 71:
        return 0.000035
    if epoch >= 72:
        return 0.0000035


def main():
    args = parser.parse_args()
    image_names = os.listdir(args.data_root) #the best way is to use sorted list,i.e., sorted()
    image_names = sorted(image_names)[:-1]
    img_path = [os.path.join(args.data_root,x) for x in image_names]
    person_id_original_list = [x[:4] for x in image_names]
    num_person_ids = len(set(person_id_original_list))
    print('Number of Person IDs is {}'.format(num_person_ids))
    id_encoder = LabelEncoder()
    id_encoder.fit(person_id_original_list)
    person_id_encoded = id_encoder.transform(person_id_original_list)
    train_img_path, val_img_path, train_person_ids, val_person_ids = train_test_split(
        img_path, person_id_encoded, test_size=0.2, random_state=42)

    # model
    cnn_model = MobileNetV2(include_top=False, alpha=0.5, weights='imagenet',
                            input_shape=(args.img_height,args.img_width,3), pooling='max')
    global_pool = cnn_model.layers[-1].output
    normalized_global_pool = Lambda(lambda x: K.l2_normalize(x, axis=1), name='triplet')(global_pool)
    if args.USE_BNNeck:
        global_pool_bn = BatchNormalization(name= 'feature_bn')(global_pool)
        pred = Dense(num_person_ids, activation='softmax')(global_pool_bn)
    else:
        pred = Dense(num_person_ids, activation='softmax')(global_pool)

    triplet_model = Model(inputs=cnn_model.input, outputs=[pred, normalized_global_pool])


    # model compile
    if args.USE_Semihard:
        triplet_loss = tripletSemihardloss
    else:
        triplet_loss = triplet_hard_loss
    optimizer = opt_ways[args.optimizer](learning_rate=args.learning_rate)
    if args.USE_Label_Smoothing:
        triplet_model.compile(loss=[cross_entropy_label_smoothing, triplet_loss], optimizer=optimizer,
                            loss_weights=[1, 1],
                            metrics=['accuracy'])
    else:
        triplet_model.compile(loss=['categorical_crossentropy', triplet_loss], optimizer=optimizer,
                            loss_weights=[1, 1],
                            metrics=['accuracy'])
    #triplet_model.load_weights('triplet_hard_weights.h5')
    triplet_model.summary()
    # save model
    checkpoint = ModelCheckpoint(os.path.join(args.log_dir, 'triplet_hard_weights.h5'),
                                 monitor='val_dense_accuracy',
                                 verbose=1, save_best_only=True, mode='auto')
    reduce_lr = LearningRateScheduler(lr_decay_warmup, verbose=1)
    # data loader
    train_generator = generator_batch_triplet_hard(train_img_path, train_person_ids,
                    num_person_ids, args.img_width, args.img_height, args.batch_size, args.num_instances,
                                      shuffle=True, aug=True)
    val_generator = generator_batch_triplet_hard(val_img_path, val_person_ids,
                    num_person_ids, args.img_width, args.img_height, args.batch_size, args.num_instances,
                                      shuffle=False, aug=False)

    # fit data to model
    history = triplet_model.fit(
                        train_generator,
                        steps_per_epoch=len(train_img_path)//args.batch_size,
                        validation_data=val_generator,
                        validation_steps=len(val_img_path)//args.batch_size,
                        verbose=2,
                        shuffle=True,
                        epochs=args.num_epochs,
                        callbacks=[checkpoint, reduce_lr])
    #print(history.history)

    # Plot training & validation accuracy and loss values
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['dense_loss'], color='b', label='cross-entropy loss')
    ax[0].plot(history.history['triplet_loss'], color='r', label='triplet loss')
    ax[0].plot(history.history['val_dense_loss'], color='g', label='val cross-entropy loss')
    ax[0].plot(history.history['val_triplet_loss'], color='y', label='val triplet loss')
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['dense_accuracy'], color='b', label='Training accuracy')
    ax[1].plot(history.history['val_dense_accuracy'], color='r', label='Validation accuracy')
    legend = ax[1].legend(loc='best', shadow=True)
    plt.savefig('./loss_acc.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss")
    # data
    parser.add_argument('--img_width', type=int, default='64')
    parser.add_argument('--img_height', type=int, default='128')
    parser.add_argument('--learning_rate', type=float, default='0.01')
    parser.add_argument('--batch_size', type=int, default='128')
    parser.add_argument('--num_epochs', type=int, default='120')
    parser.add_argument('--num_instances', type=int, default='4')
    parser.add_argument('--USE_Label_Smoothing', type=bool, default=True)
    parser.add_argument('--USE_Semihard', type=bool, default=True)
    parser.add_argument('--USE_BNNeck', type=bool, default=True)
    parser.add_argument('--optimizer', type=str, default='adam')
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_root', type=str,
            default=os.path.join(working_dir, 'dataset/market1501/bounding_box_train'))
    parser.add_argument('--log_dir', type=str,
            default=os.path.join(working_dir, 'log'))
    main()

