import os
import numpy as np
import matplotlib
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Activation, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from utils import cross_entropy_label_smoothing, generator_batch_triplet, triplet_loss


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

    #cnn_model.load_weights('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_128_no_top.h5') # load imagenet prereain
    anchor_input = Input((args.img_height, args.img_width, 3), name='anchor_input')
    positive_input = Input((args.img_height, args.img_width, 3), name='positive_input')
    negative_input = Input((args.img_height, args.img_width, 3), name='negative_input')
    anchor_embedding = cnn_model(anchor_input)
    positve_embedding = cnn_model(positive_input)
    negative_embedding = cnn_model(negative_input)
    merged_vector = Concatenate(axis=-1, name='triplet')([anchor_embedding, positve_embedding, negative_embedding])
    #print('merged_vector shape: ', merged_vector.shape)
    dense_anchor = Dense(num_person_ids)(anchor_embedding)

    anchor_softmax_output = Activation('softmax')(dense_anchor)
    triplet_model = Model(inputs=[anchor_input, positive_input, negative_input],
                          outputs=[anchor_softmax_output, merged_vector])

    triplet_model.load_weights('triplet_weights.h5')

    # model compile
    optimizer = SGD(learning_rate=args.learning_rate)
    if args.USE_Label_Smoothing:
        triplet_model.compile(loss=[cross_entropy_label_smoothing, triplet_loss], optimizer=optimizer,
                            loss_weights=[1, 1],
                            metrics=['accuracy'])
    else:
        triplet_model.compile(loss=['categorical_crossentropy', triplet_loss], optimizer=optimizer,
                            loss_weights=[1, 1],
                            metrics=['accuracy'])

    # save model
    checkpoint = ModelCheckpoint('./triplet_weights.h5', monitor='val_activation_acc',
                                 verbose=1, save_best_only=True, mode='auto')
    #reduce_lr = ReduceLROnPlateau(monitor='val_activation_acc', patience=5, mode='auto')
    # data loader
    train_generator = generator_batch_triplet(train_img_path, train_person_ids, num_person_ids,
                                      args.img_width, args.img_height, args.batch_size, shuffle=True, aug=True)
    val_generator = generator_batch_triplet(val_img_path,val_person_ids, num_person_ids,
                                    args.img_width, args.img_height, args.batch_size, shuffle=False, aug=False)

    # fit data to model
    history = triplet_model.fit(
                        train_generator,
                        steps_per_epoch = len(train_img_path)//args.batch_size,
                        validation_data = val_generator,
                        validation_steps = len(val_img_path)//args.batch_size,
                        verbose = 2,
                        shuffle = True,
                        epochs = args.num_epochs,
                        callbacks=[checkpoint])
    #print(history.history)

    # Plot training & validation accuracy and loss values
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['loss'],color='b',label='Training loss')
    ax[0].plot(history.history['val_activation_loss'],color='r',label='validation loss',axes=ax[0])
    legend = ax[0].legend(loc='best',shadow=True)

    ax[1].plot(history.history['activation_acc'], color='b', label='Training accuracy')
    ax[1].plot(history.history['val_activation_acc'], color='r', label='Validation accuracy')
    legend = ax[1].legend(loc='best',shadow=True)
    plt.savefig('./loss_acc.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss")
    # data
    parser.add_argument('--data_root', type=str,
            default='/data/PersonReID/market1501/bounding_box_train')
    parser.add_argument('--img_width', type=int, default='64')
    parser.add_argument('--img_height', type=int, default='128')
    parser.add_argument('--learning_rate', type=float, default='0.001')
    parser.add_argument('--batch_size', type=int, default='128')
    parser.add_argument('--num_epochs', type=int, default='30')
    parser.add_argument('--USE_Label_Smoothing', type=bool, default=True)
    main()

