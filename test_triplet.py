from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import SGD
from utils import cross_entropy_label_smoothing
from train_triplet_hard import tripletSemihardloss
from utils import generator_batch_test
from evaluate import evaluate
from sklearn.preprocessing import normalize
import os
import numpy as np

USE_Label_Smoothing = True
model_path = './logs/triplet_hard_weights.h5'
optimizer = SGD(learning_rate=0.1)
img_width = 64
img_height = 128
batch_size = 32
query_dir = './datasets/market1501/query'
gallery_dir = './datasets/market1501/bounding_box_test'


def get_data_information(data_root):
    img_path_list = []
    img_name_list = []
    img_cams_list = []
    image_names = os.listdir(data_root) #the best way is to use sorted list,i.e., sorted()
    image_names = sorted(image_names)[:-1]
    for item in image_names:
        if item[-4:] == '.jpg':
            img_path_list.append(os.path.join(data_root, item))
            img_name_list.append(item.split('_')[0])
            img_cams_list.append(item.split('c')[1][0])
    return img_path_list, np.array(img_name_list), np.array(img_cams_list)


def main():
    # build model to extract features
    if USE_Label_Smoothing:
        model = load_model(model_path, custom_objects= \
            {'cross_entropy_label_smoothing': cross_entropy_label_smoothing,
               'tripletSemihardloss':tripletSemihardloss})
    else:
        model = load_model(model_path, custom_objects= \
            {'triplet_loss':tripletSemihardloss})
    model.summary()
    cnn_input = model.input
    dense_feature = model.get_layer('global_max_pooling2d').output
    #print('111', model.input[0])
    model_extract_features = Model(inputs=cnn_input, outputs=dense_feature)

    model_extract_features.compile(loss=['categorical_crossentropy'],
                                   optimizer=optimizer, metrics=['accuracy'])

    #image_path, image_names, image_cams
    query_img_list, query_name_list, query_cams_list = \
        get_data_information(query_dir)
    gallery_img_list, gallery_name_list, gallery_cams_list = \
        get_data_information(gallery_dir)

    # obtain features
    query_generator = generator_batch_test(query_img_list, img_width, img_height,
                                           batch_size, shuffle=False)
    query_features = model_extract_features.predict(query_generator, verbose=1,
                    steps=len(query_img_list)//batch_size if len(query_img_list)%batch_size==0 \
                    else len(query_img_list)//batch_size+1)
    query_features = normalize(query_features, norm='l2')
    assert len(query_img_list) == query_features.shape[0], "something wrong with query samples"

    gallery_generator = generator_batch_test(gallery_img_list, img_width, img_height,
                                             batch_size, shuffle=False)
    gallery_features = model_extract_features.predict(gallery_generator,verbose=1,
                        steps=len(gallery_img_list)//batch_size if len(gallery_img_list)%batch_size==0 \
                        else len(gallery_img_list)//batch_size+1)
    gallery_features = normalize(gallery_features, norm='l2')
    assert len(gallery_img_list) == gallery_features.shape[0], "something wrong with gallery samples"
    #evaluate
    evaluate(query_features, query_name_list, query_cams_list,
             gallery_features, gallery_name_list, gallery_cams_list )


if __name__ == '__main__':
    main()

