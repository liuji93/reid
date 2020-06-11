from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import SGD
from train_triplet import cross_entropy_label_smoothing
from train_triplet import triplet_loss
from dataloader import generator_batch_test
from evaluate import evaluate
from sklearn.preprocessing import normalize
import os, argparse
import numpy as np

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
    args = parser.parse_args()
    # build model to extract features
    if args.USE_Label_Smoothing:
        model = load_model(args.model_path, custom_objects= \
            {'cross_entropy_label_smoothing': cross_entropy_label_smoothing,
               'triplet_loss':triplet_loss})
    else:
        model = load_model(args.model_path, custom_objects= \
            {'triplet_loss':triplet_loss})
    model.summary()
    cnn_model = model.get_layer('mobilenetv2_0.50_224')
    dense_feature = cnn_model.get_layer('global_max_pooling2d').output
    model_extract_features = Model(inputs=cnn_model.input, outputs=dense_feature)

    model_extract_features.compile(loss=['categorical_crossentropy'],
                            optimizer=SGD(lr=0.1), metrics=['accuracy'])

    #image_path, image_names, image_cams
    query_img_list, query_name_list, query_cams_list = \
        get_data_information(args.query_dir)
    gallery_img_list, gallery_name_list, gallery_cams_list = \
        get_data_information(args.gallery_dir)

    # obtain features
    query_generator = generator_batch_test(query_img_list, args.img_width, args.img_height,
                                           args.batch_size, shuffle=False)
    query_features = model_extract_features.predict(query_generator, verbose=1,
            steps=len(query_img_list)//args.batch_size if len(query_img_list)%args.batch_size==0 \
            else len(query_img_list)//args.batch_size+1)
    query_features = normalize(query_features, norm='l2')
    assert len(query_img_list) == query_features.shape[0], "something wrong with query samples"

    gallery_generator = generator_batch_test(gallery_img_list, args.img_width, args.img_height,
                                             args.batch_size, shuffle=False)
    gallery_features = model_extract_features.predict(gallery_generator,verbose=1,
            steps=len(gallery_img_list)//args.batch_size if len(gallery_img_list)%args.batch_size==0 \
            else len(gallery_img_list)//args.batch_size+1)
    gallery_features = normalize(gallery_features, norm='l2')
    assert len(gallery_img_list) == gallery_features.shape[0], "something wrong with gallery samples"
    #evaluate
    evaluate(query_features, query_name_list, query_cams_list,
             gallery_features, gallery_name_list, gallery_cams_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test")
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(working_dir, 'logs/triplet_weights.h5'))
    parser.add_argument('--query_dir', type=str,
                default=os.path.join(working_dir, 'datasets/Market-1501-v15.09.15/query'))
    parser.add_argument('--gallery_dir', type=str,
                default=os.path.join(working_dir, 'datasets/Market-1501-v15.09.15/bounding_box_test'))
    parser.add_argument('--img_width', type=int, default='64')
    parser.add_argument('--img_height', type=int, default='128')
    parser.add_argument('--learning_rate', type=float, default='0.01')
    parser.add_argument('--batch_size', type=int, default='128')
    parser.add_argument('--USE_Label_Smoothing', type=bool, default=True)
    main()