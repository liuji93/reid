import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from utils import cross_entropy_label_smoothing, generator_batch, scheduler

# configure params
img_width = 64
img_height = 128
learning_rate = 0.001
optimizer = SGD(learning_rate=learning_rate)
batch_size = 128
num_epochs = 30
USE_Label_Smoothing = True

# data and label preparing
data_root = '/data/PersonReID/market1501/bounding_box_train' #the project folder
image_names = sorted(os.listdir(data_root))[:-1] #the best way is to use sorted list,i.e., sorted()
img_path = [os.path.join(data_root,x) for x in image_names]
person_id_original_list = [x[:4] for x in image_names]
num_person_ids = len(set(person_id_original_list))
print('Number of Person IDs is {}'.format(num_person_ids))
id_encoder = LabelEncoder()
id_encoder.fit(person_id_original_list)
person_id_encoded = id_encoder.transform(person_id_original_list)
train_img_path, val_img_path, train_person_ids, val_person_ids = train_test_split(
    img_path, person_id_encoded, test_size=0.2, random_state=42)

#model
cnn_model = MobileNetV2(include_top=False, weights='imagenet', alpha=0.5,
                        input_shape=(img_height, img_width, 3), pooling='max')
global_pool = cnn_model.layers[-1].output
softmax_output = Dense(num_person_ids, activation='softmax')(global_pool)
baseline_model = Model(cnn_model.input, softmax_output)
baseline_model.load_weights('baseline_weights.h5')

if USE_Label_Smoothing:
    baseline_model.compile(loss=cross_entropy_label_smoothing, optimizer=optimizer,
                       metrics=['accuracy'])
else:
    baseline_model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                       metrics=['accuracy'])
checkpoint = ModelCheckpoint('./baseline_weights.h5', monitor='val_acc',
                             verbose=1, save_best_only=True, mode='auto')
#reduce_lr = LearningRateScheduler(scheduler)

train_generator = generator_batch(train_img_path, train_person_ids, num_person_ids,
                                  img_width, img_height, batch_size, shuffle=True, aug=True)
val_generator = generator_batch(val_img_path,val_person_ids,num_person_ids,
                                img_width, img_height, batch_size, shuffle=False, aug=False)

history = baseline_model.fit(
                    train_generator,
                    steps_per_epoch = len(train_img_path)//batch_size,
                    validation_data = val_generator,
                    validation_steps = len(val_img_path)//batch_size,
                    verbose = 2,
                    shuffle = True,
                    epochs = num_epochs,
                    callbacks=[checkpoint])
#print(history.history)
# Plot training & validation accuracy and loss values
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'],color='b',label='Training loss')
ax[0].plot(history.history['val_loss'],color='r',label='validation loss',axes=ax[0])
legend = ax[0].legend(loc='best',shadow=True)

ax[1].plot(history.history['acc'], color='b', label='Training accuracy')
ax[1].plot(history.history['val_acc'], color='r', label='Validation accuracy')
legend = ax[1].legend(loc='best',shadow=True)
plt.savefig('./loss_acc.jpg')


