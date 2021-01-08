#由于地图中左转较多，为防止小车易朝左偏出车道，对部分训练图片进行水平翻转，角度取负值。
def horizontal_flip(img, degree):
    choice = np.random.choice([0, 1])
    if choice == 1:
        img, degree = cv2.flip(img, 1), -degree
    return (img, degree)

#进行色彩空间转换并随机调整亮度。
def random_brightness(img, degree):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    alpha = np.random.uniform(low = 0.1, high = 1.0, size = None)
    v = hsv[:, :, 2]
    v = v * alpha
    hsv[:, :, 2] = v.astype('uint8')
    rgb = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)
    return(rgb, degree)

#训练数据中转向角为0的情况过多，将部分转向角为0的情况删除，设置丢弃率rate。
def discard_zero_steering(degrees, rate):
    steering_zero_idx = np.where(degrees == 0)
    steering_zero_idx = steering_zero_idx[0]
    size_del = int(len(steering_zero_idx) * rate)
    return np.random.choice(steering_zero_idx, size = size_del, replace = False)

#为实时查看迭代过程中loss变化，安装livelossplot
pip  install livelossplot

#网络训练（train.py）：
import numpy as np
from keras.optimizers import SGD, Adam
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, PReLU, BatchNormalization
from keras.models import Sequential, Model
from keras import backend as K
from keras.regularizers import l2
import os.path
import csv
import cv2
import glob
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
from keras import callbacks
import math
from matplotlib import pyplot
from livelossplot import PlotLossesKeras

SEED = 13


def horizontal_flip(img, degree):
    choice = np.random.choice([0, 1])
    if choice == 1:
        img, degree = cv2.flip(img, 1), -degree
    return (img, degree)


def random_brightness(img, degree):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    alpha = np.random.uniform(low=0.1, high=1.0, size=None)
    v = hsv[:, :, 2]
    v = v * alpha
    hsv[:, :, 2] = v.astype('uint8')
    rgb = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)
    return (rgb, degree)


def left_right_random_swap(img_address, degree, degree_corr=1.0 / 4):
    swap = np.random.choice(['L', 'R', 'C'])
    if swap == 'L':
        img_address = img_address.replace('center', 'left')
        corrected_label = np.arctan(math.tan(degree) + degree_corr)
        return (img_address, corrected_label)
    elif swap == 'R':
        img_address = img_address.replace('center', 'right')
        corrected_label = np.arctan(math.tan(degree) - degree_corr)
        return (img_address, corrected_label)
    else:
        return (img_address, degree)


def discard_zero_steering(degrees, rate):
    steering_zero_idx = np.where(degrees == 0)
    steering_zero_idx = steering_zero_idx[0]
    size_del = int(len(steering_zero_idx) * rate)
    return np.random.choice(steering_zero_idx, size=size_del, replace=False)


def get_model(shape):
    '''
    预测方向盘角度: 以图像为输入, 预测方向盘的转动角度
    shape: 输入图像的尺寸, 例如(128, 128, 3)
    '''
    model = Sequential()
    model.add(Conv2D(8, (5, 5), strides=(1, 1), padding="valid", input_shape=shape))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(8, (5, 5), strides=(1, 1), padding="valid", ))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (4, 4), strides=(1, 1), padding="valid", ))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (5, 5), strides=(1, 1), padding="valid", ))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(50))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(10))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1, activation='linear'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mean_squared_error')

    return model


def image_transformation(img_address, degree, data_dir):
    img_address, degree = left_right_random_swap(img_address, degree)
    img = cv2.imread(img_address)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, degree = random_brightness(img, degree)
    img, degree = horizontal_flip(img, degree)
    return (img, degree)


def batch_generator(x, y, batch_size, shape, training=True, data_dir='data/', monitor=True, yieldXY=True,
                    discard_rate=0.95):
    if training:
        y_bag = []
        x, y = shuffle(x, y)
        new_x = x
        new_y = y
    else:
        new_x = x
        new_y = y

    offset = 0
    while True:
        X = np.empty((batch_size, *shape))
        Y = np.empty((batch_size, 1))

        for example in range(batch_size):
            img_address, img_steering = new_x[example + offset], new_y[example + offset]
            # print(img_address)
            if training:
                img, img_steering = image_transformation(img_address, img_steering, data_dir)
            else:
                # img = cv2.imread(data_dir + img_address)
                img = cv2.imread(img_address)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            X[example, :, :, :] = cv2.resize(img[80:140, 0:320], (shape[0], shape[1])) / 255 - 0.5

            Y[example] = img_steering
            if training:
                y_bag.append(img_steering)

            if (example + 1) + offset > len(new_y) - 1:
                x, y = shuffle(x, y)
                rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
                new_x = x
                new_y = y
                new_x = np.delete(new_x, rand_zero_idx, axis=0)
                new_y = np.delete(new_y, rand_zero_idx, axis=0)
                offset = 0
        if yieldXY:
            yield (X, Y)
        else:
            yield X

        offset = offset + batch_size
        if training:
            np.save('y_bag.npy', np.array(y_bag))
            np.save('Xbatch_sample.npy', X)


if __name__ == '__main__':

    data_path = 'data/'
    with open(data_path + 'driving_log.csv', 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        log = []
        for row in file_reader:
            log.append(row)

    log = np.array(log)

    # 判断图像文件数量是否等于csv日志文件中记录的数量
    ls_imgs = glob.glob(data_path + 'IMG/*.jpg')
    assert len(ls_imgs) == len(log) * 3, 'number of images does not match'

    # 使用20%的数据作为测试数据
    validation_ratio = 0.2
    shape = (128, 128, 3)
    batch_size = 8
    nb_epoch = 400

    x_ = log[:, 0]
    y_ = log[:, 3].astype(float)
    x_, y_ = shuffle(x_, y_)
    X_train, X_val, y_train, y_val = train_test_split(x_, y_, test_size=validation_ratio, random_state=SEED)

    print('batch size: {}'.format(batch_size))
    print('Train set size: {} | Validation set size: {}'.format(len(X_train), len(X_val)))

    samples_per_epoch = batch_size
    # 使得validation数据量大小为batch_size的整数陪
    nb_val_samples = len(y_val) - len(y_val) % batch_size
    model = get_model(shape)
    print(model.summary())

    # 根据validation loss保存最优模型
    save_best = callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='min')

    # 如果训练持续没有validation loss的提升, 提前结束训练，可通过更改patience参数确定终止条件。
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=400,
                                         verbose=0, mode='auto')
    callbacks_list = [early_stop, save_best, PlotLossesKeras()]

    history = model.fit_generator(batch_generator(X_train, y_train, batch_size, shape, training=True),
                                  steps_per_epoch=samples_per_epoch,
                                  validation_steps=nb_val_samples // batch_size,
                                  validation_data=batch_generator(X_val, y_val, batch_size, shape,
                                                                  training=False, monitor=False),
                                  epochs=nb_epoch, verbose=1, callbacks=callbacks_list)

    with open('./trainHistoryDict.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.savefig('train_val_loss.jpg')

    # 保存模型
    with open('model.json', 'w') as f:
        f.write(model.to_json())
    model.save('model.h5')
    print('Done!')
