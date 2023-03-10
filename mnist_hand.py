import random
import numpy as np
from Fei_dataset import *
from six.moves import xrange
from scipy.misc import imsave as ims
from HSICSupport import *
from ops import *
from Utlis2 import *
import gzip
import cv2
import keras as keras

def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec

def Give_InverseFashion32():
    mnistName = "Fashion"
    data_X, data_y = load_mnist(mnistName)
    data_X = np.reshape(data_X, (-1, 28, 28))
    for i in range(np.shape(data_X)[0]):
        for k1 in range(28):
            for k2 in range(28):
                data_X[i, k1, k2] = 1.0 - data_X[i, k1, k2]

    data_X = np.reshape(data_X, (-1, 28, 28, 1))
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)
    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i], size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    return mnist_train_x,mnist_train_label,mnist_test,mnist_label_test

def Give_InverseMNIST32():
    mnistName = "mnist"
    data_X, data_y = load_mnist(mnistName)
    data_X = np.reshape(data_X, (-1, 28, 28))
    for i in range(np.shape(data_X)[0]):
        for k1 in range(28):
            for k2 in range(28):
                data_X[i, k1, k2] = 1.0 - data_X[i, k1, k2]

    data_X = np.reshape(data_X, (-1, 28, 28, 1))
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)
    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i], size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    return mnist_train_x,mnist_train_label,mnist_test,mnist_label_test

def GiveMNIST32():
    mnistName = "mnist"
    data_X, data_y = load_mnist(mnistName)

    # data_X = np.expand_dims(data_X, axis=3)
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)

    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i], size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    return mnist_train_x, mnist_train_label, mnist_test, mnist_label_test


def GiveMNIST_SVHN():
    mnistName = "mnist"
    data_X, data_y = load_mnist(mnistName)

    #data_X = np.expand_dims(data_X, axis=3)
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)

    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i],size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test


    myTest = mnist_train_x[0:64]

    ims("results/" + "gggg" + str(0) + ".jpg", merge2(myTest[:64], [8, 8]))

    x_train, y_train, x_test, y_test = GetSVHN_DataSet()
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    return mnist_train_x,mnist_train_label,mnist_test,mnist_label_test,x_train,y_train,x_test,y_test

def Split_Dataset_ByClasses(x,y):
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    arr6 = []
    arr7 = []
    arr8 = []
    arr9 = []
    arr10 = []

    count = np.shape(x)[0]
    y1 = [np.argmax(one_hot)for one_hot in y]
    for i in range(count):
        if y1[i] == 0:
            arr1.append(x[i])
        if y1[i] == 1:
            arr2.append(x[i])
        if y1[i] == 2:
            arr3.append(x[i])
        if y1[i] == 3:
            arr4.append(x[i])
        if y1[i] == 4:
            arr5.append(x[i])
        if y1[i] == 5:
            arr6.append(x[i])
        if y1[i] == 6:
            arr7.append(x[i])
        if y1[i] == 7:
            arr8.append(x[i])
        if y1[i] == 8:
            arr9.append(x[i])
        if y1[i] == 9:
            arr10.append(x[i])

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    arr3 = np.array(arr3)
    arr4 = np.array(arr4)
    arr5 = np.array(arr5)
    arr6 = np.array(arr6)
    arr7 = np.array(arr7)
    arr8 = np.array(arr8)
    arr9 = np.array(arr9)
    arr10 = np.array(arr10)

    return arr1,arr2,arr3,arr4,arr5,arr6,arr7,arr8,arr9,arr10

def GiveFashion32():
    mnistName = "Fashion"
    data_X, data_y = load_mnist(mnistName)

    # data_X = np.expand_dims(data_X, axis=3)
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)

    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i], size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    return mnist_train_x,mnist_train_label,mnist_test,mnist_label_test

def Split_dataset(x,y,n_label):
    y = np.argmax(y,axis=1)
    n_each = n_label / 10
    isRun = True
    x_train = []
    y_train = []
    index = np.zeros(10)
    while(isRun):
        a = random.randint(0, np.shape(x)[0])-1
        x1 = x[a]
        y1 = y[a]
        if index[y1] < n_each:
            x_train.append(x1)
            y_train.append(y1)
            index[y1] = index[y1]+1
        isOk1 = True
        for i in range(10):
            if index[i] < n_each:
                isOk1 = False
        if isOk1:
            break

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train,y_train

def Give_InverseFashion():
    mnistName = "Fashion"
    data_X, data_y = load_mnist(mnistName)
    data_X = np.reshape(data_X,(-1,28,28))
    for i in range(np.shape(data_X)[0]):
        for k1 in range(28):
            for k2 in range(28):
                data_X[i,k1,k2] = 1.0 - data_X[i,k1,k2]

    data_X = np.reshape(data_X,(-1,28,28,1))
    return data_X,data_y

def Exchange_Dataset(x1,y1,x2,y2):
    x1_arr1,x1_arr2,x1_arr3,x1_arr4,x1_arr5,x1_arr6,x1_arr7,x1_arr8,x1_arr9,x1_arr10 = Split_Dataset_ByClasses(x1,y1)
    x2_arr1,x2_arr2,x2_arr3,x2_arr4,x2_arr5,x2_arr6,x2_arr7,x2_arr8,x2_arr9,x2_arr10 = Split_Dataset_ByClasses(x2,y2)

    newX1 = np.concatenate((x1_arr1,x1_arr2,x1_arr3,x1_arr4,x1_arr4,x1_arr5,x1_arr6,x1_arr7,x1_arr8,x2_arr9,x2_arr10),axis=0)
    newX2 = np.concatenate((x2_arr1,x2_arr2,x2_arr3,x2_arr4,x2_arr4,x2_arr5,x2_arr6,x2_arr7,x2_arr8,x1_arr9,x1_arr10),axis=0)

    return newX1,newX2


