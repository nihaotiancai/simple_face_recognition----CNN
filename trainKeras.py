from keras.models import Sequential
from keras.optimizers import SGD, Adam
import keras.utils as np_utils
from keras.layers import Activation, Dense, MaxPooling2D, Convolution2D, Flatten
import pickle
import numpy

# There are 40 different classes
nb_classes = 40
nb_epoch = 40
batch_size = 40

# input image dimensions
img_rows, img_cols = 57, 47
# number of convolutional filters to use
nb_filters1, nb_filters2 = 5, 10
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


def load_data():
    read_file = open('olivettifaces.pkl', 'rb')
    faces = pickle.load(read_file)
    label = pickle.load(read_file)
    read_file.close()

    # train:320,valid:40,test:40
    train_data = numpy.empty((320, 2679))
    train_label = numpy.empty(320)
    valid_data = numpy.empty((40, 2679))
    valid_label = numpy.empty(40)
    test_data = numpy.empty((40, 2679))
    test_label = numpy.empty(40)

    for i in range(40):
        train_data[i * 8:i * 8 + 8] = faces[i * 10:i * 10 + 8]
        train_label[i * 8:i * 8 + 8] = label[i * 10:i * 10 + 8]
        valid_data[i] = faces[i * 10 + 8]
        valid_label[i] = label[i * 10 + 8]
        test_data[i] = faces[i * 10 + 9]
        test_label[i] = label[i * 10 + 9]

    rval = [(train_data, train_label), (valid_data, valid_label),
            (test_data, test_label)]
    return rval



def build_model():
    model = Sequential()

    # Conv layer 1 output shape (32, 28, 28)
    model.add(Convolution2D(
        filters=32,
        kernel_size=(5, 5),
        padding='same',  # Padding method
        input_shape=(1,  # channels
                     57, 47,)  # height & width
    ))
    model.add(Activation('tanh'))

    # Pooling layer 1 (max pooling) output shape (32, 14, 14)
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',  # Padding method
    ))

    # Conv layer 2 output shape (64, 14, 14)
    model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
    model.add(Activation('tanh'))

    # Pooling layer 2 (max pooling) output shape (64, 7, 7)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))

    # Fully connected layer 2 to shape (10) for 10 classes
    model.add(Dense(40))
    model.add(Activation('softmax'))

    # to define your optimizer
    adam = Adam(lr=1e-4)

    # We add metrics to get more results you want to see
    # model.compile(optimizer=adam,
    #              loss='categorical_crossentropy',
    #              metrics=['accuracy'])
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def train(X_train, y_train, model):
    print('Training ------------')
    # Another way to train the model
    model.fit(X_train, y_train, epochs=200, batch_size=40)
    return model


def complete_accuary(X_test, y_test, model):
    print('\nTesting ------------')
    # Evaluate the model with the metrics we defined earlier
    loss, accuracy = model.evaluate(X_test, y_test)
    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)

def start():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    X_train = X_train.reshape(-1, 1, 57, 47)
    X_val = X_val.reshape(-1, 1, 57, 47)
    X_test = X_test.reshape(-1, 1, 57, 47)
    y_train = np_utils.to_categorical(y_train, num_classes=40)
    y_val = np_utils.to_categorical(y_val, num_classes=40)
    y_test = np_utils.to_categorical(y_test, num_classes=40)
    model = build_model()
    model = train(X_train, y_train, model)
    complete_accuary(X_test, y_test, model)


if __name__ == "__main__":
    start()

