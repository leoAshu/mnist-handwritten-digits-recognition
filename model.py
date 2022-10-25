from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

from keras.datasets import mnist
from keras.utils import to_categorical

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((60000, 28, 28, 1))
    x_train = x_train.astype('float32') / 255

    x_test = x_test.reshape((10000, 28, 28, 1))
    x_test = x_test.astype('float32') / 255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def fetch_model():
    model = define_model()
    
    model = train_model(model)
    
    return model


def define_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.summary()
    return model


def train_model(model):
    x_train, y_train, x_test, y_test = load_data()
    
    model.compile(
        optimizer='rmsprop', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=5, batch_size=64)

    test_loss, test_acc = model.evaluate(x_test, y_test)

    print(test_loss, test_acc)

    model.save('model/mnistModel.h5')


fetch_model()
