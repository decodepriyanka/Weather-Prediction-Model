import tensorflow as tf
import numpy as np
import cv2
import os


def train_save_model():
    train_dir = 'data/Train'
    test_dir = 'data/Test'

    IMAGE_SIZE = 224
    BATCH_SIZE = 64

    # pre=processing
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.1
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1
    )

    train_datagen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='training'
    )

    test_datagen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation'
    )

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64, padding='same', strides=2,
              kernel_size=3, activation='relu', input_shape=(224, 224, 3)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Conv2D(filters=32, padding='same',
              strides=2, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Conv2D(filters=32, padding='same',
              strides=2, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_datagen, epochs=20, validation_data=test_datagen)

    model.save(os.path.join('models', 'weather.h5'))


def predict():
    data_dir = 'data/Test'
    data_set = ['cloudy','water','green_area','desert']
        
    for image_class in os.listdir(data_dir):
        try:
            loaded_model = tf.keras.models.load_model('models/weather.h5')
            for image_file in os.listdir(os.path.join(data_dir, image_class)):
                file_src = os.path.join(data_dir, image_class, image_file)
                img = cv2.imread(file_src)
                resize = tf.image.resize(img, (224, 224))
                prediction = loaded_model.predict(np.expand_dims(resize/255, 0),batch_size=1, verbose=1)
                [answer] = np.argmax(prediction, axis=1)
                print('Image={} and prediction={}'.format(
                    image_class, data_set[answer]))
        except Exception as e:
            print(e)


def main():
    dirs = os.listdir(os.getcwd())
    if 'models' in  dirs:
        predict()
    else:
        train_save_model()
        predict()


main()