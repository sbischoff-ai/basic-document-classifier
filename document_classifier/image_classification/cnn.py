"""Train and use a CNN for document image classification."""

from io import BytesIO

import numpy as np
from PIL import Image
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.initializers import Constant
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

class CNN:
    """
    Predict document classes based on images and train the underlying model with examples.
    """

    input_size = (384, 384) # document aspect ratio is warped

    def __init__(self, classes: list,  weights_path=None):
        self.classes = classes
        self.is_binary = len(classes) == 2

        model = Sequential()

        # Basic LeNet or AlexNet convolution layers

        model.add(Conv2D(
            16,
            kernel_size=(11, 11),
            strides=(4,4),
            input_shape=(self.input_size[0], self.input_size[1], 1),
            kernel_initializer='he_uniform',
            bias_initializer=Constant(0.1)
        ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(Conv2D(
            16,
            kernel_size=(5, 5),
            strides=(1,1),
            kernel_initializer='he_uniform',
            bias_initializer=Constant(0.1)
        ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        # Flatten to 1D Feature Set
        model.add(Flatten())

        # Transform to probabilities in 2 classification layers
        model.add(Dense(
            16,
            kernel_initializer='random_uniform',
            bias_initializer=Constant(0.1)
        ))
        model.add(Activation('relu'))
        model.add(Dropout(0.5)) # Overfitting prevention and increased non-linearity
        model.add(Dense(1 if self.is_binary else len(classes))) # output shape!
        model.add(Activation('sigmoid' if self.is_binary else 'softmax'))

        self.model = model
        # compile model
        self.model.compile(
            loss='binary_crossentropy' if self.is_binary else 'categorical_crossentropy',
            optimizer=SGD(lr=0.01, decay=1e-6), 
            metrics=['accuracy']
        )

        if weights_path is not None:
            self.model.load_weights(weights_path)

    def train(self, batch_size, epochs, train_data_path, test_data_path=None):
        class_mode = "binary" if self.is_binary else "categorical"
        train_generator = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        ).flow_from_directory(
            train_data_path,
            target_size=self.input_size,
            batch_size=batch_size,
            class_mode=class_mode,
            color_mode='grayscale'
        )
        test_generator = ImageDataGenerator(
            rescale=1./255
        ).flow_from_directory(
            test_data_path,
            target_size=self.input_size,
            batch_size=batch_size,
            class_mode=class_mode,
            color_mode='grayscale'
        ) if test_data_path is not None else None
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=3000 // batch_size,
            epochs=epochs,
            validation_data=test_generator,
            validation_steps=800 // batch_size if test_generator is not None else None
        )

    def save(self, weights_path):
        self.model.save_weights(weights_path)

    def predict(self, image):
        if type(image) == bytes: # Allow to pass binary image file content
            img = Image.open(BytesIO(image))
            img.convert("L")
            img.resize(self.input_size)
        img = load_img(image, color_mode='grayscale', target_size=self.input_size)
        data = img_to_array(img)/255
        data = data.reshape((1,) + data.shape)
        prediction = self.model.predict(data)
        if self.is_binary:
            if prediction[0][0] <= 0.5:
                return self.classes[0]
            return self.classes[1]
        return self.classes[np.argmax(prediction[0])]
