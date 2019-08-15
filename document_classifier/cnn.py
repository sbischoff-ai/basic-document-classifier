"""Train and use a CNN for document image classification."""

import os
from io import BytesIO
import json

import numpy as np
from PIL import Image
from keras.callbacks import History
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.initializers import Constant
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow import Graph, Session

class CNN:
    """
    Predict document classes based on images and train the underlying model with training data.
    """

    input_size: tuple = (384, 384) # document aspect ratio is warped

    def __init__(self, class_number: int) -> None:
        self.classes = [f"class_{i}" for i in range(class_number)]
        self.is_binary = class_number == 2

        self.graph = Graph()
        with self.graph.as_default():
            self.session = Session()
            with self.session.as_default():
                model = Sequential()

                # Basic LeNet/AlexNet convolution layers

                model.add(Conv2D(
                    16,
                    kernel_size=(11, 11),
                    strides=(4,4),
                    input_shape=(self.input_size[0], self.input_size[1], 1),
                    kernel_initializer='he_uniform',
                    bias_initializer=Constant(0.1)
                ))
                model.add(Activation('relu'))
                # consider replacing pooling layers with batch normalization
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

                # Basic 2 layer classification with high dropout
                model.add(Dense(
                    16,
                    kernel_initializer='random_uniform',
                    bias_initializer=Constant(0.1)
                ))
                model.add(Activation('relu'))
                model.add(Dropout(0.5)) # Overfitting prevention and increased non-linearity
                model.add(Dense(1 if self.is_binary else class_number))
                model.add(Activation('sigmoid' if self.is_binary else 'softmax'))

                self.model = model
                # compile model
                self.model.compile(
                    loss='binary_crossentropy' if self.is_binary else 'categorical_crossentropy',
                    optimizer=SGD(lr=0.01, decay=1e-6), 
                    metrics=['accuracy']
                )

    def train(self, batch_size: int, epochs: int, train_data_path: str, test_data_path: str = None) -> History:
        """Train the CNN and return a History object with the training metrics.

        This method expects training and test data directories to contain subfolders
        corresponding to the different document classes.
        """
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
        self.classes = list(train_generator.class_indices.keys())
        test_generator = ImageDataGenerator(
            rescale=1./255
        ).flow_from_directory(
            test_data_path,
            target_size=self.input_size,
            batch_size=batch_size,
            class_mode=class_mode,
            color_mode='grayscale'
        ) if test_data_path is not None else None
        with self.graph.as_default():
            with self.session.as_default():
                return self.model.fit_generator(
                    train_generator,
                    steps_per_epoch=3000 // batch_size,
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=800 // batch_size if test_generator is not None else None
                )

    def save(self, model_path: str) -> None:
        """Persist the weights and metadata for a trained CNN on the file system."""
        metadata_string = json.dumps({ "classes": self.classes })
        with open(os.path.join(model_path, "metadata.json"), "w") as metadata_file:
            metadata_file.write(metadata_string)
        with self.graph.as_default():
            with self.session.as_default():
                self.model.save_weights(os.path.join(model_path, "weights.h5"))

    @classmethod
    def load(cls, model_path: str) -> "CNN":
        """Load a trained CNN model that was saved previously."""
        with open(os.path.join(model_path, "metadata.json"), "r") as metadata_file:
            metadata = json.loads(metadata_file.read())
        cnn = cls(len(metadata["classes"]))
        cnn.classes = metadata["classes"]
        with cnn.graph.as_default():
            with cnn.session.as_default():
                cnn.model.load_weights(os.path.join(model_path, "weights.h5"))
                cnn.model._make_predict_function() # necessary to prevent some tf bugs
        return cnn

    def predict(self, image) -> tuple:
        """Return the predicted class and confidence score for a given document image."""
        if type(image) == bytes: # allow to pass binary image file content
            img = Image.open(BytesIO(image))
            img = img.convert("L").resize(self.input_size) # convert("L") -> grayscale
        else: # otherwise expect filepath
            img = load_img(image, color_mode='grayscale', target_size=self.input_size)
        data = img_to_array(img)/255 # normalize pixel intensity -> [0,1]
        data = data.reshape((1,) + data.shape)
        with self.graph.as_default():
            with self.session.as_default():
                prediction = self.model.predict(data)
        # generate and return the (class, confidence) tuple
        if self.is_binary:
            if prediction[0][0] <= 0.5:
                return (self.classes[0], float(1.0 - prediction[0][0]))
            return (self.classes[1], float(prediction[0][0]))
        return (self.classes[np.argmax(prediction[0])], float(np.max(prediction[0])))
