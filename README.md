# basic-document-classifier
A simple CNN for n-class classification of document images.

It doesn't take colour into account (it transforms to grayscale).
For small numbers of classes (2 to 4) this model can achieve > 90% accuracy with as little as 10 to 30 training images per class.
Training data can be provided in [any image format supported by *PIL*](https://pillow.readthedocs.io/en/5.1.x/handbook/image-file-formats.html).

## Installation

```pip install document-classifier```
or
```poetry add document-classifier```

## Usage

```python
from document_classifier import CNN

# Create a classification model for 3 document classes.
classifier = CNN(class_number=3)

# Train the model based on images stored on the file system.
training_metrics = classifier.train(
    batch_size=8,
    epochs=40,
    train_data_path="./train_data",
    test_data_path="./test_data"
)
# "./train_data" and "./test_data" have to contain a subfolder for
# each document class, e.g. "./train_data/letter" or "./train_data/report".

# View training metrics like the validation accuracy on the test data.
print(training_metrics.history["val_acc"])

# Save the trained model to the file system.
classifier.save(model_path="./my_model")

# Load the model from the file system.
classifier = CNN.load(model_path="./my_model")

# Predict the class of some document image stored in the file system.
prediction = classifier.predict(image="./my_image.jpg")
# The image parameter also taks binary image data as a bytes object.
```

The prediction result is a 2-tuple containing the document class label as a string and the confidence score as a float.

## TODO

The model architecture is fixed for now and geared towards smaller numbers of classes and training images.
I'm working on automatic scaling for the CNN.
