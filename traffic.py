# python3 traffic.py gtsrb

import cv2
import numpy as np
import os
import sys
import tensorflow as tf
 
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4 


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    # reference: https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
    # looping for all the folders in gtsrb (from 1st folder until (NUM-CATEGORIES-1) folder)
    # read the images in the folders in gtsrb using os.path directory
    for category in range(0, NUM_CATEGORIES):
        folder_name = str(category)
        path = os.path.join(data_dir, folder_name)
        length = len([name for name in os.listdir(path)])
        
        # for every image, resize to width and height = 30, 30
        # read the resize image using numpy = (30, 30, 3) where 3 is BGR
        # convert BGR to RGB using openCV 
        for item in os.listdir(path):
    
            img_received = cv2.imread(os.path.join(data_dir, folder_name, item))
            img_resized = cv2.resize(img_received, (IMG_WIDTH, IMG_HEIGHT))
            img = np.array(img_resized)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

            # add to images list and labels list
            images.append(image)
            labels.append(category)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([

        # Convolutional layer. Learn 32 different filters using a 3x3 kernel
        # input_shape = (30, 320)
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional layer and max-pooling layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        # Hidden layer with 128 units
        # Randomly dropout half of the units in hidden layer
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Softmax activation will turn the output into a probability distribution 
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return (model)

if __name__ == "__main__":
    main()
