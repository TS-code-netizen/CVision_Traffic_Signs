# Computer Vision: Traffic Signs

## Summary
A computer vision project using Convolutional Neural Network (CNN) to recognize and distinguish road signs. This project is a short implementation on developing an understanding with the environment using digital images, either stop signs, speed limit signs, yield signs, and more. 
The dataset consists all the labelled road signs images derived from German Traffic Sign Recognition Benchmark (GTSRB) with its link:
- https://benchmark.ini.rub.de/?section=gtsrb&subsection=news 
A total of 43 different road signs will be used to build a neural network using Tensorflow 

## Technologies used:
```
Python 
sklearn.model_selection
numpy
```
Algorithm/framework:
```
Tensorflow
Convolutional Neural Network (CNN)
Keras
```

## How this project works?
The data will first be process and read into memory. The data will be split into testing and training set via #train_test_split from sklearn. Apply CNN to build a compiled neural network model based on different approaches (see below). This neural network then fitted in training set. The model will be used to evaluate neural network performance using testing data. 

## Different Experiment Approaches 

For get_model function, I tried on different approaches:
1.  convolutional and pooling layers = 1, 1
    numbers and sizes of filters for convolutional layers = 32, (3, 3)
    pool sizes = (2, 2)
    numbers and sizes of hidden layers = 1, 128
    dropout = 0.5

    Result: 333/333 - 2s - loss: 3.5009 - accuracy: 0.0561
    Conclusion: The feature map might be very big 

2.  convolutional and pooling layers = 2, 2
    others the same

    Result: 333/333 - 2s - loss: 0.1544 - accuracy: 0.9598
    Conclusion: Improve a lot. Good model

3.  pool sizes = (4, 4)
    others the same

    Result: 333/333 - 2s - loss: 3.4945 - accuracy: 0.0550
    Conclusion: Accuracy dropped. Might caused by the big size of max pooling that create the missing of some feature during extraction. 

4.  numbers and sizes of filters for convolutional layers = 32, (2, 2)
    pool sizes = (2, 2)
    others the same

    Result: 333/333 - 2s - loss: 0.4621 - accuracy: 0.8535
    Conclusion: The filter size is too small. This might affect the feature extraction.

5.  numbers and sizes of filters for convolutional layers = 32, (3, 3)
    numbers and sizes of hidden layers = 1, 150
    others the same

    Result: 333/333 - 2s - loss: 0.2871 - accuracy: 0.9084
    Conclusion: Improved the size of hidden layer units to 150 improved the accuracy but not accurate enough.

6.  numbers and sizes of hidden layers = 2, 128
    others the same

    Result: 333/333 - 2s - loss: 0.1301 - accuracy: 0.9687 - 2s/epoch
    Conclusion: Adding a hidden layer doubles up the accuracy for every iteration. 

## Conclude My Model Choice

Model 6 is my final choice:
- convolutional and pooling layers = 2, 2
- numbers and sizes of filters for convolutional layers = 32, (3, 3)
- pool sizes = (2, 2)
- numbers and sizes of hidden layers = 2, 128
- dropout = 0.5

## Requirement:
1. Install python3 in Visual Studio Code
2. Install tensorflow
``pip3 install tensorflow``
3. Install sklearn
``pip install -U scikit-learn``
5. Install cv2
``pip install opencv-python`` 
7. Download the whole package in this master branch
