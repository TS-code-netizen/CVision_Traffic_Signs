






## Different Experimentation Approaches 

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

**Conclusion**

Model 6 is my final choice
- convolutional and pooling layers = 2, 2
- numbers and sizes of filters for convolutional layers = 32, (3, 3)
- pool sizes = (2, 2)
- numbers and sizes of hidden layers = 2, 128
- dropout = 0.5
