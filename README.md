# ObjectRecognition
Uses the iOS device camera and a custom deep learning algorithm to perform real time object recognition.

I designed & trained a convolutional neural network on the CIFAR-100 database of object images for 100 different classes of objects (such as trains, aircraft, different types of flowers).

The neural network is run uses iOS' new CoreML to efficiently run in real time.

## Training the Model
I have included the tensormodel.mlmodel file in the DeepLearning directory. This is a fully trained model. However, if you would like to train this model on your own computer, you can find the files in the TrainModel directory.

### Required Python 2.7 Packages:
* Keras (2.0.8)
* TensorFlow (1.3.0)
* coremltools (0.6.3)

The model requires some training data and will download it automatically for you if it is not present on your machine. This is not the standard CIFAR-100 format, I have converted it to .npy files so that there is no further normalization/transformations needed on the dataset.

The model takes a bit over an hour on my machine with a GTX Titan X (Pascal). Your mileage may vary. The model, with current hyperparameters, will achieve roughly 70% accuracy on the test set. There are certainly better neural network architectures that can be used (ie. residual neural networks), and this is a simple convolutional neural network.

Eventually I will run a hyperparameter search to find the most optimal hyperparameters. The current architecture/hyperparams were simply found with rough trial-and-error. 


![alt text](https://i.imgur.com/26U7QnW.jpg)
