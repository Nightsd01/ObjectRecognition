# ObjectRecognition
Uses the iOS device camera and a custom deep learning algorithm to perform real time object recognition. This neural network achieves 68.8% accuracy. In 2015, the [best](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d313030) algorithms hit 76%, so this is quite good for a simple convolutional neural network. 

I designed & trained a convolutional neural network on the CIFAR-100 database of object images for 100 different classes of objects (such as trains, aircraft, different types of flowers).

The neural network is run uses iOS' new CoreML to efficiently run in real time.

## Testing the Model
To test the iOS application, I would recommend using Google Images, trying categories like 'sunflower', 'rose', 'wolf', 'clouds', and so on. Since the algorithm currently doesn't apply bounding box detection, you need to ensure that the subject is mostly centered inside of the image square. For example, the algorith, currently wouldn't be able to detect a 'flower' in an image if that flower is a relatively tiny part of the image. In the future, I will be adding bounding box detection & cropping.

## Training the Model
I have included the tensormodel.mlmodel file in the DeepLearning directory. This is a fully trained model. However, if you would like to train this model on your own computer, you can find the files in the TrainModel directory.

### Required Python 2.7 Packages:
* Keras (2.0.8)
* TensorFlow (1.3.0)
* coremltools (0.6.3)

The model requires some training data and will download it automatically for you if it is not present on your machine. This is not the standard CIFAR-100 format, I have converted it to .npy files so that there is no further normalization/transformations needed on the dataset.

The model takes a bit over an hour on my machine with a GTX Titan X (Pascal). Your mileage may vary. The model, with current hyperparameters, will achieve 68.8% accuracy on the test set. There are certainly better neural network architectures that can be used (ie. residual neural networks), and this is a simple convolutional neural network.

Eventually I will run a hyperparameter search to find the most optimal hyperparameters. The current architecture/hyperparams were simply found with rough trial-and-error. 

The algorithm currently uses a Softmax activation function on the output layer. This means that we are essentially forcing the classifier to classify whatever it sees. Currently, if no single output exceeds 0.5, the iOS application will show 'none'. If you would like the neural network to better handle situations where it isn't being shown anything it was trained on, you can switch the output layer to use the sigmoid activation function instead.


![alt text](https://i.imgur.com/26U7QnW.jpg)
