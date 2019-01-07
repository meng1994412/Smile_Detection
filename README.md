# Smile Detection
## Project Objectives
Implement a convolutional neural network capable of detecting a person smiling or not:
* Constructed LeNet architecture from scratch.
* Trained a model on a dataset of images that contain faces of people who are smiling or not smiling.
* Developed a script to detect smile in real-time.

## Language / Packages Used
* Python 3.5
* [OpenCV](https://docs.opencv.org/3.4.4/) 3.4.4
* [keras](https://keras.io/) 2.2.4
* [Imutils](https://github.com/jrosebr1/imutils)
* [NumPy](http://www.numpy.org/)

## Approaches
The dataset, named SMILES, comes from Daniel Hromada (check [reference](https://github.com/hromi/SMILEsmileD)). There are 13,165 images in the dataset, where each image has a dimension of 64x64x1 (grayscale). And the images in the dataset are tightly cropped around the face.

[//]: # (Image References)

[image1]: ./dataset/SMILEsmileD/SMILEs/positives/positives7/3.jpg
[image2]: ./dataset/SMILEsmileD/SMILEs/positives/positives7/6.jpg
[image3]: ./dataset/SMILEsmileD/SMILEs/positives/positives7/13.jpg
[image4]: ./dataset/SMILEsmileD/SMILEs/positives/positives7/15.jpg
[image5]: ./dataset/SMILEsmileD/SMILEs/positives/positives7/16.jpg
[image6]: ./dataset/SMILEsmileD/SMILEs/negatives/negatives7/4.jpg
[image7]: ./dataset/SMILEsmileD/SMILEs/negatives/negatives7/5.jpg
[image8]: ./dataset/SMILEsmileD/SMILEs/negatives/negatives7/7.jpg
[image9]: ./dataset/SMILEsmileD/SMILEs/negatives/negatives7/8.jpg
[image10]: ./dataset/SMILEsmileD/SMILEs/negatives/negatives7/9.jpg
[training-plot]: ./output/training_loss_and_accuracy_plot.png
[evaluation]: ./output/evaluation.png

The Figure 1 shows some examples of smiling image, and Figure 2 shows some example of not smiling image.

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

Figure 1: Positive example of the dataset (smiling).

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

Figure 2: Negative example of the dataset (not smiling).

## Results
### Build the LeNet architecture from scratch
The LeNet architecture can be found in `lenet.py` inside `pipeline/nn/conv/` directory. The input to the model includes dimensions of the image (height, width, the depth), and number of classes. In this project, the input would be (width = 28, height = 28, depth = 1, classes = 2).

Table 1 demonstrates the architecture of LeNet. The activation layer is not shown in the table, which should be one after each `CONV` layer. The `ReLU` activation function is used in the project.

| Layer Type    | Output Size   | Filter Size / Stride  |
| ------------- |:-------------:| ---------------------:|
| Input Image   | 28 x 28 x 1   |                       |
| CONV          | 28 x 28 x 20  | 5 x 5, K = 20         |
| POOL          | 14 x 14 x 20  | 2 x 2                 |
| CONV          | 14 x 14 x 50  | 3 x 3, K = 50         |
| POOL          | 7 x 7 x 50    | 2 x 2                 |
| FC            | 500           |                       |
| softmax       | 2             |                       |

Table 1: Summary of the LeNet architecture.

### Train the Smile CNN
The `train_model.py` is used for the training process. The weighted model will be saved after training ([chere here](https://github.com/meng1994412/Smile_Detection/blob/master/output/lenet.hdf5)).The saved model can be used for detecting smile in real-time later.

Figure 3 shows the plot of loss and accuracy for the training and validation set. As we can see from the figure, validation loss past 6th epoch starts to stagnate. Further training past 15th epoch may result in overfitting. Implement data augmentation on training set would be a good future "next-step" plan.

![alt text][training-plot]

Figure 3: Plot of loss and accuracy for the training and validation set.

Figure 4 illustrates the evaluation of the network, which obtains about 92% classification accuracy on validation set.

![alt text][evaluation]

Figure 4: Evaluation of the network.

### Run the Smile CNN in real-time
