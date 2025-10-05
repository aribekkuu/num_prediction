# MNIST Digit Recognition with TensorFlow

This project demonstrates how to build and train simple neural networks using TensorFlow and Keras to recognize handwritten digits from the MNIST dataset. The notebook walks through preprocessing the data, creating models, training, evaluating, and visualizing predictions.

## What the Model Does

The goal of this project is to classify 28x28 grayscale images of handwritten digits (0 through 9). The model learns from a labeled dataset of digits and attempts to correctly predict the digit in unseen images.

We use the MNIST dataset, which is preloaded in TensorFlow and consists of 60,000 training images and 10,000 testing images. Each image is represented as a 28x28 array of pixel values between 0 and 255. These are normalized to a range of 0 to 1 to improve training performance.

### Model Architecture

We start by flattening each 28x28 image into a 1D array of 784 pixels. Then we pass it through one or more dense (fully connected) layers. Two types of architectures are explored:

1. A simple model with a single dense layer using sigmoid activation.
2. A deeper model with a hidden layer of 100 neurons and ReLU activation, followed by a 10-neuron output layer with softmax-like behavior using sigmoid.

Finally, we use a more structured approach with the `Flatten` layer, which allows working with 2D inputs more cleanly.

### Training and Evaluation

The model is compiled with the Adam optimizer and trained using sparse categorical crossentropy as the loss function, which is appropriate for multi-class classification tasks with integer labels. Accuracy is used as the primary evaluation metric.

The model is trained over multiple epochs, where it iteratively adjusts weights to minimize the error on the training set. After training, the model is tested on unseen data (test set), and predictions are compared to actual labels.

### Visualization

We use `matplotlib` to show example input images and predicted outputs, and `seaborn` to display a confusion matrix that highlights where the model is making correct and incorrect predictions.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn

## How to Run

1. Clone the notebook or open it in Google Colab.
2. Install dependencies:
   ```bash
   pip install tensorflow matplotlib seaborn
Run all cells in order.
You will see training progress, test accuracy, and a confusion matrix showing how well the model performs on each digit.
