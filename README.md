# Handwritten Digit Recognizer

This project aims to develop a Handwritten Digit Recognizer using Convolutional Neural Networks (CNN) with the help of TensorFlow and Keras libraries. The CNN model is trained and tested on the MNIST dataset, which consists of 60,000 handwritten images of digits ranging from 0 to 9. The project also includes a web application deployed on the Heroku platform to allow users to draw digits and recognize them in real-time.

## Dataset

The MNIST dataset is used for training and testing the model. It includes 60,000 28x28 grayscale images of digits, along with a test set of 10,000 images. It is a widely-used dataset in computer vision and serves as a benchmark for classification algorithms. More details about the dataset can be found [here](http://yann.lecun.com/exdb/mnist/index.html).

## Libraries Used

The main libraries used in this project are:

- TensorFlow: An open-source framework developed by Google researchers for machine learning and deep learning tasks.
- Keras: A high-level neural network API that runs on top of TensorFlow, providing a clean and easy way to create deep learning models.
- Pandas: A Python library for data analysis, providing data structure and tools for faster data analysis, cleaning, and preprocessing.
- NumPy: A Python library for scientific computing that provides support for n-dimensional arrays and mathematical functions.
- Matplotlib: A Python library for creating visualizations such as 2D plots and graphs.

## The Model: Convolutional Neural Networks (CNN)

The model architecture consists of an 8-layer CNN. It includes Conv2D, MaxPooling2D, Dropout, Flatten, and Dense layers. The input layer has 32 neurons, and the output layer has 10 neurons representing the 10 different digit classes (0-9).

- Layer 1: Convolutional layer with relu activation function, filter size 3x3, and 32 filters.
- Layer 2: Max pooling layer with a size of 2x2 and stride of 2.
- Layer 3: Convolutional layer with relu activation function, filter size 3x3, and 48 filters.
- Layer 4: Max pooling layer with a size of 2x2 and stride of 2.
- Layer 5: Dropout layer to prevent overfitting.
- Layer 6: Flatten layer to convert the data into a 1-dimensional array.
- Layer 7: Dense layer with relu activation function and 500 units.
- Layer 8: Final dense layer with softmax activation function for digit classification.

## Training and Testing

The model is trained on 50,000 images and validated on 10,000 images from the MNIST dataset. The training process involves fitting the model for 15 epochs with a batch size of 128. The model achieves an impressive testing accuracy of 99.21%.

## Deployment of the Model

A web application has been developed using HTML, CSS, and JavaScript for the deployment of the model. The application allows users to draw any digit on a canvas and predict the digit using the trained model. The canvas image is preprocessed, fed into the model, and the predicted digit is displayed on the screen.

## Conclusion

This project successfully develops a Handwritten Digit Recognizer using CNN and achieves high accuracy in recognizing handwritten digits. The CNN model is trained and tested on the MNIST dataset. The project showcases the application of deep learning techniques and demonstrates the potential for real-world applications such as assisting visually impaired individuals, human-robot interaction, and automatic data entry. This project serves as a foundation for further exploration of artificial intelligence and computer vision.
