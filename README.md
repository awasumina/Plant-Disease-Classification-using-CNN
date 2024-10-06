# Plant Disease Classification using CNN

This repository contains a Convolutional Neural Network (CNN) model built with TensorFlow and Keras to classify plant diseases using the [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset). The model is trained on images of plant leaves to identify various diseases. The dataset includes color, segmented, and grayscale images of healthy and unhealthy plants.

## Dataset

The dataset used for this project is from the Kaggle PlantVillage dataset, which can be downloaded [here](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).

It contains three types of images:
- **Color**
- **Segmented**
- **Grayscale**

The dataset is organized into different folders based on plant types and their health condition (e.g., "Grape___healthy").


## Project Structure

- **data_preprocessing**: Preprocesses the images for training, resizing them to a fixed size and rescaling the pixel values.
- **cnn_model**: Defines the CNN architecture using Keras layers, including Conv2D and MaxPooling layers.
- **training**: The model is trained using the preprocessed data with an 80-20 train-validation split.
- **evaluation**: Evaluation of the model using validation data, confusion matrix, and misclassified images.


## Model Architecture

The CNN architecture is built as follows:

- 2D Convolution layers with ReLU activation
- MaxPooling layers
- Dense layers for classification

The final layer uses softmax activation for multi-class classification.

## Results

- **Validation Accuracy**: 88.15%  
- **Loss**: 0.4389

## Conclusion

The CNN model provides a decent accuracy for plant disease classification. Future improvements can be made by adjusting hyperparameters, increasing epochs, or using transfer learning techniques like pre-trained models (e.g., VGG16, ResNet).
