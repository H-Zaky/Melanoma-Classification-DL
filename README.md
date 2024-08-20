# Melanoma Classification Using ResNet50
![image](https://github.com/user-attachments/assets/fe6ddbbb-9021-4a6a-afa3-ec24a42870fc)

This project demonstrates the use of transfer learning for melanoma classification using the ResNet50 model, a pre-trained convolutional neural network (CNN). The model is fine-tuned on a custom dataset of melanoma images to classify images into two categories.

## Project Overview

- **Dataset**: The project uses a dataset of melanoma images, split into training, validation, and test sets.
- **Model Architecture**: ResNet50, pre-trained on ImageNet, is used as the base model. Additional layers are added on top of it to fine-tune the model for the specific task of melanoma classification.
- **Training**: The model is trained using the training set, and the training process is monitored using early stopping based on validation loss.
- **Evaluation**: The model's performance is evaluated on a test set, and accuracy and loss metrics are recorded.
- **Visualization**: Training history, including accuracy and loss curves, is plotted to assess the model's performance.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn

## Installation

1. **Clone the repository:**

2. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Download the dataset:**
   - Download the melanoma dataset and place it in the `input/melanomadataset/` directory. The directory should have the following structure:
   
    ```
    input/melanomadataset/
    ├── train/
    │   ├── class1/
    │   └── class2/
    └── test/
        ├── class1/
        └── class2/
    ```

## Running the Project

1. **Prepare the Data:**
   - The dataset is loaded using `ImageDataGenerator`, which also handles data augmentation for the training set. The training data is split into training and validation subsets.

2. **Model Definition:**
   - ResNet50 is loaded with pre-trained weights from ImageNet. Additional dense and dropout layers are added to the model for fine-tuning.

3. **Compile the Model:**
   - The model is compiled using the Adam optimizer and categorical crossentropy as the loss function. Accuracy is used as the metric.

4. **Train the Model:**
   - The model is trained on the training data with early stopping enabled to prevent overfitting. The training process is validated on the validation set.

5. **Evaluate the Model:**
   - After training, the model is evaluated on the test set, and the final accuracy and loss are reported.

6. **Visualize the Results:**
   - The training and validation accuracy and loss are plotted to visualize the model's performance over time.

## Example Output

- **Training Output:**
  
    ```sh
    Epoch 1/50
    297/297 [======] - accuracy: 0.8235 - loss: 0.4703 - val_accuracy: 0.5291 - val_loss: 2.9570
    ...
    Epoch 11/50
    297/297 [======] - accuracy: 0.8725 - loss: 0.2969 - val_accuracy: 0.6921 - val_loss: 0.6087
    ```

- **Evaluation Output:**
  
    ```sh
    Test Loss: 0.26877355575561523
    Test Accuracy: 0.8794999718666077
    ```
