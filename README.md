#Age and Gender Prediction from Facial Images Using Deep Learning
## Objective:
#### The main goal of this project is to automatically predict a person’s age and gender from a facial image using a Convolutional Neural Network (CNN) and transfer learning. This task is a common real-world application of computer vision and has wide-ranging use cases, such as in digital advertising, smart surveillance systems, demographic analysis, and personalization engines.

## Dataset:
#### UTKFace Dataset: A publicly available dataset containing over 20,000 labeled facial images.

#### Each image filename encodes the age, gender, and race of the individual:
Example: 25_0_1_20170109150557335.jpg → Age: 25, Gender: 0 (Male), Race: 1

## Problem Type:
Multi-output model:

Age prediction: a regression task (predict a numerical value)

Gender prediction: a binary classification task (male/female)

Techniques and Workflow:
Image Preprocessing:

Images resized to 128x128 pixels

Normalization (pixel values scaled to [0, 1])

Labels extracted from filenames

One-hot encoding used for gender

Data Splitting:

Dataset split into training and testing sets (80/20)

Imbalance in gender data addressed via data augmentation

Image Augmentation:

Applied to training data for better generalization:

Rotation, zoom, horizontal shift, and flip

Model Architecture:

Used MobileNetV2 (a lightweight CNN pre-trained on ImageNet) as the base

Added custom fully connected layers for:

age_output: Dense layer with linear activation for age

gender_output: Dense layer with softmax activation for gender

The model was compiled with:

Loss: MSE for age, categorical crossentropy for gender

Metrics: MAE (mean absolute error) for age, accuracy for gender

Training:

Trained for up to 30 epochs with early stopping

Used Adam optimizer (learning rate = 0.0001)

Tracked training and validation loss/accuracy

Evaluation and Visualization:

Plotted age MAE and gender classification accuracy over epochs

Predicted age and gender for a sample test image

Results:
The model successfully learned to:

Predict age with reasonable MAE (e.g., ±5 years error range)

Predict gender with high accuracy (typically above 90%)

The use of transfer learning enabled efficient training even on limited data.

Possible Enhancements:
Implement Grad-CAM to visualize which parts of the face the model focuses on

Add a Streamlit web interface for uploading images and live predictions

Further fine-tune MobileNetV2 or experiment with models like ResNet or EfficientNet

Handle age outliers (e.g., very young or very old) through stratified sampling or loss weighting

Conclusion:
This project demonstrates a practical application of deep learning in the multi-task learning setting. By combining transfer learning with a dual-output architecture, it delivers both regression and classification results from facial data with good performance. It's a strong portfolio piece for showcasing expertise in image processing, neural networks, and real-world AI deployment.
