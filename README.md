# Musical-Therapy-Using-Facial-Emotion-Recognition

## Overview
This project utilizes the FER2013 dataset to build a model that classifies facial expressions as either "Happy" or "Sad." Due to GPU limitations, only images labeled as "Happy" and "Sad" were included in the dataset. The model achieves an accuracy of 92%.

## Business Understanding
Emotion recognition is an important task in various applications such as human-computer interaction, mental health analysis, and security systems. Understanding and predicting emotions accurately can lead to more responsive and empathetic systems. This project aims to create a model that can effectively distinguish between happy and sad facial expressions, contributing to the broader goal of emotion recognition.

## Data Understanding
The FER2013 dataset consists of grayscale images of facial expressions categorized into seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. However, for this project, only images labeled as "Happy" and "Sad" were utilized. The dataset is well-suited for training convolutional neural networks (CNNs) due to its size and variety.

Key features include:
- **Grayscale Images**: 48x48 pixel resolution.
- **Emotions Used**: Happy, Sad.
- **Dataset Size**: 35,887 images (after filtering for "Happy" and "Sad").

## Modeling and Evaluation
A convolutional neural network (CNN) was implemented to classify images as "Happy" or "Sad." The model was trained on the filtered FER2013 dataset.

### Model Performance:
- **Accuracy**: 92%
The high accuracy indicates that the model is effective at distinguishing between happy and sad expressions.

## Conclusion
The project successfully demonstrates the use of a CNN for emotion classification on the FER2013 dataset. By focusing on only two emotions due to GPU limitations, the model achieved a strong performance, making it a useful tool for applications requiring basic emotion recognition capabilities.

## Future Work
Future enhancements could include:
- Expanding the model to classify all seven emotions in the FER2013 dataset.
- Improving the model's performance with a more robust GPU to handle the full dataset.
- Experimenting with different architectures and preprocessing techniques to further enhance accuracy.

