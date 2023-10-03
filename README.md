
# Malaria Detection with TensorFlow
Computer vision project focused on malaria detection using TensorFlow. This repository is organized into three key components:

## 1. Exploratory Data Analysis (EDA)
- **Dataset Analysis**: We've meticulously examined the dataset to understand its nuances, size, and characteristics.
- **Channel Visualization**: We've visually explored the three RGB channels of the images, uncovering potential patterns and variations.
- **Dimensionality Reduction**: We've applied dimensionality reduction techniques to distill the essential information from the dataset.
- **Feature Extraction**: We've brainstormed innovative feature extraction ideas to enhance model performance.
- **Data Augmentation**: We've considered data augmentation strategies to increase the robustness of our model.
## 2. Main Model Pipeline
- **Model Storage**: Data stored in a TFRecords format, ensuring efficient handling and processing.
- **Preprocessing**: Images have been standardized, resized to a uniform 256x256 size, and retained in RGB format. Augmentation wasn´t applied in this iteration.
- **Model Architecture**: LeNet-inspired model, structured into a feature extractor and a classifier. The feature extractor employs the sequential API, while the classifier is constructed using the functional API.
- **Hyperparameters**: We've defined and fine-tuned the following key hyperparameters: number of convolutional layers, filter sizes, dense layers, dropout rates and regularization techniques. 
- **Callbacks**: Learning rate schedules, early stopping, and ReduceLROnPlateau.
## 3. Hyperparameter Tuning Class
- **Model Builder**: Keras Tuner was used to systematically explore and optimize all relevant hyperparameters.
- **Tuner Definition**: The hyperparameter tuner is set up with the objective of maximizing validation accuracy. It performs up to 100 trials over 100 epochs, incorporating early stopping for efficiency.
