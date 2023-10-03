# Malaria Analysis with TensorFlow

Computer vision project on the malarya dataset. This repositories contains 3 main files, an Exploratory Data Analysis about the dataset, the model pipeline where data is preprocessed and the Neural Network 
architecture is defined and constructed and an additional file which contains a class designed for tuning the hyperparameters defined in the main pipeline.

- **Exploratory Data Analysis (EDA)**: 
  General analysis of the dataset, visualization of the three RGB channels, dimensionality reduction techniques. feature extraction and data augmentation ideas.
  
- **Main model pipeline**:
    1. Model storage
       -  TFRecords (10 files)
       - 
    3. Preprocessing
       -  Images normalized and resized to a (256*256) size
       -  No agumentation applied
       -  Images will mantain RGB channel
         
    4. Model architecture
       LeNet model organized in a feature extractor and classifier. 
       -  Feature extractor: Built with sequential API
       -  Classifier: Built with functional API
    
    5. Hyperparameters
       -  Convolutional layers (number of conv layers, number of filters, filter size, L2 - Convolutional regularizer)
       -  Dense layers (number of dense layers, number of dense units, L2 - Dense regularizer)
       -  Output layer (L2 - Dense regularizer)
       -  Dropout rate
         
       -  Callbacks (lr_schedule_factor, early_stopping_patience, ReduceLROnPlateau)
       
- **Hyperparameter tunning class**:
   1. Model builder (Method: Keras Tuner - all hyperparameters included)
   2. Tuner definition
      - Objective: "Val acuracy"
      - MaxTrials: 100
      - Epochs: 100
      - Callbacks: EarlyStopping
