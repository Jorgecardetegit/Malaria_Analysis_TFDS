import numpy as np
import tensorflow as tf
from kerastuner.tuners import RandomSearch
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

class MyModelTuner:
    def __init__(self, config):
        self.config = config

    def build_model(self, hp):
        # Learning rate hyperparameter
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        # Convolutional layers Hyperparameters
        num_conv_layers = hp.Int('num_conv_layers', min_value=1, max_value=3, default=2)
        num_filters = hp.Int('num_filters', min_value=8, max_value=64, step=8)
        filter_size = hp.Int('filter_size', min_value=3, max_value=5)
        conv_regularizer = hp.Choice('conv_regularizer', values=[0.0, 0.01, 0.001, 0.0001])

        # Dense layers Hyperparameters
        num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=3, default=2)
        dense_units = hp.Int('dense_units', min_value=32, max_value=512, step=32)
        dense_regularizer = hp.Choice('dense_regularizer', values=[0.0, 0.01, 0.001, 0.0001])

        # Output layer Hyperparameters
        output_regularizer = hp.Choice('output_regularizer', values=[0.0, 0.01, 0.001, 0.0001])

        # Dropout Hyperparameters
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)

        # Learning Rate Scheduler Hyperparameters
        lr_schedule_factor = hp.Float('lr_schedule_factor', min_value=0.1, max_value=1.0, step=0.1)

        # Early Stopping Hyperparameters
        early_stopping_patience = hp.Int('early_stopping_patience', min_value=5, max_value=20, step=5)

        # ReduceLROnPlateau Hyperparameters
        reduce_lr_factor = hp.Float('reduce_lr_factor', min_value=0.1, max_value=1.0, step=0.1)
        reduce_lr_patience = hp.Int('reduce_lr_patience', min_value=2, max_value=10, step=1)
        reduce_lr_min_lr = hp.Float('reduce_lr_min_lr', min_value=1e-7, max_value=1e-3, step=1e-7)

        # Callbacks configuration
        def lr_schedule(epoch):
            return learning_rate * lr_schedule_factor**epoch

        lr_scheduler = LearningRateScheduler(lr_schedule)      

        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)

        reduce_lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=reduce_lr_min_lr)

        callbacks = [lr_scheduler, early_stopping, reduce_lr_plateau]
        
        # Feature extractor architecture
        feature_extractor_seq_model = tf.keras.Sequential([
            InputLayer(input_shape=(self.config["IM_HEIGHT"], self.config["IM_WIDTH"], 3)),
        ])

        for _ in range(num_conv_layers):
            feature_extractor_seq_model.add(Conv2D(filters=num_filters, kernel_size=filter_size, strides=1, padding='valid', activation='relu', kernel_regularizer=L2(l2=conv_regularizer)))
            feature_extractor_seq_model.add(BatchNormalization())
            feature_extractor_seq_model.add(MaxPooling2D(pool_size=2, strides=2))

        # Classifier architecture
        func_input = Input(shape=(self.config["IM_HEIGHT"], self.config["IM_WIDTH"], 3), name="Input Image")
        x = feature_extractor_seq_model(func_input)
        x = Flatten()(x)

        for _ in range(num_dense_layers):
            x = Dense(dense_units, activation="relu", kernel_regularizer=L2(l2=dense_regularizer))(x)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)

        func_output = Dense(self.config["num_classes"], activation="softmax", kernel_regularizer=L2(l2=output_regularizer))(x)

        # Model architecture
        model = Model(func_input, func_output, name="Lenet_model")

        # Model compilation
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)      #Add early stopping to the tuner to avoid overfitting

        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      callbacks=stop_early)
        return model

    def run_tuner(self):
        # RandomSearch tuner creation
        tuner = RandomSearch(
            self.build_model,
            objective='val_accuracy',
            max_trials=self.config["max_trials"],
            directory='my_dir',
            project_name='my_project'
        )

        # Parameter search
        tuner.search(x=self.config["x_train"],
                     y=self.config["y_train"],
                     validation_data=(self.config["x_val"], self.config["y_val"]),
                     epochs=self.config["epochs"])

        # Get the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        return best_hps

# Configuration Example:
# config = {
#    "IM_HEIGHT": 128,
#    "IM_WIDTH": 128,
#    "max_trials": 10,
#    "num_classes": 2,  # Number of classes for classification
#    "x_train": np.random.rand(100, 128, 128, 3),
#    "y_train": tf.keras.utils.to_categorical(np.random.randint(0, 10, size=(100,)), num_classes=2),
#    "x_val": np.random.rand(20, 128, 128, 3),
#    "y_val": tf.keras.utils.to_categorical(np.random.randint(0, 10, size=(20,)), num_classes=2),
#    "epochs": 10 }