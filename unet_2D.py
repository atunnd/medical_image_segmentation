import tensorflow as tf
from load_data import CustomDataset
from tensorflow.keras.callbacks import TensorBoard
import os


inputs = tf.keras.Input(shape=(112, 112, 1))

# Block 1a
block1a = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
block1a = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(block1a)

# Block 2a
block2a = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(block1a)
block2a = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(block2a)
block2a = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(block2a)

# Block 3a
block3a = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(block2a)
block3a = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(block3a)
block3a = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(block3a)

# Block 4a
block4a = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(block3a)
block4a = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(block4a)
block4a = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(block4a)
block4a = tf.keras.layers.Dropout(0.5)(block4a)

# Block 5
block5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(block4a)
block5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(block5)
block5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(block5)
block5 = tf.keras.layers.Dropout(0.5)(block5)

# Block 4b
block4b = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(block5)
block4b = tf.keras.layers.Concatenate(axis=-1)([block4a, block4b])
block4b = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(block4b)
block4b = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(block4b)

# Block 3b
block3b = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(block4b)
block3b = tf.keras.layers.Concatenate(axis=-1)([block3a, block3b])
block3b = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(block3b)
block3b = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(block3b)

# Block 2b
block2b = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(block3b)
block2b = tf.keras.layers.Concatenate(axis=-1)([block2a, block2b])
block2b = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(block2b)
block2b = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(block2b)

# Block 1b
block1b = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(block2b)
block1b = tf.keras.layers.Concatenate(axis=-1)([block1a, block1b])
block1b = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(block1b)
block1b = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(block1b)

# Classification Layer
clf = tf.keras.layers.Conv2D(2, (1, 1), activation='softmax')(block1b)

# Create Model
model = tf.keras.Model(inputs=inputs, outputs=clf)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model

# Define log directory
log_dir = os.path.join("logs", "fit", "model")

# Create TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

Training_path = "../dataset/LA_dataset/Training Set"
Testing_path = "../dataset/LA_dataset/Testing Set"

data_train = CustomDataset(
    training_path=Training_path,
    testing_path=None,
    n_train_patients=2,
    input_size=112
)
x_train, y_train = data_train.get_data()
data_test = CustomDataset(
    training_path=None,
    testing_path=Testing_path,
    n_test_patients=1,
    load_train=False
)
x_test, y_test = data_test.get_data()

# Train the model with TensorBoard callback
model.fit(x_train,
          y_train,
          validation_data=(x_test, y_test),
          batch_size=8,
          epochs=10,
          verbose=1,
          callbacks=[tensorboard_callback])