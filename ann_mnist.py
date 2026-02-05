import numpy as np # Import the numpy library for numerical operations
from tensorflow.keras.datasets import mnist # Import the MNIST dataset from Keras
from tensorflow.keras.models import Sequential # Import the Sequential model from Keras
from tensorflow.keras.layers import Dense, Flatten # Import Dense and Flatten layers from Keras
from tensorflow.keras.utils import to_categorical # Import to_categorical for one-hot encoding

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data() # Load the MNIST dataset into training and testing sets

# Normalize (0-255 -> 0-1)
x_train = x_train.astype('float32') / 255 # Normalize training images to the range [0, 1]
x_test = x_test.astype('float32') / 255 # Normalize testing images to the range [0, 1]

# One-hot encode labels
y_train = to_categorical(y_train, 10) # One-hot encode training labels (0-9)
y_test = to_categorical(y_test, 10) # One-hot encode testing labels (0-9)

# ANN model
model = Sequential() # Create a Sequential model
model.add(Flatten(input_shape=(28, 28))) # Add a Flatten layer to reshape the 28x28 images into a 1D array
model.add(Dense(512, activation='relu')) # Add a Dense layer with 512 units and ReLU activation
model.add(Dense(256, activation='relu')) # Add another Dense layer with 256 units and ReLU activation
model.add(Dense(10, activation='softmax')) # Add a Dense layer with 10 units (for 10 classes) and softmax activation

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metric

# Train
history = model.fit(x_train, y_train, epochs=10, batch_size=200, validation_split=0.2, verbose=2) # Train the model for 10 epochs with a batch size of 200 and 20% validation split

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0) # Evaluate the model on the test set
print("Test accuracy:", test_acc) # Print the test accuracy

fig, axes = plt.subplots(1, 5, figsize=(12, 12))

# Insert images
for i in range(5):
    axes[i].imshow(x_train[i])

plt.show()

fig, axes = plt.subplots(1, 5, figsize=(12, 12))

# Insert images
for i in range(5):
    axes[i].imshow(x_test[i])

plt.show()

import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
