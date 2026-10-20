# 1. Artificial Intelligence, Data Science, Machine Learning and Deep Learning:
# Write a Python program to design, visualize, implement, train, and evaluate Convolutional Neural 
# Network (CNN), and Recurrent Neural Network (RNN) models using the Mushroom/Fungi 
# Classification dataset. Convert the categorical features into a suitable numerical format and 
# prepare/reshape the data according to the input requirements of the CNN and RNN models. Perform the 
# experiments according to the specifications given in the table below. Evaluate the performance of each 
# model using the specified evaluation metric, compare the results, and recommend the most suitable 
# deep learning model that provides the best classification performance. Split the dataset into 70% 
# training and 30% testing data. 

# task: For a RNN model: Model the Batch_Size will be 40 and  Epochs is 20 and Different Optimizers will be RMSprop.


import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, SimpleRNN, Dense, Dropout
)

# Step 1: Reshape data for RNN
X_train_rnn = X_train.reshape(
    X_train.shape[0], X_train.shape[1], 1
)

X_test_rnn = X_test.reshape(
    X_test.shape[0], X_test.shape[1], 1
)

# Step 2: Build RNN model
tf.keras.utils.set_random_seed(42)

rnn_model = Sequential([
    Input(shape=(X_train_rnn.shape[1], 1)),
    SimpleRNN(32, activation="tanh"),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation="softmax")
])

rnn_model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("RNN Model Summary:")
rnn_model.summary()

# Step 3: Visualize RNN architecture
tf.keras.utils.plot_model(
    rnn_model,
    to_file="rnn_architecture.png",
    show_shapes=True,
    show_layer_names=True
)

from IPython.display import display, Image
display(Image(filename="rnn_architecture.png"))

# Step 4: Train RNN
rnn_history = rnn_model.fit(
    X_train_rnn,
    y_train,
    batch_size=40,
    epochs=20,
    validation_split=0.10,
    verbose=1
)

# Step 5: Plot training accuracy and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(rnn_history.history["accuracy"], label="Training Accuracy")
plt.plot(rnn_history.history["val_accuracy"], label="Validation Accuracy")
plt.title("RNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(rnn_history.history["loss"], label="Training Loss")
plt.plot(rnn_history.history["val_loss"], label="Validation Loss")
plt.title("RNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Step 6: Evaluate RNN on test data
rnn_test_loss, rnn_test_accuracy = rnn_model.evaluate(
    X_test_rnn, y_test, verbose=0
)

print("\nRNN Test Loss:", rnn_test_loss)
print("RNN Test Accuracy:", rnn_test_accuracy)
