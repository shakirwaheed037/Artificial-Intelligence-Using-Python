
# 1. Artificial Intelligence, Data Science, Machine Learning and Deep Learning:
# Write a Python program to design, visualize, implement, train, and evaluate Convolutional Neural 
# Network (CNN), and Recurrent Neural Network (RNN) models using the Mushroom/Fungi 
# Classification dataset. Convert the categorical features into a suitable numerical format and 
# prepare/reshape the data according to the input requirements of the CNN and RNN models. Perform the 
# experiments according to the specifications given in the table below. Evaluate the performance of each 
# model using the specified evaluation metric, compare the results, and recommend the most suitable 
# deep learning model that provides the best classification performance. Split the dataset into 70% 
# training and 30% testing data. 

# task: For a CNN model: Model the Batch_Size will be 30 and  Epochs is 10 and Different Optimizers will be Adam.

#task no 3: Generate the visualization & Plotting of the given task using matplotlib  library. 
# a) Generate Predictions of both Models.
# b) Compare CNN, and RNN Performance.
# c) Report which model achieved the higher accuracy on the test data..
# d) Generate the Confusion Matrices of both Models.
# e) Generate the Classification Reports of both Models.




# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout
)

# Step 2: Load dataset
df = pd.read_csv("mushrooms.csv")

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nColumns:")
print(df.columns.tolist())

# Step 3: Separate features and target
target_col = df.columns[0]

X = df.drop(columns=[target_col]).fillna("Unknown").astype(str)
y = df[target_col].astype(str)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nTarget classes:", label_encoder.classes_)

# Step 4: Split data into 70% training and 30% testing
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train_raw))
print("Testing samples:", len(X_test_raw))

# Step 5: One-hot encode categorical features
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
    dtype=np.float32
)

X_train = encoder.fit_transform(X_train_raw)
X_test = encoder.transform(X_test_raw)

# Step 6: Reshape data for CNN
X_train_cnn = X_train.reshape(
    X_train.shape[0], X_train.shape[1], 1
)

X_test_cnn = X_test.reshape(
    X_test.shape[0], X_test.shape[1], 1
)

# Step 7: Build CNN model
tf.keras.utils.set_random_seed(42)

cnn_model = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),
    Conv1D(32, kernel_size=3, activation="relu", padding="same"),
    GlobalMaxPooling1D(),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation="softmax")
])

cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Model Summary:")
cnn_model.summary()

# Step 8: Visualize CNN architecture
tf.keras.utils.plot_model(
    cnn_model,
    to_file="cnn_architecture.png",
    show_shapes=True,
    show_layer_names=True
)

from IPython.display import display, Image
display(Image(filename="cnn_architecture.png"))

# Step 9: Train CNN
cnn_history = cnn_model.fit(
    X_train_cnn,
    y_train,
    batch_size=30,
    epochs=10,
    validation_split=0.10,
    verbose=1
)

# Step 10: Plot training accuracy and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(cnn_history.history["accuracy"], label="Training Accuracy")
plt.plot(cnn_history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history["loss"], label="Training Loss")
plt.plot(cnn_history.history["val_loss"], label="Validation Loss")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Step 11: Evaluate CNN on test data
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(
    X_test_cnn, y_test, verbose=0
)

print("\nCNN Test Loss:", cnn_test_loss)
print("CNN Test Accuracy:", cnn_test_accuracy)
