# 1. Artificial Intelligence, Data Science, Machine Learning and Deep Learning:
# Write a Python program to design, visualize, implement, train, and evaluate Convolutional Neural 
# Network (CNN), and Recurrent Neural Network (RNN) models using the Mushroom/Fungi 
# Classification dataset. Convert the categorical features into a suitable numerical format and 
# prepare/reshape the data according to the input requirements of the CNN and RNN models. Perform the 
# experiments according to the specifications given in the table below. Evaluate the performance of each 
# model using the specified evaluation metric, compare the results, and recommend the most suitable 
# deep learning model that provides the best classification performance. Split the dataset into 70% 
# training and 30% testing data.

#task: Generate the visualization & Plotting of the given task using matplotlib  library. 
# a) Generate Predictions of both Models.
# b) Compare CNN, and RNN Performance.
# c) Report which model achieved the higher accuracy on the test data..
# d) Generate the Confusion Matrices of both Models.
# e) Generate the Classification Reports of both Models.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Step 1: Generate predictions
cnn_probabilities = cnn_model.predict(X_test_cnn, verbose=0)
cnn_predictions = np.argmax(cnn_probabilities, axis=1)

rnn_probabilities = rnn_model.predict(X_test_rnn, verbose=0)
rnn_predictions = np.argmax(rnn_probabilities, axis=1)

print("First 10 actual labels:")
print(y_test[:10])

print("\nFirst 10 CNN predictions:")
print(cnn_predictions[:10])

print("\nFirst 10 RNN predictions:")
print(rnn_predictions[:10])

# Step 2: Calculate test accuracy
cnn_accuracy = accuracy_score(y_test, cnn_predictions)
rnn_accuracy = accuracy_score(y_test, rnn_predictions)

print("\nCNN Test Accuracy:", round(cnn_accuracy, 4))
print("RNN Test Accuracy:", round(rnn_accuracy, 4))

# Step 3: Compare accuracy using Matplotlib
models = ["CNN", "RNN"]
accuracies = [cnn_accuracy, rnn_accuracy]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies)

plt.title("CNN vs RNN Test Accuracy")
plt.xlabel("Model")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1.05)

for bar, score in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{score:.2%}",
        ha="center"
    )

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Step 4: Generate confusion matrices
class_names = [str(c) for c in label_encoder.classes_]
class_labels = np.arange(len(class_names))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    cnn_predictions,
    labels=class_labels,
    display_labels=class_names,
    cmap="Blues",
    ax=axes[0],
    colorbar=False
)

axes[0].set_title("CNN Confusion Matrix")

ConfusionMatrixDisplay.from_predictions(
    y_test,
    rnn_predictions,
    labels=class_labels,
    display_labels=class_names,
    cmap="Oranges",
    ax=axes[1],
    colorbar=False
)

axes[1].set_title("RNN Confusion Matrix")

plt.tight_layout()
plt.show()

# Step 5: Classification reports
print("\nCNN Classification Report:")
print(classification_report(
    y_test,
    cnn_predictions,
    labels=class_labels,
    target_names=class_names,
    zero_division=0
))

print("\nRNN Classification Report:")
print(classification_report(
    y_test,
    rnn_predictions,
    labels=class_labels,
    target_names=class_names,
    zero_division=0
))

# Step 6: Final results table
results = pd.DataFrame({
    "Model": ["CNN", "RNN"],
    "Optimizer": ["Adam", "RMSprop"],
    "Batch Size": [30, 40],
    "Epochs": [10, 20],
    "Test Accuracy": [cnn_accuracy, rnn_accuracy]
})

print("\nFinal Model Comparison:")
print(results.to_string(index=False))

# Step 7: Recommend the model based on test accuracy
if cnn_accuracy > rnn_accuracy:
    print("\nCNN achieved the higher test accuracy.")
elif rnn_accuracy > cnn_accuracy:
    print("\nRNN achieved the higher test accuracy.")
else:
    print("\nBoth models achieved the same test accuracy.")
