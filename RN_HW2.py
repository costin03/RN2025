import numpy as np
import pickle
import pandas as pd

train_file = "/kaggle/input/fii-nn-2025-homework-2/extended_mnist_train.pkl"
test_file = "/kaggle/input/fii-nn-2025-homework-2/extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)

train_data = []
train_labels = []
for image, label in train:
    train_data.append(image.flatten())
    train_labels.append(label)

test_data = []
for image, label in test:
    test_data.append(image.flatten())

X_train = np.array(train_data, dtype=np.float32) / 255.0
y_train = np.array(train_labels, dtype=np.int32)
X_test = np.array(test_data, dtype=np.float32) / 255.0

def initialize_parameters(input_dim=784, num_classes=10):
    W = np.random.randn(input_dim, num_classes) * 0.01
    b = np.zeros(num_classes)
    return W, b


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(X, W, b):
    z = np.dot(X, W) + b
    predictions = softmax(z)
    return predictions


def cross_entropy_loss(predictions, labels, num_classes=10):
    batch_size = labels.shape[0]
    targets_binary = np.zeros((batch_size, num_classes))
    targets_binary[np.arange(batch_size), labels] = 1

    epsilon = 1e-10
    loss = -np.sum(targets_binary * np.log(predictions + epsilon)) / batch_size
    return loss


def backward(X, predictions, labels, W, b, learning_rate, num_classes=10):
    batch_size = X.shape[0]

    targets_binary = np.zeros((batch_size, num_classes))
    targets_binary[np.arange(batch_size), labels] = 1

    gradient = predictions - targets_binary

    dW = np.dot(X.T, gradient) / batch_size
    db = np.sum(gradient, axis=0) / batch_size

    W = W - learning_rate * dW
    b = b - learning_rate * db

    return W, b


def predict(X, W, b):
    predictions = forward(X, W, b)
    return np.argmax(predictions, axis=1)

def accuracy(X, labels, W, b):
    predicted_classes = predict(X, W, b)
    return np.mean(predicted_classes == labels)

def train(X, y, epochs=100, batch_size=256, validation_split=0.2, learning_rate=0.05):
    W, b = initialize_parameters(input_dim=X.shape[1], num_classes=10)

    m = X.shape[0]
    val_size = int(m * validation_split)

    indices = np.random.permutation(m)
    X_val = X[indices[:val_size]]
    y_val = y[indices[:val_size]]
    X_train_split = X[indices[val_size:]]
    y_train_split = y[indices[val_size:]]

    for epoch in range(epochs):
        perm = np.random.permutation(X_train_split.shape[0])
        X_shuffled = X_train_split[perm]
        y_shuffled = y_train_split[perm]

        for i in range(0, X_shuffled.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            predictions = forward(X_batch, W, b)
            W, b = backward(X_batch, predictions, y_batch, W, b, learning_rate)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_predictions = forward(X_val, W, b)
            val_loss = cross_entropy_loss(val_predictions, y_val)
            val_acc = accuracy(X_val, y_val, W, b)
            train_acc = accuracy(X_train_split, y_train_split, W, b)

            print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Train Acc: {train_acc:.4f}")

    return W, b


print("\nAntrenare Perceptron\n")

W, b = train(X_train, y_train, epochs=100, batch_size=256, learning_rate=0.05)

predictions = predict(X_test, W, b)

submission = {
    "ID": list(range(len(predictions))),
    "target": predictions.tolist()
}

df = pd.DataFrame(submission)
df.to_csv("submission.csv", index=False)