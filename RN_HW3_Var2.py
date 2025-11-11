import numpy as np
import pickle
import pandas as pd

train_file = "/kaggle/input/fii-nn-2025-homework-3/extended_mnist_train.pkl"
test_file = "/kaggle/input/fii-nn-2025-homework-3/extended_mnist_test.pkl"

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


def initialize_parameters(input_size=784, hidden_size=100, output_size=10):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros(output_size)
    return W1, b1, W2, b2


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2, dropout_rate=0.3, training=True):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    dropout_mask = None
    if training:
        dropout_mask = (np.random.rand(*a1.shape) > dropout_rate) / (1 - dropout_rate)
        a1 = a1 * dropout_mask

    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2, dropout_mask


def backward(X, y_true, z1, a1, a2, dropout_mask, W1, b1, W2, b2, learning_rate):
    batch_size = y_true.shape[0]
    num_classes = a2.shape[1]

    y_one_hot = np.zeros((batch_size, num_classes))
    y_one_hot[np.arange(batch_size), y_true] = 1

    dz2 = a2 - y_one_hot
    dW2 = np.dot(a1.T, dz2) / batch_size
    db2 = np.sum(dz2, axis=0) / batch_size

    da1 = np.dot(dz2, W2.T)
    if dropout_mask is not None:
        da1 = da1 * dropout_mask

    dz1 = da1 * relu_derivative(z1)
    dW1 = np.dot(X.T, dz1) / batch_size
    db1 = np.sum(dz1, axis=0) / batch_size

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    return W1, b1, W2, b2


def cross_entropy_loss(predictions, labels):
    batch_size = labels.shape[0]
    num_classes = predictions.shape[1]

    y_one_hot = np.zeros((batch_size, num_classes))
    y_one_hot[np.arange(batch_size), labels] = 1

    epsilon = 1e-10
    loss = -np.sum(y_one_hot * np.log(predictions + epsilon)) / batch_size
    return loss


def predict(X, W1, b1, W2, b2):
    z1, a1, z2, predictions, dropout_mask = forward(X, W1, b1, W2, b2, training=False)
    return np.argmax(predictions, axis=1)


def accuracy(X, y, W1, b1, W2, b2):
    predictions = predict(X, W1, b1, W2, b2)
    return np.mean(predictions == y)


def train(X, y, epochs=50, batch_size=128, validation_split=0.2, learning_rate=0.05, dropout_rate=0.3):
    W1, b1, W2, b2 = initialize_parameters(input_size=784, hidden_size=100, output_size=10)

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

            z1, a1, z2, a2, dropout_mask = forward(X_batch, W1, b1, W2, b2, dropout_rate, training=True)
            W1, b1, W2, b2 = backward(X_batch, y_batch, z1, a1, a2, dropout_mask, W1, b1, W2, b2, learning_rate)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_predictions = forward(X_val, W1, b1, W2, b2, training=False)[3]
            val_loss = cross_entropy_loss(val_predictions, y_val)
            val_acc = accuracy(X_val, y_val, W1, b1, W2, b2)
            train_acc = accuracy(X_train_split, y_train_split, W1, b1, W2, b2)

            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Val Acc: {val_acc:.4f} - "
                  f"Train Acc: {train_acc:.4f}")

    print(f"\nEvaluare finala:")
    final_train_acc = accuracy(X_train_split, y_train_split, W1, b1, W2, b2)
    final_val_acc = accuracy(X_val, y_val, W1, b1, W2, b2)
    print(f"  Train Accuracy: {final_train_acc:.4f}")
    print(f"  Validation Accuracy: {final_val_acc:.4f}\n")

    return W1, b1, W2, b2



W1, b1, W2, b2 = train(
    X_train,
    y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    learning_rate=0.05,
    dropout_rate=0.3
)

predictions = predict(X_test, W1, b1, W2, b2)

predictions_csv = {
    "ID": [],
    "target": [],
}

for i, label in enumerate(predictions):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)

print(f"Submission salvat in 'submission.csv'")
