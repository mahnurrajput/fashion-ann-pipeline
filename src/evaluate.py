import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():
    model = tf.keras.models.load_model("models/model.h5")

    x_test = np.load("data/processed/x_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    os.makedirs("metrics", exist_ok=True)

    with open("metrics/metrics.json", "w") as f:
        json.dump(
            {
                "test_loss": float(loss),
                "test_accuracy": float(accuracy),
            },
            f,
            indent=4,
        )

    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.savefig("metrics/confusion_matrix.png")
    plt.close()

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("Metrics saved to metrics/metrics.json")
    print("Confusion matrix saved to metrics/confusion_matrix.png")


if __name__ == "__main__":
    main()