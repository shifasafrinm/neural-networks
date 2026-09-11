from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def prepare_output_path(path):
    """
    Create the parent output directory if necessary.
    """

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    return path


def plot_sample_images(
    X,
    y,
    label_to_character,
    output_path=None,
    number_of_samples=10
):
    """
    Display sample images from the dataset.

    Parameters
    ----------
    X : numpy.ndarray
        Flattened image data.

    y : numpy.ndarray
        Labels.

    label_to_character : callable
        Function for converting labels into characters.

    output_path : str, optional
        Location to save the figure.

    number_of_samples : int
        Number of images to display.
    """

    plt.figure(figsize=(12, 5))

    for i in range(number_of_samples):

        plt.subplot(2,5,i + 1)

        image = X[i].reshape(28, 28)

        plt.imshow(image, cmap="gray")

        plt.title(label_to_character(int(y[i])))

        plt.axis("off")

    plt.tight_layout()

    if output_path is not None:

        output_path = prepare_output_path(
            output_path
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

    plt.show()


def plot_training_history(history, output_path):
    """
    Plot training and validation loss and accuracy.

    Parameters
    ----------
    history : dict
        Training history returned by the model.

    output_path : str
        Destination image path.
    """

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1,2,1)

    plt.plot(epochs, history["train_loss"], label="Training Loss")

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel(
        "Cross-Entropy Loss"
    )

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.subplot(
        1,
        2,
        2
    )

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Training Accuracy"
    )

    plt.plot(epochs, history["val_accuracy"],label="Validation Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Training and Validation Accuracy")

    plt.legend()

    plt.tight_layout()

    output_path = prepare_output_path(output_path)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_confusion_matrix(
    confusion_matrix,
    class_names,
    output_path
):
    """
    Display the 35x35 confusion matrix.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray

    class_names : list
        Labels 1-9 and A-Z.

    output_path : str
        Destination image file.
    """

    plt.figure(
        figsize=(14, 12)
    )

    plt.imshow(
        confusion_matrix,
        interpolation="nearest"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.colorbar()

    positions = np.arange(
        len(class_names)
    )

    plt.xticks(
        positions,
        class_names,
        rotation=90
    )

    plt.yticks(
        positions,
        class_names
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.tight_layout()

    output_path = prepare_output_path(
        output_path
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


def plot_misclassified_examples(
    X,
    y_true,
    y_pred,
    label_to_character,
    output_path,
    number_of_examples=5
):
    """
    Display incorrectly classified test examples.

    Parameters
    ----------
    X : numpy.ndarray
        Test image data.

    y_true : numpy.ndarray
        True labels.

    y_pred : numpy.ndarray
        Predicted labels.

    label_to_character : callable
        Converts numerical label into readable character.

    output_path : str
        Destination image file.

    number_of_examples : int
        Number of mistakes to display.

    Returns
    -------
    numpy.ndarray
        Indices of displayed incorrect examples.
    """

    incorrect_indices = np.where(
        y_true != y_pred
    )[0]

    if len(incorrect_indices) == 0:
        print(
            "No incorrect predictions found."
        )

        return np.array([])

    selected_indices = (
        incorrect_indices[
            :number_of_examples
        ]
    )

    plt.figure(
        figsize=(15, 4)
    )

    for position, index in enumerate(
        selected_indices
    ):

        plt.subplot(
            1,
            len(selected_indices),
            position + 1
        )

        image = X[index].reshape(
            28,
            28
        )

        plt.imshow(
            image,
            cmap="gray"
        )

        true_character = (
            label_to_character(
                int(y_true[index])
            )
        )

        predicted_character = (
            label_to_character(
                int(y_pred[index])
            )
        )

        plt.title(
            f"True: {true_character}\n"
            f"Pred: {predicted_character}"
        )

        plt.axis("off")

    plt.tight_layout()

    output_path = prepare_output_path(
        output_path
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    return selected_indices