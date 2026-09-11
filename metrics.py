from pathlib import Path

import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Calculate classification accuracy.

    Parameters
    ----------
    y_true : numpy.ndarray
        True labels.

    y_pred : numpy.ndarray
        Predicted labels.

    Returns
    -------
    float
        Proportion of correct predictions.
    """

    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, num_classes=35):
    """
    Build a confusion matrix manually using NumPy.

    Rows represent true classes.
    Columns represent predicted classes.

    Parameters
    ----------
    y_true : numpy.ndarray
    y_pred : numpy.ndarray

    num_classes : int
        Number of classes.

    Returns
    -------
    numpy.ndarray
        Confusion matrix.
    """

    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, predicted_label in zip(
        y_true,
        y_pred
    ):
        matrix[
            true_label,
            predicted_label
        ] += 1

    return matrix


def classification_metrics(
    y_true,
    y_pred,
    num_classes=35
):
    """
    Calculate accuracy, precision, recall and F1 score.

    Metrics are calculated separately for every class
    and then macro-averaged.

    Parameters
    ----------
    y_true : numpy.ndarray
    y_pred : numpy.ndarray
    num_classes : int

    Returns
    -------
    dict
        Classification metrics.
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        num_classes
    )

    precision = np.zeros(
        num_classes
    )

    recall = np.zeros(
        num_classes
    )

    f1 = np.zeros(
        num_classes
    )

    for class_id in range(num_classes):

        true_positive = (cm[class_id, class_id])

        false_positive = (np.sum(cm[:, class_id]) - true_positive)

        false_negative = (np.sum(cm[class_id, :]) - true_positive)

        precision_denominator = (true_positive + false_positive)

        recall_denominator = (
            true_positive
            + false_negative
        )

        if precision_denominator > 0:
            precision[class_id] = (
                true_positive
                / precision_denominator
            )

        if recall_denominator > 0:
            recall[class_id] = (
                true_positive
                / recall_denominator
            )

        denominator = (
            precision[class_id]
            + recall[class_id]
        )

        if denominator > 0:
            f1[class_id] = (
                2
                * precision[class_id]
                * recall[class_id]
                / denominator
            )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    return {
        "accuracy": accuracy,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "macro_precision": np.mean(precision),
        "macro_recall": np.mean(recall),
        "macro_f1": np.mean(f1),
        "confusion_matrix": cm
    }


def save_metrics_report(
    metrics,
    class_names,
    output_path
):
    """
    Save classification metrics into a text file.

    Parameters
    ----------
    metrics : dict
        Metrics returned by classification_metrics().

    class_names : list
        Human-readable class names.

    output_path : str or pathlib.Path
        Destination text file.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as file:

        file.write("CHARACTER RECOGNITION RESULTS\n")

        file.write("=============================\n\n")

        file.write(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}\n")

        file.write(
            f"Macro Precision: "
            f"{metrics['macro_precision']:.4f}\n"
        )

        file.write(
            f"Macro Recall: "
            f"{metrics['macro_recall']:.4f}\n"
        )

        file.write(
            f"Macro F1: "
            f"{metrics['macro_f1']:.4f}\n\n"
        )

        file.write(
            "PER-CLASS METRICS\n"
        )

        file.write(
            "-----------------\n"
        )

        for i, name in enumerate(
            class_names
        ):

            file.write(
                f"{name}: "
                f"Precision="
                f"{metrics['precision_per_class'][i]:.4f}, "
                f"Recall="
                f"{metrics['recall_per_class'][i]:.4f}, "
                f"F1="
                f"{metrics['f1_per_class'][i]:.4f}\n"
            )