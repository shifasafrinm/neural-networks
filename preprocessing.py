from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset():
    """
    Load the character recognition dataset from the dataset folder.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing 784 pixel columns and one class column.
    """

    base_dir = Path(__file__).resolve().parent
    dataset_path = (base_dir/ "dataset"/ "data.csv")

    data = pd.read_csv(dataset_path)

    return data


def inspect_dataset(data):
    """
    Display basic information about the raw dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw character-recognition dataset.
    """

    print("\n========== RAW DATASET ==========")

    print("\nDataset shape:")
    print(data.shape)

    print("\nNumber of columns:")
    print(len(data.columns))

    print("\nAvailable classes:")
    print(np.sort(data["class"].unique()))

    print("\nNumber of classes:")
    print(data["class"].nunique())

    print("\nSamples per class:")
    print(
        data["class"]
        .value_counts()
        .sort_index()
    )

    print("\nMissing values:")
    print(data.isnull().sum().sum())

    print("\nDuplicate rows:")
    print(data.duplicated().sum())


def clean_dataset(data):
    """
    Clean the raw dataset.

    Cleaning operations:
    1. Remove rows containing missing values.
    2. Remove digit class 0 because the assignment requires digits 1-9.
    3. Ensure class labels are integers.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset containing classes 1-35.
    """

    cleaned_data = data.dropna().copy()

    cleaned_data = cleaned_data[
        cleaned_data["class"] != 0
    ].copy()

    cleaned_data["class"] = (
        cleaned_data["class"].astype(int)
    )

    return cleaned_data


def verify_cleaning(data):
    """
    Verify the result of dataset cleaning.

    Parameters
    ----------
    data : pandas.DataFrame
        Cleaned dataset.
    """

    classes = np.sort(
        data["class"].unique()
    )

    print("\n========== CLEANED DATA ==========")
    print("\nClasses:")
    print(classes)

    print("\nNumber of classes:")
    print(len(classes))

    print("\nDataset shape:")
    print(data.shape)

    if len(classes) == 35:
        print("\nCorrect: 35 classes remain.")
    else:
        print("\nWARNING: Expected 35 classes.")


def separate_features_labels(data):
    """
    Separate image pixels from target class labels.

    Parameters
    ----------
    data : pandas.DataFrame
        Cleaned dataset.

    Returns
    -------
    X : numpy.ndarray
        Feature matrix with shape
        (number_of_samples, 784).

    y : numpy.ndarray
        Label vector with shape
        (number_of_samples,).
    """

    X = data.drop("class",axis=1).values

    y = data["class"].values

    return X, y


def inspect_features(X, y):
    """
    Display information about the extracted features and labels.

    Parameters
    ----------
    X : numpy.ndarray
        Image pixel data.

    y : numpy.ndarray
        Class labels.
    """

    print("\n========== FEATURE INFORMATION ==========")

    print("\nX shape:")
    print(X.shape)

    print("\ny shape:")
    print(y.shape)

    print("\nPixels per image:")
    print(X.shape[1])

    print("\nMinimum pixel value:")
    print(X.min())

    print("\nMaximum pixel value:")
    print(X.max())


def normalize_features(X):
    """
    Normalize grayscale pixels from 0-255 into the range 0-1.

    Parameters
    ----------
    X : numpy.ndarray": Raw pixel values.
    Returns
    -------
    numpy.ndarray:  Normalized pixel values stored as float32.
    """

    X = X.astype(np.float32)
    X = X / 255.0
    return X


def remap_labels(y):
    """
    Convert original labels 1-35 to internal labels 0-34.
    Original:
        1-9  -> digits 1-9
        10-35 -> letters A-Z

    Internal:
        0-8  -> digits 1-9
        9-34 -> letters A-Z

    Parameters
    ----------
    y : numpy.ndarray
        Original labels.

    Returns
    -------
    numpy.ndarray
        Labels ranging from 0 to 34.
    """

    return y - 1


def label_to_character(label):
    """
    Convert an internal label 0-34 into a readable character.

    Parameters
    ----------
    label : int
        Internal neural-network label.

    Returns
    -------
    str
        Character represented by the label.
    """

    original_label = label + 1

    if original_label <= 9:
        return str(original_label)

    letter_index = original_label - 10

    return chr(ord("A") + letter_index)


def balance_dataset(X, y, samples_per_class=500, seed=42):
    """
    Create a balanced dataset containing an equal number
    of examples for every class.

    Parameters
    ----------
    X : numpy.ndarray
        Input features.

    y : numpy.ndarray
        Internal labels 0-34.

    samples_per_class : int
        Number of samples to select from each class.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_balanced : numpy.ndarray
        Balanced feature matrix.

    y_balanced : numpy.ndarray
        Balanced target labels.
    """

    rng = np.random.default_rng(seed)

    X_balanced = []
    y_balanced = []

    for class_id in range(35):

        indices = np.where(
            y == class_id
        )[0]

        if len(indices) < samples_per_class:
            raise ValueError(
                f"Class {class_id} only contains "
                f"{len(indices)} examples."
            )

        selected_indices = rng.choice(
            indices,
            size=samples_per_class,
            replace=False
        )

        X_balanced.append(
            X[selected_indices]
        )

        y_balanced.append(
            y[selected_indices]
        )

    X_balanced = np.concatenate(
        X_balanced,
        axis=0
    )

    y_balanced = np.concatenate(
        y_balanced,
        axis=0
    )

    return X_balanced, y_balanced


def shuffle_dataset(X, y, seed=42):
    """
    Shuffle features and labels while preserving their pairing.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix.

    y : numpy.ndarray
        Target labels.

    seed : int
        Random seed.

    Returns
    -------
    X_shuffled : numpy.ndarray
    y_shuffled : numpy.ndarray
    """

    rng = np.random.default_rng(seed)

    indices = rng.permutation(
        len(X)
    )

    return X[indices], y[indices]


def split_dataset(
    X,
    y,
    train_ratio=0.70,
    validation_ratio=0.15,
    seed=42
):
    """
    Perform a stratified train-validation-test split.

    Every class is divided separately to ensure that
    all 35 classes occur in all three splits.

    Parameters
    ----------
    X : numpy.ndarray
        Balanced feature matrix.

    y : numpy.ndarray
        Balanced labels.

    train_ratio : float
        Fraction used for training.

    validation_ratio : float
        Fraction used for validation.

    seed : int
        Random seed.

    Returns
    -------
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
    """

    rng = np.random.default_rng(seed)

    X_train = []
    y_train = []

    X_val = []
    y_val = []

    X_test = []
    y_test = []

    for class_id in range(35):

        indices = np.where(
            y == class_id
        )[0]

        indices = rng.permutation(
            indices
        )

        total = len(indices)

        train_end = int(
            total * train_ratio
        )

        validation_end = int(
            total
            * (
                train_ratio
                + validation_ratio
            )
        )

        train_indices = (
            indices[:train_end]
        )

        validation_indices = (
            indices[
                train_end:
                validation_end
            ]
        )

        test_indices = (indices[validation_end:])

        X_train.append(X[train_indices])
        y_train.append(y[train_indices])

        X_val.append(X[validation_indices])
        y_val.append(y[validation_indices])

        X_test.append(X[test_indices])
        y_test.append(y[test_indices])

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)

    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    X_train, y_train = shuffle_dataset(X_train, y_train, seed)

    X_val, y_val = shuffle_dataset(X_val,y_val,seed + 1)

    X_test, y_test = shuffle_dataset(X_test, y_test, seed + 2)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


def verify_splits(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Verify final dataset dimensions and distributions.

    Parameters
    ----------
    X_train, X_val, X_test : numpy.ndarray
        Feature matrices.

    y_train, y_val, y_test : numpy.ndarray
        Corresponding labels.
    """

    print("\n========== FINAL DATASET ==========")

    print("\nTraining:")
    print(
        X_train.shape,
        y_train.shape
    )

    print("\nValidation:")
    print(
        X_val.shape,
        y_val.shape
    )

    print("\nTesting:")
    print(
        X_test.shape,
        y_test.shape
    )

    print("\nPixel range:")
    print(
        X_train.min(),
        X_train.max()
    )

    print("\nTraining classes:")
    print(len(np.unique(y_train)))

    print("\nValidation classes:")
    print(len(np.unique(y_val)))

    print("\nTesting classes:")
    print(len(np.unique(y_test)))