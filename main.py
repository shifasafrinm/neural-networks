from pathlib import Path

import numpy as np

from preprocessing import (
    load_dataset,
    inspect_dataset,
    clean_dataset,
    verify_cleaning,
    separate_features_labels,
    inspect_features,
    normalize_features,
    remap_labels,
    label_to_character,
    balance_dataset,
    split_dataset
)

from neural_network import NeuralNetwork

from metrics import (
    classification_metrics,
    save_metrics_report
)

from visualization import (
    plot_sample_images,
    plot_training_history,
    plot_confusion_matrix,
    plot_misclassified_examples
)
BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    BASE_DIR / "outputs"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


SAMPLES_PER_CLASS = 500

HIDDEN_NEURONS = 128

LEARNING_RATE = 0.05

BATCH_SIZE = 64

EPOCHS = 20

data = load_dataset()

inspect_dataset(data)

data = clean_dataset(data)

verify_cleaning(data)

X, y = separate_features_labels(data)

inspect_features(X, y)

X = normalize_features(X)

y = remap_labels(y)

X, y = balance_dataset(X, y, samples_per_class=SAMPLES_PER_CLASS)

print("\nBalanced X shape:", X.shape)

print("Balanced y shape:", y.shape)

plot_sample_images(
    X,
    y,
    label_to_character,
    output_path=(
        OUTPUT_DIR
        / "sample_images.png"
    )
)

(X_train,y_train, X_val,y_val,X_test,y_test) = split_dataset(X, y)


print("\nTraining shape:", X_train.shape, y_train.shape)

print("Validation shape:", X_val.shape, y_val.shape)

print("Testing shape:", X_test.shape, y_test.shape)

model = NeuralNetwork(input_size=784, hidden_size=HIDDEN_NEURONS, output_size=35, activation="relu",seed=42)


print("\nNeural Network Architecture:")

print(f"784 -> "
    f"{HIDDEN_NEURONS} -> 35")

history = model.train(X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE,learning_rate=LEARNING_RATE)

plot_training_history(history,OUTPUT_DIR/ "training_history.png")

test_probabilities = (
    model.predict_proba(
        X_test
    )
)

test_loss = (
    model.cross_entropy_loss(
        test_probabilities,
        y_test
    )
)

test_predictions = np.argmax(test_probabilities,axis=1)

metrics = classification_metrics(y_test, test_predictions, num_classes=35)


print(
    "\n========== TEST RESULTS =========="
)

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{metrics['accuracy']:.4f}"
)

print(
    f"Macro Precision: "
    f"{metrics['macro_precision']:.4f}"
)

print(
    f"Macro Recall: "
    f"{metrics['macro_recall']:.4f}"
)

print(
    f"Macro F1 Score: "
    f"{metrics['macro_f1']:.4f}"
)


class_names = [
    label_to_character(i)
    for i in range(35)
]


save_metrics_report(
    metrics,
    class_names,
    OUTPUT_DIR
    / "metrics.txt"
)

plot_confusion_matrix(
    metrics[
        "confusion_matrix"
    ],
    class_names,
    OUTPUT_DIR
    / "confusion_matrix.png"
)

incorrect_indices = (
    plot_misclassified_examples(
        X_test,
        y_test,
        test_predictions,
        label_to_character,
        OUTPUT_DIR
        / "misclassified_examples.png",
        number_of_examples=5
    )
)

print(
    "\n========== MISCLASSIFIED EXAMPLES =========="
)

for number, index in enumerate(
    incorrect_indices,
    start=1
):

    true_label = (
        label_to_character(
            int(y_test[index])
        )
    )

    predicted_label = (
        label_to_character(
            int(
                test_predictions[index]
            )
        )
    )

    print(
        f"Example {number}: "
        f"True={true_label}, "
        f"Predicted={predicted_label}"
    )