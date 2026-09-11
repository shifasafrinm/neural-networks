import numpy as np


class NeuralNetwork:
    """
    Fully-connected neural network implemented using NumPy.

    Architecture
    ------------
    Input layer:
        784 neurons

    Hidden layer:
        configurable, default 128 neurons

    Output layer:
        35 neurons

    Supported hidden activations:
        ReLU
        Tanh
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=35, activation="relu", seed=42):
        """
        Initialize neural-network parameters.

        Parameters
        ----------
        input_size : int
            Number of input features.

        hidden_size : int
            Number of hidden neurons.

        output_size : int
            Number of classes.

        activation : str
            Hidden activation function.
            Either 'relu' or 'tanh'.

        seed : int
            Random seed for reproducibility.
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.activation_name = (
            activation.lower()
        )

        rng = np.random.default_rng(seed)

        # He initialization for first layer
        self.W1 = (
            rng.standard_normal(
                (input_size, hidden_size)
            )
            * np.sqrt(2.0 / input_size)
        )

        self.b1 = np.zeros(
            (1, hidden_size)
        )

        self.W2 = (
            rng.standard_normal(
                (hidden_size, output_size)
            )
            * np.sqrt(2.0 / hidden_size)
        )

        self.b2 = np.zeros(
            (1, output_size)
        )

        self.cache = {}


    def relu(self, Z):
        """
        Apply ReLU activation.

        ReLU(z) = max(0, z)

        Parameters
        ----------
        Z : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """

        return np.maximum(0, Z)


    def relu_derivative(self, Z):
        """
        Compute derivative of ReLU.

        Parameters
        ----------
        Z : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            1 where Z > 0 and 0 otherwise.
        """

        return (Z > 0).astype(float)


    def tanh(self, Z):
        """
        Apply hyperbolic tangent activation.

        Parameters
        ----------
        Z : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """

        return np.tanh(Z)


    def tanh_derivative(self, Z):
        """
        Compute derivative of tanh.

        Parameters
        ----------
        Z : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """

        A = np.tanh(Z)

        return 1 - A ** 2


    def activation(self, Z):
        """
        Apply selected hidden-layer activation.
        """

        if self.activation_name == "relu":
            return self.relu(Z)

        if self.activation_name == "tanh":
            return self.tanh(Z)

        raise ValueError(
            "Activation must be 'relu' or 'tanh'."
        )


    def activation_derivative(self, Z):
        """
        Compute derivative of selected activation.
        """

        if self.activation_name == "relu":
            return self.relu_derivative(Z)

        if self.activation_name == "tanh":
            return self.tanh_derivative(Z)

        raise ValueError(
            "Activation must be 'relu' or 'tanh'."
        )


    def softmax(self, Z):
        """
        Convert raw output scores into probabilities.

        Softmax probabilities for every sample sum to 1.

        A numerical stability adjustment subtracts
        the maximum score before exponentiation.

        Parameters
        ----------
        Z : numpy.ndarray
            Raw output scores.

        Returns
        -------
        numpy.ndarray
            Class probabilities.
        """

        shifted_Z = (
            Z
            - np.max(
                Z,
                axis=1,
                keepdims=True
            )
        )

        exponentials = np.exp(
            shifted_Z
        )

        probabilities = (
            exponentials
            / np.sum(
                exponentials,
                axis=1,
                keepdims=True
            )
        )

        return probabilities


    def one_hot_encode(self, y):
        """
        Convert integer labels into one-hot vectors.

        Example
        -------
        Label 2 with 5 classes becomes:

        [0, 0, 1, 0, 0]

        Parameters
        ----------
        y : numpy.ndarray
            Integer labels.

        Returns
        -------
        numpy.ndarray
            One-hot encoded matrix.
        """

        one_hot = np.zeros((len(y), self.output_size))

        one_hot[np.arange(len(y)), y] = 1

        return one_hot


    def forward(self, X):
        """
        Perform forward propagation.

        Steps
        -----
        Z1 = XW1 + b1
        A1 = activation(Z1)

        Z2 = A1W2 + b2
        A2 = softmax(Z2)

        Parameters
        ----------
        X : numpy.ndarray
            Input matrix of shape
            (samples, input_size).

        Returns
        -------
        numpy.ndarray
            Output probabilities with shape
            (samples, 35).
        """

        Z1 = (X @ self.W1 + self.b1)

        A1 = self.activation(Z1)

        Z2 = (A1 @ self.W2 + self.b2)

        A2 = self.softmax(Z2)

        self.cache = {
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2
        }

        return A2


    def cross_entropy_loss(
        self,
        probabilities,
        y
    ):
        """
        Compute multiclass cross-entropy loss.

        Parameters
        ----------
        probabilities : numpy.ndarray
            Predicted Softmax probabilities.

        y : numpy.ndarray
            True integer labels.

        Returns
        -------
        float
            Mean cross-entropy loss.
        """

        epsilon = 1e-12

        probabilities = np.clip(
            probabilities,
            epsilon,
            1.0
        )

        correct_probabilities = (
            probabilities[
                np.arange(len(y)),
                y
            ]
        )

        loss = -np.mean(
            np.log(
                correct_probabilities
            )
        )

        return loss


    def backward(self, X, y):
        """
        Perform backpropagation and calculate gradients.

        Parameters
        ----------
        X : numpy.ndarray
            Input batch.

        y : numpy.ndarray
            True class labels.

        Returns
        -------
        dict
            Gradients for W1, b1, W2 and b2.
        """

        number_of_samples = X.shape[0]

        Z1 = self.cache["Z1"]
        A1 = self.cache["A1"]
        A2 = self.cache["A2"]

        Y = self.one_hot_encode(y)

        # Softmax + cross entropy gradient
        dZ2 = (
            A2 - Y
        ) / number_of_samples

        dW2 = (
            A1.T @ dZ2
        )

        db2 = np.sum(
            dZ2,
            axis=0,
            keepdims=True
        )

        dA1 = (
            dZ2 @ self.W2.T
        )

        dZ1 = (
            dA1
            * self.activation_derivative(Z1)
        )

        dW1 = (
            X.T @ dZ1
        )

        db1 = np.sum(
            dZ1,
            axis=0,
            keepdims=True
        )

        gradients = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }

        return gradients


    def update_parameters(
        self,
        gradients,
        learning_rate
    ):
        """
        Update weights and biases using gradient descent.

        Rule
        ----
        parameter =
        parameter - learning_rate * gradient

        Parameters
        ----------
        gradients : dict
            Gradients calculated by backpropagation.

        learning_rate : float
            Gradient-descent step size.
        """

        self.W1 -= (
            learning_rate
            * gradients["dW1"]
        )

        self.b1 -= (
            learning_rate
            * gradients["db1"]
        )

        self.W2 -= (
            learning_rate
            * gradients["dW2"]
        )

        self.b2 -= (
            learning_rate
            * gradients["db2"]
        )


    def predict_proba(self, X):
        """
        Return class probabilities for input samples.
        """

        return self.forward(X)


    def predict(self, X):
        """
        Predict the most likely class for every sample.

        Returns
        -------
        numpy.ndarray
            Integer class predictions.
        """

        probabilities = (
            self.predict_proba(X)
        )

        return np.argmax(
            probabilities,
            axis=1
        )


    def accuracy(self, y_true, y_pred):
        """
        Calculate classification accuracy.
        """

        return np.mean(
            y_true == y_pred
        )


    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=20,
        batch_size=64,
        learning_rate=0.05,
        seed=42
    ):
        """
        Train the neural network using mini-batch
        gradient descent.

        Training order for each batch
        -----------------------------
        1. Forward propagation
        2. Cross-entropy calculation
        3. Backpropagation
        4. Gradient-descent update

        After every epoch, a separate validation
        forward pass is performed without updating
        model parameters.

        Parameters
        ----------
        X_train : numpy.ndarray
        y_train : numpy.ndarray
        X_val : numpy.ndarray
        y_val : numpy.ndarray

        epochs : int
            Number of complete passes through
            the training dataset.

        batch_size : int
            Number of examples processed before
            one parameter update.

        learning_rate : float
            Gradient-descent learning rate.

        seed : int
            Random seed.

        Returns
        -------
        dict
            Training and validation history.
        """

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": []
        }

        rng = np.random.default_rng(seed)

        number_of_samples = len(X_train)

        for epoch in range(epochs):

            permutation = rng.permutation(
                number_of_samples
            )

            X_train_shuffled = (
                X_train[permutation]
            )

            y_train_shuffled = (
                y_train[permutation]
            )

            for start in range(
                0,
                number_of_samples,
                batch_size
            ):

                end = start + batch_size

                X_batch = (
                    X_train_shuffled[
                        start:end
                    ]
                )

                y_batch = (
                    y_train_shuffled[
                        start:end
                    ]
                )

                # Forward propagation
                batch_probabilities = (
                    self.forward(X_batch)
                )

                # Loss is calculated here because
                # it is part of the required
                # training sequence.
                self.cross_entropy_loss(
                    batch_probabilities,
                    y_batch
                )

                # Backpropagation
                gradients = self.backward(
                    X_batch,
                    y_batch
                )

                # Gradient descent
                self.update_parameters(
                    gradients,
                    learning_rate
                )

            # ---------- TRAINING METRICS ----------

            train_probabilities = (
                self.forward(X_train)
            )

            train_loss = (
                self.cross_entropy_loss(
                    train_probabilities,
                    y_train
                )
            )

            train_predictions = np.argmax(
                train_probabilities,
                axis=1
            )

            train_accuracy = self.accuracy(
                y_train,
                train_predictions
            )

            # ---------- VALIDATION LOOP ----------

            val_probabilities = (
                self.forward(X_val)
            )

            val_loss = (
                self.cross_entropy_loss(
                    val_probabilities,
                    y_val
                )
            )

            val_predictions = np.argmax(
                val_probabilities,
                axis=1
            )

            val_accuracy = self.accuracy(
                y_val,
                val_predictions
            )

            history["train_loss"].append(
                train_loss
            )

            history["val_loss"].append(
                val_loss
            )

            history[
                "train_accuracy"
            ].append(train_accuracy)

            history[
                "val_accuracy"
            ].append(val_accuracy)

            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

        return history