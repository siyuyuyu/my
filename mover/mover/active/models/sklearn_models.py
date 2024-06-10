import datetime
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


# Define the binh_korn function
def binh_korn(x, y):
    if len(x) != len(y):
        raise ValueError("x and y must have the same size.")

    f1 = 4 * x**2 + 4 * y**2
    f2 = (x - 5) ** 2 + (y - 5) ** 2
    result = np.column_stack((f1, f2))
    return result


def train_sklearn_models(model_name, x_train, y_train, hyperparameters, save=False):
    """
    :param save:
    :param plot:
    :param model_name:
    :param x_train:
    :param y_train:
    :param hyperparameters:
    :param name:
    :return:
    """
    # Create the selected model with hyperparameters
    if model_name == "GPR":
        model = GaussianProcessRegressor(**hyperparameters)
    elif model_name == "RF":
        model = RandomForestRegressor(**hyperparameters)
    elif model_name == "SVR":
        model = SVR(**hyperparameters)
    else:
        raise ValueError("Invalid model name. Choose either 'GPR', 'RF', or 'SVR'.")

    # Train the model
    model.fit(x_train, y_train)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if save:
        # Save the model using pickle
        with open(f"trained_{model_name}_{timestamp}.pkl", "wb") as file:
            pickle.dump(model, file)

    return model


def build_sklearn_models(model_name, x_train, y_train, hyperparameters):
    # Create the selected model with hyperparameters
    if model_name == "GPR":
        model = GaussianProcessRegressor(**hyperparameters)
        # Train the model
        model.fit(x_train, y_train)
        return model
    elif model_name == "RF":
        model = RandomForestRegressor(**hyperparameters)
        # Train the model
        model.fit(x_train, y_train)
        return model
    elif model_name == "SVR":
        model = SVR(**hyperparameters)
        # Train the model
        model.fit(x_train, y_train)
        return model
    else:
        raise ValueError("Invalid model name. Choose either 'GPR', 'RF', or 'SVR'.")


def predict_with_sklearn_models(model, x_test, y_test=None, name=None):
    """

    :param plot:
    :param y_test:
    :param model:
    :param x_test:
    :return:
    """
    # Predict f1 and f2 values for the test set
    y_pred = model.predict(x_test)
    f1_pred, f2_pred = y_pred[:, 0], y_pred[:, 1]

    # Calculate evaluation metrics
    if y_test is not None:
        mse_f1 = mean_squared_error(y_test[:, 0], f1_pred)
        mse_f2 = mean_squared_error(y_test[:, 1], f2_pred)
        r2_f1 = r2_score(y_test[:, 0], f1_pred)
        r2_f2 = r2_score(y_test[:, 1], f2_pred)
        if name is not None:
            # Plot the training metrics
            # Plot the training metrics
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

            # Plot for f1
            axes[0].scatter(y_test[:, 0], f1_pred, color="b", label="f1")
            axes[0].plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], "r--")
            axes[0].set_xlabel("True f1", fontsize=14)  # Increased font size
            axes[0].set_ylabel("Predicted f1", fontsize=14)  # Increased font size
            axes[0].legend()
            axes[0].set_title(f"f1: MSE={mse_f1:.2f}, R2={r2_f1:.2f}")
            # Including evaluation metrics directly in the title or as a text box inside the plot for f1
            axes[0].text(
                0.05,
                0.95,
                f"MSE: {mse_f1:.2f}\nR2: {r2_f1:.2f}",
                transform=axes[0].transAxes,
                fontsize=16,
                verticalalignment="top",
            )

            # Plot for f2
            axes[1].scatter(y_test[:, 1], f2_pred, color="g", label="f2")
            axes[1].plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], "r--")
            axes[1].set_xlabel("True f2", fontsize=14)  # Increased font size
            axes[1].set_ylabel("Predicted f2", fontsize=14)  # Increased font size
            axes[1].legend()
            axes[1].set_title(f"f2: MSE={mse_f2:.2f}, R2={r2_f2:.2f}")
            # Including evaluation metrics directly in the title or as a text box inside the plot for f2
            axes[1].text(
                0.05,
                0.95,
                f"MSE: {mse_f2:.2f}\nR2: {r2_f2:.2f}",
                transform=axes[1].transAxes,
                fontsize=16,
                verticalalignment="top",
            )

            plt.tight_layout()  # Adjust layout to not overlap
            plt.show()
    return model.predict(x_test, return_std=True)
