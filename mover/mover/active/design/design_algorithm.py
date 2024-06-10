import numpy as np
from scipy.stats import norm

from mover.active.models.sklearn_models import build_sklearn_models, predict_with_sklearn_models


# Multi-objective design algorithm
def active_design(Xtrain, Ytrain, Xsearch, strategy, tradeoff, model_name, hyperparameters):
    """

    param Xtrain: An array that holds the descriptor or feature vectors of all the known materials.
    param Ytrain:  An array that contains the property value vectors corresponding to the known materials.
    param Xsearch: An array comprising the descriptor or feature vectors of the unmeasured materials.
    param strategy: The design strategy include Maximin, Centroid, Random, Pure Exploitation, and Pure Exploration.
    param tradeoff: Exploitation, Exploration tradeoff
    param model_name: name of the surregate model (GPR, RF, SVR)
    param hyperparameters: dictionary with  hyperparameters
    return:
    """
    # Build surrogate model

    surrogate_model = build_sklearn_models(model_name, Xtrain, Ytrain, hyperparameters)

    # Initialize variables
    n_search = len(Xsearch)
    improvement = np.zeros(n_search)
    e_improvement = np.zeros(n_search)
    p_improvement = np.zeros(n_search)

    # Iterate over unmeasured materials
    for i in range(n_search):
        # Bootstrap the predictions
        g = bootstrap(surrogate_model, Xsearch[i])

        # Calculate mean and uncertainty
        mu = np.mean(g)
        sigma = np.sqrt(np.mean((g - mu) ** 2))

        # Calculate probability of improvement
        z = (np.min(Ytrain, axis=0) - mu) / sigma

        p_improvement[i] = norm.cdf(z)

        # Calculate improvement
        if strategy == "Maximin":
            improvement[i] = np.max(np.abs(mu - Ytrain), axis=0)
        elif strategy == "Centroid":
            improvement[i] = np.sum(np.abs(mu - Ytrain), axis=0) / len(Ytrain)

        # Calculate expected improvement
        e_improvement[i] = p_improvement[i] * improvement[i]

        # Select material with highest expected improvement
    x_selected = Xsearch[np.argmax(e_improvement)]

    return x_selected


# Bootstrap function
def bootstrap(model, x):
    """
    Bootstrap the predictions
    Here, we assume a simple Gaussian noise model
    :param model:
    :param x:
    :return:
    """

    n_samples = len(x)
    noise = np.random.normal(loc=0.0, scale=0.01, size=(n_samples, 2))
    f = predict_with_sklearn_models(model, x)
    return f + noise


import numpy as np


def maximin_design(X, model, epsilon=0.01):
    """
    Maximin-based design strategy.

    Parameters:
    X (pd.DataFrame): Feature matrix of materials in the search space.
    model (sklearn estimator): Trained regression model.
    epsilon (float): Small positive number to avoid division by zero.

    Returns:
    int: Index of the selected material.
    """
    mean, std = model.predict(X, return_std=True)
    improvement_probability = std / (std + epsilon)
    distance = np.abs(mean - np.mean(mean)) / (np.std(mean) + epsilon)
    maximin_score = improvement_probability * distance
    selected_material_index = np.argmax(maximin_score)
    return selected_material_index


def centroid_design(X, model):
    """
    Centroid-based design strategy.

    Parameters:
    X (pd.DataFrame): Feature matrix of materials in the search space.
    model (sklearn estimator): Trained regression model.

    Returns:
    int: Index of the selected material.
    """
    mean, _ = model.predict(X, return_std=True)
    centroid = np.mean(X, axis=0)
    distances = np.linalg.norm(X - centroid, axis=1)
    selected_material_index = np.argmin(distances)
    return selected_material_index


def random_selection(X):
    """
    Random selection strategy.

    Parameters:
    X (pd.DataFrame): Feature matrix of materials in the search space.

    Returns:
    int: Index of the selected material.
    """
    selected_material_index = np.random.randint(0, len(X))
    return selected_material_index


def pure_exploitation(X, model):
    """
    Pure exploitation strategy.

    Parameters:
    X (pd.DataFrame): Feature matrix of materials in the search space.
    model (sklearn estimator): Trained regression model.

    Returns:
    int: Index of the selected material.
    """
    mean, _ = model.predict(X, return_std=True)
    selected_material_index = np.argmin(mean)
    return selected_material_index


def pure_exploration(X, model):
    """
    Pure exploration strategy.

    Parameters:
    X (pd.DataFrame): Feature matrix of materials in the search space.
    model (sklearn estimator): Trained regression model.

    Returns:
    int: Index of the selected material.
    """
    _, std = model.predict(X, return_std=True)
    selected_material_index = np.argmax(std)
    return selected_material_index
