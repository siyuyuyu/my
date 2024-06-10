import numpy as np
from scipy import integrate
from scipy.stats import multivariate_normal, norm
from torch.utils.data import DataLoader, Dataset
import numpy as np
import plotly.graph_objs as go


class NumpyDataset(Dataset):
    """Custom Dataset for loading numpy arrays"""

    def __init__(self, data, labels=None):
        """
        Args:
            data (numpy.ndarray): A numpy array containing the data.
            labels (numpy.ndarray, optional): A numpy array containing the labels.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.labels is not None:
            return self.data[index], self.labels[index]
        return self.data[index]


def numpy_dataloader(data, labels=None, batch_size=32, shuffle=False):
    """
    Creates a DataLoader for a given numpy array.

    Args:
        data (numpy.ndarray): A numpy array containing the data.
        labels (numpy.ndarray, optional): A numpy array containing the labels.
        batch_size (int): The size of each batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: A DataLoader instance that yields batches of data.
    """
    dataset = NumpyDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_pareto_front(scores):
    population_size = scores.shape[0]
    pareto_front = np.ones(population_size, dtype=bool)
    for i in range(population_size):
        for j in range(population_size):
            if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
                pareto_front[i] = False
                break
    return scores[pareto_front]


  # , np.array(index)


# def get_improvement_region(mean_prediction, pareto_front):
#     # Convert mean_prediction and pareto_front to NumPy arrays if they are not already
#     mean_prediction_np = np.array(mean_prediction)
#     pareto_front_np = np.array(pareto_front)
#
#     # Expand the dimensions of mean_prediction to enable broadcasting when comparing with pareto_front
#     expanded_mean_pred = np.expand_dims(mean_prediction_np, axis=1)
#
#     # Use broadcasting to compare each mean prediction point with all Pareto front points
#     is_improvement = np.all(np.any(expanded_mean_pred < pareto_front_np, axis=2), axis=1)
#
#     # Filter mean_prediction based on the improvement condition
#     improvement_region = mean_prediction_np[is_improvement]
#
#     return improvement_region

def get_improvement_region(mean_prediction, pareto_front):
     # Convert mean_prediction and pareto_front to NumPy arrays if they are not already
    mean_prediction_np = np.array(mean_prediction)
    pareto_front_np = np.array(pareto_front)

     # Expand the dimensions of mean_prediction to enable broadcasting when comparing with pareto_front
    expanded_mean_pred = np.expand_dims(mean_prediction_np, axis=1)

     # Use broadcasting to compare each mean prediction point with all Pareto front points
    is_improvement = np.all(np.any(expanded_mean_pred < pareto_front_np, axis=2), axis=1)

     # Filter mean_prediction based on the improvement condition
    improvement_region = mean_prediction_np[is_improvement]
#
    return improvement_region



def calculate_maximin_distance(mean_pred, pareto_front):
    d_maximin = np.finfo(np.float32).eps
    # Loop over each point in the Pareto front
    for pf_point in pareto_front:
        # Calculate the minimum distance between the mean prediction and the Pareto front point
        distance = np.minimum(pf_point[0] - mean_pred[0], pf_point[1] - mean_pred[1])
        # If this distance is greater than 0 and greater than the current maximin distance, update it
        if distance > 0 and distance > d_maximin:
            d_maximin = distance
    return d_maximin


def calculate_closest_point(improvement_region, pareto_front):
    closest_point = min(pareto_front, key=lambda p: np.linalg.norm(p - improvement_region))

    return closest_point


def calculate_centroid_shaded(improvement_region, prob_of_improvement):
    y1_weighted = np.sum(improvement_region[0] * prob_of_improvement[:, np.newaxis]) / np.sum(prob_of_improvement)
    y2_weighted = np.sum(improvement_region[1] * prob_of_improvement[:, np.newaxis]) / np.sum(prob_of_improvement)

    return y1_weighted, y2_weighted


def calculate_EI_centroid(improvement_region, pareto_front, prob_of_improvement):
    # First, find the closest point on the subPareto front
    # This assumes Euclidean distance.

    closest_point = calculate_closest_point(improvement_region, pareto_front)

    # Calculate the centroid of the shaded region

    y1_weighted, y2_weighted = calculate_centroid_shaded(improvement_region, prob_of_improvement)
    # Calculate the distance L
    # L = np.sqrt((y1_weighted - closest_point[0]) ** 2 + (y2_weighted - closest_point[1]) ** 2)

    # Calculate the distance L
    L = np.linalg.norm([y1_weighted - closest_point[0], y2_weighted - closest_point[1]])

    # Calculate E[I(x)]

    EI_centroid = prob_of_improvement * L

    return EI_centroid


# Strategy Implementations
def random_selection(X_search):
    return X_search[np.random.randint(len(X_search))]


def pure_exploitation(X_search, model):
    means = [model.predict([x])[0] for x in X_search]
    return X_search[np.argmin(means, axis=0)]


def pure_exploration(X_search, model):
    variances = [model.predict([x], return_std=True)[1] for x in X_search]
    return X_search[np.argmax(variances)]


def get_joint_prob_dist(improvement_region):
    # Convert improvement_region to a NumPy array for efficient operations
    improvement_region_np = np.array(improvement_region)

    # Calculate φ(y1) and φ(y2) using vectorized operations
    phi_y1 = norm.pdf(improvement_region_np[:, 0])
    phi_y2 = norm.pdf(improvement_region_np[:, 1])

    # Calculate the product of φ(y1) and φ(y2) : the Probability of Improvement
    prob_of_improvement = phi_y1 * phi_y2

    return prob_of_improvement


def calculate_EI_maximun(improvement_region, pareto_front, prob_of_improvement):
    # Calculate EI_maximin for each candidate
    EI_maximin_values = []
    for mean_pred, pi in zip(improvement_region, prob_of_improvement):
        d_maximin = calculate_maximin_distance(mean_pred, pareto_front)
        EI_maximin = d_maximin * pi
        EI_maximin_values.append(EI_maximin)

    return np.array(EI_maximin_values)


def select_next_point_1(improvement_region, pareto_front, prob_of_improvement):
    # Find the index of the candidate with the highest EI_maximin value
    ei_maximum = calculate_EI_maximun(improvement_region, pareto_front, prob_of_improvement)
    ei_centroid = calculate_EI_centroid(improvement_region, pareto_front, prob_of_improvement)

    index_of_selected = np.argmax(ei_centroid)
    # If EI-Centroid is 0, choose candidate point with largest EI-maximin
    if ei_centroid[index_of_selected] == 0:
        index_of_selected = np.argmax(ei_maximum)

    x_selected = improvement_region[index_of_selected]
    return x_selected, index_of_selected


def select_next_point(x_search, ei_maximum, ei_centroid):
    index_of_selected = np.argmax(ei_centroid)
    # If EI-Centroid is 0, choose candidate point with largest EI-maximin
    if ei_centroid[index_of_selected] == 0:
        index_of_selected = np.argmax(ei_maximum)
    x_selected = x_search[index_of_selected]
    return x_selected, index_of_selected


#
#
# # Define the joint probability distribution function
# # def joint_prob_dist(y1, y2):
# #     """
# #
# #     :param y1:
# #     :param y2:
# #     :return:
# #     """
# #     return norm.pdf(y1, loc=mu1, scale=sigma1) * norm.pdf(y2, loc=mu2, scale=sigma2)
#
#
# # Define the subPareto front
# def subpareto_front(y1, y2):
#     """
#
#     :param y1:
#     :param y2:
#     :return:
#     """
#     is_dominated = np.zeros_like(y1, dtype=bool)
#     pareto_indices = []
#     non_pareto_indices = []
#
#     for i, _ in enumerate(y1):
#         is_pareto = True
#         for j, _ in enumerate(y2):
#             if i != j and np.all(y1[i] >= y1[j]) and np.all(y2[i] >= y2[j]):
#                 is_dominated[i] = True
#                 is_pareto = False
#                 break
#         if is_pareto:
#             pareto_indices.append(i)
#         else:
#             non_pareto_indices.append(i)
#
#     sub_front_y1 = y1[~is_dominated]
#     sub_front_y2 = y2[~is_dominated]
#
#     pareto_indices = np.array(pareto_indices)
#     non_pareto_indices = np.array(non_pareto_indices)
#
#     return sub_front_y1, sub_front_y2, pareto_indices, non_pareto_indices
#
#
# # Define the Probability of Improvement
# def P_I(Y):
#     """
#
#     :param Y:
#     :return:
#     """
#     # Calculate the mean and variance of the joint probability distribution
#     mu = np.array([mu1, mu2])
#     cov = np.array([[sigma1**2, 0], [0, sigma2**2]])
#     # Calculate the distance between the subPareto front and the candidate point
#     dist = np.linalg.norm(Y - subpareto_front(Y)[-1], axis=1)
#     # Calculate the probability of improvement
#     P = norm.cdf((subpareto_front(Y)[-1] - mu) / np.sqrt(cov))
#     P = np.prod(P, axis=1)
#     P = np.mean(P * (dist == np.min(dist)))
#     return P
#
#
# # Define the Centroid approach to Expected Improvement
# def ei_centroid_1(Y):
#     """
#
#     :param Y:
#     :return:
#     """
#     mu = np.array([mu1, mu2])
#     cov = np.array([[sigma1**2, 0], [0, sigma2**2]])
#     phi = norm.pdf(Y, mean=mu, cov=cov)
#     P_I_Y = P_I(Y)
#     Y1 = np.sum(Y[:, 0] * phi) / P_I_Y
#     Y2 = np.sum(Y[:, 1] * phi) / P_I_Y
#     sub_PF = subpareto_front(Y)
#     dist = np.linalg.norm(np.array([Y1, Y2]) - sub_PF[-1])
#     return P_I_Y * dist
#
#
# # Define the Maximin approach to Expected Improvement
# def ei_maximin_1(y1, y2):
#     """
#
#     :param Y:
#     :return:
#     """
#     mu = np.array([mu1, mu2])
#     cov = np.array([[sigma1**2, 0], [0, sigma2**2]])
#     dist = np.linalg.norm(Y - subpareto_front(y1, y2)[-1], axis=1)
#     index = np.argmin(dist)
#     closest_point = subpareto_front(y1, y2)[index]
#     d_maximin = np.max([np.min([closest_point[0] - mu[0], closest_point[1] - mu[1], 0]), 0])
#     return P_I(y1, y2) * d_maximin
#
#
# # Define function for calculating EI-Centroid
# def ei_centroid_2(y1, y2):
#     """
#
#     :param Y:
#     :return:
#     """
#     # Calculate the mean and variance of the joint probability distribution
#     mu = np.array([mu1, mu2])
#     cov = np.array([[sigma1**2, 0], [0, sigma2**2]])
#     # Calculate the centroid of the probability distribution
#     integral_y1 = lambda y1: integrate.quad(lambda y2: multivariate_normal.pdf([y1, y2], mu, cov), -np.inf, np.inf)[0]
#     integral_y2 = lambda y2: integrate.quad(lambda y1: multivariate_normal.pdf([y1, y2], mu, cov), -np.inf, np.inf)[0]
#     Y1 = integrate.quad(integral_y1, -np.inf, np.inf)[0] / P_I
#     Y2 = integrate.quad(integral_y2, -np.inf, np.inf)[0] / P_I
#     # Find the closest point on the subPareto front to the candidate point
#     dist = np.linalg.norm(Y - subpareto_front(y1, y2)[-1], axis=1)
#     index = np.argmin(dist)
#     closest_point = subpareto_front(y1, y2)[index]
#     # Calculate the distance between the centroid and closest point on subPareto front
#     L = np.sqrt((y1 - closest_point[0]) ** 2 + (y2 - closest_point[1]) ** 2)
#     # Calculate EI-Centroid
#     return P_I * L
#
#
# # Define function for calculating EI-maximin
# def ei_maximin_2(y1, y2):
#     """
#
#     :param Y:
#     :return:
#     """
#     # Calculate the mean and variance of the joint probability distribution
#     mu = np.array([mu1, mu2])
#     cov = np.array([[sigma1**2, 0], [0, sigma2**2]])
#     # Find the minimum distance between means of candidate point and subPareto points
#     dists = []
#     for pi in subpareto_front(y1, y2):
#         dists.append(np.min([pi[0] - mu[0], pi[1] - mu[1], 0]))
#     d_maximin = np.max(dists)
#     # Calculate EI-maximin
#     return P_I * d_maximin
#
#
# # Define function for selecting the next data point
# def select_next_point(x, y):
#     """
#     To use these functions, you would need to provide the values for mu1, mu2, sigma1, sigma2,
#       P_I, and the subPareto front function subpareto_front(Y)
#       specific to your problem. You would also need to provide the candidate points X and
#       their corresponding objective values Y. Finally, you could call select_next_point(X, Y)
#     to get the candidate point with the largest EI-C
#     """
#     # Calculate EI-Centroid for all candidate points
#     ei_centroids = [ei_centroid_1(y.reshape(1, -1) + val.reshape(1, -1)) for val in x]
#     # Calculate EI-maximin for all candidate points
#     ei_maximins = [ei_maximin_1(y.reshape(1, -1) + val.reshape(1, -1)) for val in x]
#     # Choose candidate point with largest EI-Centroid
#     index = np.argmax(ei_centroids)
#     # If EI-Centroid is 0, choose candidate point with largest EI-maximin
#     if ei_centroids[index] == 0:
#         index = np.argmax(ei_maximins)
#     # Return candidate point
#     return x[index]


def constraint_violation2(x):
    # Initialize the constraint violation sum
    cv = 0

    # Material Composition Constraints
    # x3 + x4 + x5 >= 30%
    cv += max(0, 30 - (x[2] + x[3] + x[4]))

    # x4 (Cu) <= 20%
    cv += max(0, x[3] - 20)

    # x3 (Fe) <= 5%
    cv += max(0, x[2] - 5)

    # x5 (Pd) <= 20%
    cv += max(0, x[4] - 20)

    # Non-negativity Constraints
    # All components should be non-negative
    for xi in x:
        cv += max(0, -xi)

    # Equality constraint for sum to 100%
    # The sum of all components should equal 100%
    cv += abs(np.sum(x) - 100)

    # Specific Material Constraints for Ni (x[0]) and Ti (x[1])
    # Add constraints if there are any upper or lower bounds for Ni and Ti
    # Example: cv += max(0, lower_bound - x[0]) + max(0, x[0] - upper_bound) for Ni
    # Example: cv += max(0, lower_bound - x[1]) + max(0, x[1] - upper_bound) for Ti

    return cv


def constraint_violation(x):
    # Constraints
    g = []
    h = []

    # Material Composition Constraints
    g.append(x[2] + x[3] + x[4] - 30)  # x3 + x4 + x5 >= 30%
    g.append(20 - x[3])  # x4 <= 20%
    g.append(5 - x[2])  # x3 <= 5%
    g.append(20 - x[4])  # x5 <= 20%

    # Non-negativity Constraints
    for xi in x:
        g.append(-xi)  # xi >= 0

    # Equality constraint for sum to 100%
    h.append(np.sum(x) - 100)

    # cv = sum(max(0, gi) for gi in g) + sum(abs(hj) for hj in h)

    # Assuming g and h are lists or arrays of NumPy arrays
    cv = sum(np.maximum(0, gi).sum() for gi in g) + sum(np.abs(hj).sum() for hj in h)

    # Specific Material Constraints for Ni (x[0]) and Ti (x[1])
    # Add constraints if there are any upper or lower bounds for Ni and Ti
    # Example: cv += max(0, lower_bound - x[0]) + max(0, x[0] - upper_bound) for Ni
    # Example: cv += max(0, lower_bound - x[1]) + max(0, x[1] - upper_bound) for Ti

    return cv


# Function to animate the scatter plot with Plotly
def animate_moo(mean_predictions, pareto_fronts, x_selected_points, xlabel, ylabel):
    # Create figure
    fig = go.Figure()

    frames = []

    for i in range(len(mean_predictions)):
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=mean_predictions[i][:, 0],
                    y=mean_predictions[i][:, 1],
                    mode="markers",
                    marker=dict(color="blue"),
                    name="Mean Prediction",
                ),
                go.Scatter(
                    x=pareto_fronts[i][:, 0],
                    y=pareto_fronts[i][:, 1],
                    mode="markers",
                    marker=dict(color="red"),
                    name="Pareto Front",
                ),
                go.Scatter(
                    x=[x_selected_points[i][0]],
                    y=[x_selected_points[i][1]],
                    mode="markers",
                    marker=dict(color="orange", size=12, symbol="star"),
                    name="Selected Point",
                ),
            ],
            name=f"frame{i}",
        )
        frames.append(frame)

    fig.frames = frames

    fig.update_layout(
        title="Pareto Front with Region of Improvement",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        updatemenus=[
            {
                "buttons": [
                    {"args": [None, {"frame": {"duration": 500, "redraw": True}}], "label": "Play", "method": "animate"}
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
    )

    # Set the first frame to be displayed
    fig.update(frames=[frames[0]])

    # Show the animation
    fig.show()

# Run the animation function with the placeholder data
# animate_kmc(mean_predictions, pareto_fronts, x_selected_points, n_iterations)


#
#
# def surrogate_model(x, L):
#     # Placeholder for the surrogate model prediction at highest level of fidelity L
#     pass
#
#
# def corr_function(x, l, L):
#     # Placeholder for the posterior correlation function between the level of fidelity l and L
#     pass
#
#
# def cost_ratio(l, L):
#     # Placeholder for the ratio of computational cost between the high-fidelity level L
#     # and the l-th level of fidelity
#     pass
#
#
# def calculate_UMFEI(x, l, L, sigma_epsilon):
#     # UEI(x, L) is the expected improvement at the highest level of fidelity L
#     # and should be calculated according to the user's method.
#     UEI = surrogate_model(x, L)  # Placeholder for UEI
#
#     # Calculate utility functions alpha1, alpha2, and alpha3 as defined in the provided pseudocode
#     alpha1 = corr_function(x, l, L)
#     alpha2 = 1 - sigma_epsilon / np.sqrt(surrogate_model.variance(x, L) + sigma_epsilon ** 2)
#     alpha3 = cost_ratio(l, L)
#
#     # Calculate the Multifidelity Expected Improvement (MFEI)
#     UMFEI = UEI * alpha1 * alpha2 * alpha3
#
#     return UMFEI
#
#
# from scipy.stats import norm
#
#
# # Placeholder functions for the components of the MFPI calculation. They need actual implementations.
# def PI(x, L):
#     """Placeholder for the probability of improvement function at fidelity level L."""
#     pass
#
#
# def eta1(x, l, L):
#     """Placeholder for the correlation coefficient between fidelity levels l and L."""
#     pass
#
#
# def lambda_ratio(l, L):
#     """Placeholder for the computational cost ratio between fidelity levels l and L."""
#     pass
#
#
# def R(x, xi):
#     """Placeholder for the correlation function."""
#     pass
#
#
# def calculate_MFPI(x, l, L, X_search_l):
#     """
#     Calculate the Multifidelity Probability of Improvement for a candidate point x.
#
#     Parameters:
#     - x: The candidate point.
#     - l: The current level of fidelity being evaluated.
#     - L: The highest level of fidelity available.
#     - X_search_l: The points at level l.
#
#     Returns:
#     - The MFPI value for the candidate point x.
#     """
#     eta_1 = eta1(x, l, L)
#     eta_2 = lambda_ratio(l, L)
#     eta_3 = np.prod([1 - R(x, xi_l) for xi_l in X_search_l])  # Assuming X_search_l is a list of points at level l
#
#     # Calculate the Multifidelity Probability of Improvement (MFPI)
#     U_MFPI = PI(x, L) * eta_1 * eta_2 * eta_3
#
#     return U_MFPI
#
# def calculate_eta3(x, X_search_l, correlation_function):
#     """
#     Calculate the sample density function (η3) for a candidate point x.
#
#     Parameters:
#     - x: The candidate point.
#     - X_search_l: The points at level l.
#     - correlation_function: The function to calculate the spatial correlation.
#
#     Returns:
#     - The η3 value for the candidate point x.
#     """
#     eta_3 = np.prod([1 - correlation_function(x, xi_l) for xi_l in X_search_l])
#     return eta_3
#
# # Placeholder function for the spatial correlation function R(·)
# def correlation_function(x, xi):
#     """Placeholder for the spatial correlation function."""
#     # An actual implementation should return a value between 0 and 1, where 1 indicates perfect correlation.
#     pass
#
#
#
#
#
#
#
#
