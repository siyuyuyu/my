import copy
import os
import pickle
from pathlib import Path
import pandas as pd
from pymatgen.core.composition import Composition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import preprocessing
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from mover.active.design.tools import (
    calculate_centroid_shaded,
    calculate_EI_centroid,
    calculate_EI_maximun,
    calculate_maximin_distance,
    constraint_violation,
    get_improvement_region,
    get_joint_prob_dist,
    get_pareto_front,
    numpy_dataloader,
    pure_exploitation,
    pure_exploration,
    random_selection,
    select_next_point
)
from mover.active.models.sklearn_models import predict_with_sklearn_models, train_sklearn_models

from mover.active.models.nn_models import MLP, boostrap_model, plot_prediction

# %%
df_hea = pd.read_csv("rhea_data.csv")
df_hea = df_hea.drop(columns=['Unnamed: 0'])
# df_hea = df_hea.dropna(how='any', subset=["density", "Pugh ratio (B/G)"])
# print(df_hea.info())

# fig=px.scatter(df_hea, x='dH', y='lnPeqo', color='GBTPredict')

# fig.write_html("figures/dH_vs_lnPeqo.html")
# fig.write_image("figures/dH_vs_lnPeqo.png")
# fig.show()
# %%
# comps = Composition('Nb20Ta30Ti20Hf10Zr20')
#
# compos = df_hea['composition'].tolist()
# fractional_composition = [Composition(comp).fractional_composition.as_dict() for comp in compos]
#
# # Define the elements to create columns for
# elements = ['Hf', 'Ta', 'Nb', 'Zr', 'Ti']
#
# # Add columns for each element and fill with the corresponding percentage, put 0 if the element is not present
# for element in elements:
#     df_hea[element] = [composition.get(element, 0) for composition in fractional_composition]
#
# print(df_hea.head())

# fig=px.scatter(df_hea, x='SSOS avg formation energy', y='pughs ratio', color='SSOS stdev lattice constant')

# fig.write_html("figures/dS_vs_lnPeqo.html")
# fig.write_image("figures/dS_vs_lnPeqo.png")
# fig.show()


# %%
descriptors = ['Al', 'Ti', 'V', 'Cr', 'Zr', 'Nb', 'Mo', 'Pd', 'Hf', 'Ta']
target = ['dH', 'lnPeqo']

X = df_hea[descriptors].to_numpy()
Y = df_hea[target].to_numpy()

Y_scaler = preprocessing.StandardScaler().fit(Y)
Y_scaled = Y_scaler.transform(Y)

# %%
model_name = "GPR"
kernel = RBF(length_scale=1.0)
hyperparameters = {"kernel": kernel, "alpha": 1e-3, "n_restarts_optimizer": 5, "random_state": 65}

# workdir = Path(__file__).parent.absolute()

# Get the current working directory as a Path object
current_directory = Path.cwd()

# If you need the parent directory
parent_directory = current_directory.parent

# To get the absolute path of the current directory
workdir = current_directory.absolute()

name = os.path.join(workdir, model_name)

X_train, X_search, Y_train, Y_search = train_test_split(X, Y_scaled, test_size=3 / 4, random_state=564)

"""
## Pseudocode for DESIGN procedure

**Inputs:**
1. `X_train` ← list containing descriptor or feature vectors of all known materials
2. `Y_train` ← list containing property values vectors of all known materials
3. `X_search` ← list containing descriptor or feature vectors of all unmeasured materials

**Procedure: DESIGN(`X_train`, `Y_train`, `X_search`)**
```python
Build a surrogate model, f(X_train) = Y_train
for all descriptor vectors x_i^search in X_search do
    Bootstrap the predictions f(x_i^search) -> g(x_i^search)
    Mean value of the predicted distribution, mu_i^search <- E[g(x_i^search)]
    Uncertainty in the predicted distribution, sigma_i^search <- sqrt(E[(g(x_i^search) - mu_i^search)^2])
    Calculate the Probability of Improvement, P[I_i^search] = P[g(x_i^search) in Region of Improvement]
    Calculate Improvement, I_i^search = Maximin(|mu_i^search - PF|) or I_i^search = Centroid(|mu_i^search - PF|)
    Expected Improvement, E[I_i^search] = I_i^search x P[I_i^search]
end for
x_selected = x_i^search in X_search such that E[I_i^search] > E[I_j^search] for all j in X_search, i != j
return x_selected

"""

# 1.  Build a surrogate model
model = train_sklearn_models(model_name, X_train, Y_train, hyperparameters, save=False)

# %%
# 2. Bootstrap the predictions +
mean_prediction, std_prediction = predict_with_sklearn_models(model, X_train, Y_train, name="trainset_GRP")

_, _ = predict_with_sklearn_models(model, X_search, Y_search, name="test_set_GRP")

mean_prediction_rescaled = Y_scaler.inverse_transform(mean_prediction)
# Y_search_rescaled = Y_scaler.inverse_transform(Y_search)

# %%

data = Y_scaler.inverse_transform(Y_train)

pareto_front = get_pareto_front(data)

plot = 1
if plot == 1:
    # Plot
    plt.scatter(data[:, 0], data[:, 1], c="blue")
    # first column, second column
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c="red")
    plt.title("Pareto Front")
    plt.xlabel("dH")
    plt.ylabel("lnPeqo")
    plt.show()

# 3. Calculate the Probability of Improvement
improvement_region = get_improvement_region(mean_prediction_rescaled, pareto_front)
# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(mean_prediction_rescaled[:, 0], mean_prediction_rescaled[:, 1], c="blue", label="Mean Prediction")
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c="red", label="Pareto Front")
if improvement_region.size > 0:
    plt.scatter(improvement_region[:, 0], improvement_region[:, 1], c="green", label="Region of Improvement")
plt.title("Pareto Front with Region of Improvement")
plt.xlabel(target[0])
plt.ylabel(target[1])
plt.legend()
plt.show()

# 4. Calculate Improvement
prob_of_improvement = get_joint_prob_dist(improvement_region)

# 5. Expected Improvement
# Find the index of the candidate with the highest EI  value


x_selected, index_of_selected = select_next_point(X_search, pareto_front, prob_of_improvement)

print(x_selected)

y_selected = improvement_region[index_of_selected]

# running Calculation or experiment
y_calculated = Y_search[index_of_selected]

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(mean_prediction_rescaled[:, 0], mean_prediction_rescaled[:, 1], c="blue", label="Mean Prediction")
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c="red", label="Pareto Front")
if improvement_region.size > 0:
    plt.scatter(improvement_region[:, 0], improvement_region[:, 1], c="green", label="Region of Improvement")
plt.scatter(y_selected[0], y_selected[1], c="orange", marker="*", s=200, label="Selected Point")
plt.title("Pareto Front with Region of Improvement")
plt.ylabel(target[1])
plt.xlabel(target[0])
plt.legend()
plt.show()

# %%


# Convert your dataset to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# Split your dataset
X_train, X_search, Y_train, Y_search = train_test_split(X_tensor, Y_tensor, test_size=3 / 4, random_state=564)

# Create DataLoader for training and testing
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_search, Y_search)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

model = MLP(input_size=X_train.shape[1], output_size=Y_train.shape[1])
criterion = nn.MSELoss()  # For regression; use nn.CrossEntropyLoss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
# 2. Bootstrap the predictions +


# Example usage:
model = MLP(input_size=X_train.shape[1], output_size=Y_train.shape[1])
# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Evaluation for regression task. Modify accordingly for classification
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    # Example for calculating MSE for evaluation
    total_loss = 0
    predicted = []
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        predicted.append(outputs)
    avg_loss = total_loss / len(test_loader)
    print(f'Average Loss: {avg_loss}')

real_values = np.array(Y_train)
predicted_values = np.array(torch.cat(predicted, dim=0))

plot_prediction(real_values, predicted_values)

mean_predictions, std_predictions = boostrap_model(model, optimizer, criterion, train_loader, n=5, num_epochs=500)
mean_prediction_rescaled = Y_scaler.inverse_transform(mean_prediction)

data = Y_scaler.inverse_transform(Y_train)

PF = get_pareto_front(data)
I = get_improvement_region(mean_prediction_rescaled, PF)
PI = get_joint_prob_dist(I)
EI_centroid = calculate_EI_centroid(I, PF, PI)
EI_maximun = calculate_EI_maximun(I, PF, PI)
# CV = [constraint_violation(x) for x in X_search]

x_selected, index_of_selected = select_next_point(X_search, EI_maximun, EI_centroid)

y_selected = I[index_of_selected]

# running Calculation or experiment
y_calculated = Y_search[index_of_selected]

print(x_selected)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(mean_prediction_rescaled[:, 0], mean_prediction_rescaled[:, 1], c="blue", label="Mean Prediction")
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c="red", label="Pareto Front")
if improvement_region.size > 0:
    plt.scatter(improvement_region[:, 0], improvement_region[:, 1], c="green", label="Region of Improvement")
plt.scatter(y_selected[0], y_selected[1], c="orange", marker="*", s=200, label="Selected Point")
plt.title("Pareto Front with Region of Improvement")
plt.ylabel(target[1])
plt.xlabel(target[0])
plt.legend()
plt.show()
