from torch import nn  # Define your MLP model
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.layers(x)


# Assuming the class MLP and other required imports and variables are defined as above

def boostrap_model(model,optimizer, criterion, train_loader, n, num_epochs):
    all_predictions = []

    for _ in range(n):
        # Reinitialize model parameters and optimizer for each training session
        model.apply(reset_weights)

        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # Evaluate and store predictions
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            predictions = []
            for inputs, _ in train_loader:
                outputs = model(inputs)
                predictions.append(outputs)
            predictions = torch.cat(predictions, dim=0)
            all_predictions.append(predictions)
        model.train()  # Set model back to train mode for the next iteration

    all_predictions = torch.stack(all_predictions)  # Shape: [n, num_samples, output_size]

    # Calculate mean and std dev across all trainings
    mean_predictions = torch.mean(all_predictions, dim=0)
    std_predictions = torch.std(all_predictions, dim=0)

    return mean_predictions, std_predictions

# Function to reinitialize model weights
def reset_weights(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


def plot_prediction(y_test, y_pred, name=None):
    """

    :param plot:
    :param y_test:
    :param model:
    :param x_test:
    :return:
    """
    # Predict f1 and f2 values for the test set

    f1_pred, f2_pred = y_pred[:, 0], y_pred[:, 1]

    mse_f1 = mean_squared_error(y_test[:, 0], f1_pred)
    mse_f2 = mean_squared_error(y_test[:, 1], f2_pred)
    r2_f1 = r2_score(y_test[:, 0], f1_pred)
    r2_f2 = r2_score(y_test[:, 1], f2_pred)
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

