import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage


#K_P = 0.1
#K_I = 0.5
#K_D = 5

K_P = 0.8
K_I = 0.2   
K_D = 0.1

STEPS = 550
car = make_car(desired_v=20.0, dt=0.1)


#WRITE CODE HERE
velocities = []
errors = []
times = []

#ML EXTENSION: add a list to store the desired accelerations and desired velocities
desired_accelerations = []
desired_velocities = []

# function to make graphs for the ML extension
def plot_predictions(test_labels, predictions):
    """
    Plots actual acceleration vs predicted acceleration.
    """
    plt.figure(figsize=(10, 7))

    plt.scatter(
        test_labels,
        predictions,
        s=10,
        label="Predictions"
    )

    plt.xlabel("Actual Desired Acceleration")
    plt.ylabel("Predicted Desired Acceleration")
    plt.title("Predicted vs Actual Acceleration")

    plt.legend()
    plt.show()



for step in range(STEPS):
   desired_acceleration, error = calculate_desired_acceleration(car, K_P, K_I, K_D)
   throttle_percentage = acceleration_to_throttle_percentage(desired_acceleration)
   velocities.append(car["v"])
   errors.append(error)
   times.append(car["t"])

   desired_accelerations.append(desired_acceleration)
   desired_velocities.append(car["desired_v"])

   update(car, throttle_percentage)


print(f"Final velocity: {car['v']} m/s")

# turning lists into training data
X = np.column_stack((velocities, desired_velocities))
y = np.array(desired_accelerations)

# Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)

#convert numpy arrays to torch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().reshape(-1, 1) # reshape to be a column vector

X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().reshape(-1, 1) # reshape to be a column vector


# Subclass nn.Module to make our model
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        # Use nn.Linear() for creating the model parameters
        super().__init__()

        self.linear_layer = nn.Linear(
            in_features=2,
            out_features=1
        )
    # Define the forward computation (input data x flows through nn.Linear())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
torch.manual_seed(42)

model_1 = LinearRegressionModelV2()

print(model_1)
print(model_1.state_dict())
# Check model device
print(next(model_1.parameters()).device)
# Create loss function
loss_fn = nn.L1Loss()

# Create optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), # optimize newly created model's parameters
                            lr=0.01)

# Put model and data on the same device
device = "cpu"

model_1 = model_1.to(device)

X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)

X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# data for loss curves graph
train_loss_values = []
test_loss_values = []
epoch_count = []

# Set the number of epochs
epochs = 1000

for epoch in range(epochs):

    # --------------------
    # Training
    # --------------------
    model_1.train()

    # 1. Forward pass
    y_pred = model_1(X_train_tensor)

    # 2. Calculate training loss
    loss = loss_fn(y_pred, y_train_tensor)

    # 3. Zero the gradients
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Update the model's weights and bias
    optimizer.step()

    # --------------------
    # Testing
    # --------------------
    model_1.eval()

    with torch.inference_mode():

        # 1. Make predictions on test data
        test_pred = model_1(X_test_tensor)

        # 2. Calculate test loss
        test_loss = loss_fn(test_pred, y_test_tensor)

    train_loss_values.append(loss.item())
    test_loss_values.append(test_loss.item())
    epoch_count.append(epoch)

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | "
            f"Train loss: {loss.item():.4f} | "
            f"Test loss: {test_loss.item():.4f}"
        )
# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# See what the model learned
from pprint import pprint

print("\nThe model learned the following values:")
pprint(model_1.state_dict())

# Turn model into evaluation mode
model_1.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_preds = model_1(X_test_tensor)

print(y_preds)

plot_predictions(
    test_labels=y_test_tensor.cpu(),
    predictions=y_preds.cpu()
)


#time vs velocity plot
plt.figure()
plt.plot(times, velocities)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Car Velocity vs Time")
plt.show()


#error vs time plot
plt.figure()
plt.plot(times, errors)
plt.xlabel("Time (s)")
plt.ylabel("Error (m/s)")
plt.title("Car Error vs Time")
plt.show()