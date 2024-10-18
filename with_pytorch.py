import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Function to calculate L2 regularization loss
def l2_regularization(model):
    l2_loss = torch.tensor(0.0, requires_grad=True)  # Initialize L2 loss
    for param in model.parameters():
        l2_loss = l2_loss + torch.norm(param, 2)  # Add the L2 norm of the parameters
    return l2_loss

class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size,name="Pytorch"):
        super(TwoLayerNN, self).__init__()
        self.name = name
        # Define the first linear layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Define the second linear layer
        self.fc2 = nn.Linear(hidden_size, 1)
        # ReLU activation is used between the layers
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    def __str__(self):
        
        return str("Name:"+self.name+"\n"+super(TwoLayerNN, self).__str__())


def create_model_from_u_alpha(u_list,alpha_list):
    model = TwoLayerNN(len(u_list[0]), len(u_list))
    # Convert u_list and alpha_list to tensors
    u_list = np.array(u_list)
    alpha_list = np.array(alpha_list)
    u_tensor = torch.tensor(u_list, dtype=torch.float32)
    alpha_tensor = torch.tensor(alpha_list, dtype=torch.float32)
    
    # Manually set the weights
    with torch.no_grad():
        model.fc1.weight.copy_(u_tensor)        # Shape matches (m, d)
        model.fc1.bias.zero_()                  
        model.fc2.weight.copy_(alpha_tensor)    # Shape matches (1, m)
        model.fc2.bias.zero_()                  

    return model

'''

# Define the model
input_size = 10  # Number of input features
hidden_size = 5  # Number of hidden neurons
output_size = 1  # Number of outputs

model = TwoLayerNN(input_size, hidden_size)

# Define a loss function and optimizer
criterion = nn.MSELoss()  # Example: mean squared error loss
optimizer = optim.SGD(model.parameters(), lr=0.01,weight_decay=beta)

input_data = torch.randn(1, input_size)  # Batch size of 1, input size 10
output = model(input_data)

# Print the output
print(output)
'''