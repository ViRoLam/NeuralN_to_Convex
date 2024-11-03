import numpy as np
import generation_matrices as gm
import cvxpy as cp
import problem_solveur as ps
import with_pytorch as pt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split


#On va d'abord générer des valeurs aléatoires 
def generate_random_values(n,d):
    X = np.random.randn(n,d)
    y = np.random.randn(n)
    return X, y

def generate_pattern_values(n,d):
    X = np.random.randn(n,d)
    y = np.random.randn(n)/10 # On créer un pattern simple, à partir de la première colonne
    print(y)
    return X, y


n, d = 80, 1
beta = 1.0
P = 500


X,y = generate_pattern_values(n,d)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def generate_convex_model(X,y, beta,P):
    """
    Function: generate_convex_model
    Generate a model from the data X and y, using a convex solver.
    
    parameters:
        param X: np.array of size n*d
        param y: np.array of size n
    return: trained model
    """
    print("Génération des matrices D_i ...")
    D = gm.generate_D_matrices(X, P)

    # paramètre de régularisation
    P_tilde = len(D)
    print("Nombre de matrices générées:",P_tilde)
    print("Matrices générées")


    print("On résous le problème d'optimisation ...")
    v_opti, w_opti = ps.solve(X, y, D, beta)
    print("Problème résolu")
    
    print("Construction du réseau ... ")
    u_list, alpha_list = ps.construction_reseau(v_opti, w_opti, P_tilde)
    
    print(len(u_list))
    print(len(alpha_list))

    model_conv = pt.create_model_from_u_alpha(u_list, alpha_list)
    model_conv.name = "Convex"
    print("Réseau construit.")

    return model_conv, u_list, alpha_list

def generate_pytorch_model(X,y,beta,hidden_neurones_num):
    """
    Function: generate_convex_model
    Generate a model from the data X and y, using a convex solver.
    
    parameters:
        param X: np.array of size n*d
        param y: np.array of size n
    return: trained pytorch_model
    """
    n,d = X.shape
    model = pt.TwoLayerNN(d, hidden_neurones_num)
    model.name = "Pytorch"
    # Convert to torch tensors if not already
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape: (n, 1)

    # Create a dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)  

    # Define a loss function and optimizer
    criterion = nn.MSELoss()  
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
            # Calculate L2 regularization loss
            l2_loss = pt.l2_regularization(model)
            
            # Combine MSE loss and L2 loss
            total_loss = loss + beta * l2_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model
    

def test_models(models, X_test, y_test, name="Test"):
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    criterion = nn.MSELoss(reduction="sum")
    for model in models:
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_torch)
            test_loss = 0.5 * criterion(predictions, y_test_torch) + pt.l2_regularization(model)
            print(f'  {name} Loss for {model.name}: {test_loss.item():.4f}')
            print(f"  MSE Loss for {model.name}: (noting to interepret here)", criterion(predictions, y_test_torch).item())


if __name__ == '__main__':
    model_conv, u_list, alpha_list = generate_convex_model(X_train,y_train, beta, P)
    model_pytorch = generate_pytorch_model(X_train,y_train,beta, len(u_list))
    
    # Test model on train data
    # (If everything goes well, convex models should be better)
    print("Figures de mérite sur les données d'entrainement")
    test_models([model_conv, model_pytorch], X_train, y_train, name="Train")

    # Test model on testing data
    # (Nothing can be said here)
    print("Figures de mérite sur les données de test")
    test_models([model_conv, model_pytorch], X_test, y_test)
    
    print(model_conv)
    print(model_pytorch)
    
    #print(len(model_conv.parameters()))
    #print(len(model_pytorch.parameters()))

    # On peut maintenant comparer les deux modèles
    
    
    




# On va devoir aussi faire des visualisations 
# On s'entraine avec des solveurs normaux, et on compare 
# On peut faire vaier les P et comparer les performances avec la loss des solveurs normaux