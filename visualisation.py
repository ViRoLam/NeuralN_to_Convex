import torch
import torch.nn as nn
import with_pytorch as pt
import matplotlib.pyplot as plt


def visualize_performance(X_train, y_train, X_test, y_test, beta, P_values, generate_convex_model, generate_pytorch_model):
    convex_losses = []
    pytorch_losses = []

    for P in P_values:
        model_conv, u_list, alpha_list = generate_convex_model(X_train, y_train, beta, P)
        model_pytorch = generate_pytorch_model(X_train, y_train, beta, len(u_list))

        # Test models on test data
        X_test_torch = torch.tensor(X_train, dtype=torch.float32)
        y_test_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        criterion = nn.MSELoss(reduction="sum")

        model_conv.eval()
        with torch.no_grad():
            predictions = model_conv(X_test_torch)
            convex_loss = 0.5 * criterion(predictions, y_test_torch) + pt.l2_regularization(model_conv)
            convex_losses.append(convex_loss.item())
            MSE_loss = criterion(predictions, y_test_torch).item()
            Reg_Loss = pt.l2_regularization(model_conv).item()

        model_pytorch.eval()
        with torch.no_grad():
            predictions = model_pytorch(X_test_torch)
            pytorch_loss = 0.5 * criterion(predictions, y_test_torch) + pt.l2_regularization(model_pytorch)
            pytorch_losses.append(pytorch_loss.item())

    plt.figure(figsize=(10, 5))
    plt.plot(P_values, convex_losses, label='Convex Solver Loss')
    plt.plot(P_values, pytorch_losses, label='PyTorch Model Loss')
    plt.xlabel('P')
    plt.ylabel('Loss')
    plt.title('Performance Comparison with Varying P')
    plt.legend()
    plt.show()