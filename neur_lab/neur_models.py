import torch
import torch.nn as torch_nn
import torch.optim as optim

def train_xor_model(activation_name, init_type="random"):
    # Task 1 & 3: Problem Setup
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y_bin = torch.tensor([[0.], [1.], [1.], [0.]])
    
    # Select hidden activation
    activations = {
        "Sigmoid": torch_nn.Sigmoid(),
        "Tanh": torch_nn.Tanh(),
        "ReLU": torch_nn.ReLU()
    }
    
    # Task 2 & 3: Model Architecture (2-2-1)
    # Using BCEWithLogitsLoss for numerical stability (combines Sigmoid + BCE)
    model = torch_nn.Sequential(
        torch_nn.Linear(2, 2),
        activations[activation_name],
        torch_nn.Linear(2, 1)
    )
    
    # Part C: Symmetry Experiment (Zero init)
    if init_type == "zero":
        with torch.no_grad():
            model[0].weight.fill_(0.0)
            model[0].bias.fill_(0.0)
            model[2].weight.fill_(0.0)
            model[2].bias.fill_(0.0)
            
    criterion = torch_nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    
    early_grad_norm = 0.0
    
    for step in range(2000):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y_bin)
        loss.backward()
        
        # Part B & D: Gradient capture at early step
        if step == 0:
            early_grad_norm = torch.norm(model[0].weight.grad).item()
            
        optimizer.step()
        
    # Task 4 Part A: Basic learning check predictions
    with torch.no_grad():
        final_logits = model(X)
        final_probs = torch.sigmoid(final_logits)
        predictions = (final_probs > 0.5).float()
        correct = (predictions == y_bin).sum().item() == 4
        
    return loss.item(), correct, early_grad_norm, final_probs, model[0].weight.clone()

def train_three_class_model():
    # Task 5: Extend the task
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    # Class 0: (0,0), Class 1: (0,1) or (1,0), Class 2: (1,1)
    y_multi = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    
    # 2 inputs -> 2 hidden -> 3 outputs
    model = torch_nn.Sequential(
        torch_nn.Linear(2, 4), # Increased slightly to ensure capacity for 3 classes
        torch_nn.Tanh(),
        torch_nn.Linear(4, 3)
    )
    
    criterion = torch_nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    
    for step in range(2000):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y_multi)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        final_logits = model(X)
        probs = torch.softmax(final_logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        
        print("\n--- Task 5: Three-Class Decision ---")
        print(f"Final Weights Shape (Output Layer): {model[2].weight.shape}")
        print("Class Probabilities:")
        for i in range(4):
            print(f"Input: {X[i].tolist()} -> Probs: {[round(p, 4) for p in probs[i].tolist()]}")
            
        # Numerical verification that probabilities sum to 1
        sum_probs = torch.sum(probs[0]).item()
        print(f"Sum of probabilities for first example: {sum_probs}")

if __name__ == "__main__":
    torch.manual_seed(42) # Engineering best practice for reproducibility
    
    print("--- Task 4 Part D: Activation Experiment ---")
    print(f"{'Activation':<10} | {'Loss':<8} | {'Correct 4/4':<12} | {'Early Grad Norm':<15}")
    for act in ["Sigmoid", "Tanh", "ReLU"]:
        # Reset seed to give each activation the identical random starting weights
        torch.manual_seed(42) 
        loss, correct, early_grad, _, _ = train_xor_model(act)
        print(f"{act:<10} | {loss:<8.4f} | {str(correct):<12} | {early_grad:<15.4f}")

    print("\n--- Task 4 Part C: Symmetry Experiment (Zero Init) ---")
    loss, correct, _, _, final_weights = train_xor_model("Tanh", init_type="zero")
    print(f"Zero Init Final Loss: {loss:.4f}, Correct: {correct}")
    print("Hidden Layer Weights (Note identical rows):")
    print(final_weights.detach().numpy())
    
    # Run the multiclass extension
    torch.manual_seed(42)
    train_three_class_model()
