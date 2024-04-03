import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def nlog_prob_loss(net, X, y):
    """Computes a loss function suitable for multiway classification."""
    predictions = net.forward(X)
    probs = torch.gather(predictions, dim=1, index=y.unsqueeze(1))
    probs = probs.clamp(min=0.0000000001, max=0.9999999999)
    losses = -torch.log(probs)
    loss = torch.mean(losses)
    return loss


def minibatch_gd(model, num_epochs, train_set, test_set, lr=0.1):
    """Simple implementation of minibatch gradient descent."""
    accs = []
    for _ in tqdm(range(num_epochs)):  
        model.train()  
        train_loader = DataLoader(train_set, batch_size=32)
        for X, y in train_loader:
            loss = nlog_prob_loss(model, X, y)
            loss.backward()
            for param in model.parameters():
                with torch.no_grad():           
                    param -= lr*param.grad
                    param.grad = None
        test_loader = DataLoader(test_set, batch_size=128)
        model.eval()
        accuracy = evaluate(model, test_loader)
        accs.append(accuracy)
    return accs


def evaluate(classifier, test_loader):
    """Evaluates a multiway classifier on test data."""
    correct = 0
    total = 0
    for X, y in test_loader:
        predictions = classifier(X)
        preds = torch.max(predictions, 1)
        correct += torch.sum(preds.indices == y).item()
        total += torch.numel(y)
    return correct/total
