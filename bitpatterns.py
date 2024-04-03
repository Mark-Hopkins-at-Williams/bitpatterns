import torch
from torch import tensor
from torch.nn import Parameter
import matplotlib.pyplot as plt
from training import minibatch_gd

def generate_data(num_instances, bitstring_length):
    """Generates binary classification data for bitstring detection.
    
    Instances are "positive" iff they contain the substring "1010".

    """
    target = "1010"
    rands = torch.randn(num_instances, bitstring_length)
    X = (rands > 0).float()
    y = []
    for row in range(X.shape[0]):
        bits = [str(element.item()) for element in X[row].long()]
        bitstring = ''.join(bits)
        y.append(target in bitstring)
    y = tensor(y).long()
    return[(X[i].float(), y[i]) for i in range(len(y))]


class NeuralNetwork(torch.nn.Module):
    """A 2-layer feedforward network."""

    def __init__(self, input_size, output_size, hidden_layer_size):
        super().__init__()
        self.theta1 = Parameter(torch.empty(hidden_layer_size, input_size))
        self.theta_output = Parameter(torch.empty(output_size, hidden_layer_size))
        for param in self.parameters():
            torch.nn.init.xavier_uniform_(param)

    def forward(self, x):
        result = torch.matmul(self.theta1, x.t())
        result = torch.relu(result)
        result = torch.matmul(self.theta_output, result)
        result = torch.softmax(result, dim=0).t()
        return result
    

class CNN(torch.nn.Module):
    """A simple convolutional neural network."""

    def __init__(self, input_size, output_size, kernel_width, num_kernels, dropout_p):
        super().__init__()
        self.W = kernel_width
        self.K = num_kernels
        self.D = input_size
        self.offset = Parameter(torch.zeros(self.K))
        self.kernel = Parameter(torch.empty(self.W, self.K))
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.theta_output = Parameter(torch.empty(self.K*(self.D-self.W+1), output_size))
        for param in set(self.parameters()) - set([self.offset]):
            torch.nn.init.xavier_uniform_(param)

    def forward(self, x):
        # original shape of x: (N, D)
        windows = [x[:,i:i+self.W] 
                   for i in range(self.D-self.W+1)]
        windows = torch.stack(windows, dim=1)  # shape: (N, D-W+1, W) 
        product = torch.matmul(windows, self.kernel) + self.offset  # shape: (N, D-W+1, K)        
        product = torch.reshape(product, (product.shape[0], -1))    # shape: (N, K(D-W+1))
        activated = torch.relu(product)  # shape: (N, K(D-W+1))
        activated = self.dropout(activated)
        result = torch.matmul(activated, self.theta_output) # shape: (N, 2)
        result = torch.softmax(result, dim=1)
        return result


if __name__ == "__main__":
    bitstring_len = 16
    num_epochs = 500
    train_set = generate_data(200, bitstring_len)
    test_set = generate_data(200, bitstring_len)
    print(f"Baseline: {sum([y for _, y in train_set])/len(train_set)}")
    model = NeuralNetwork(input_size=bitstring_len, output_size=2, 
                          hidden_layer_size=100)
    accs_ff = minibatch_gd(model, num_epochs, train_set, test_set)
    model = CNN(input_size=bitstring_len, output_size=2, 
                kernel_width=4, num_kernels=10, dropout_p=0.0)
    accs_cnn = minibatch_gd(model, num_epochs, train_set, test_set)
    model = CNN(input_size=bitstring_len, output_size=2, 
                kernel_width=4, num_kernels=10, dropout_p=0.5)
    accs_cnn_dropout = minibatch_gd(model, num_epochs, train_set, test_set)
    x_values = list(range(1, len(accs_ff)+1))
    plt.plot(x_values, accs_ff, label="Feedforward")
    plt.plot(x_values, accs_cnn, label="CNN")
    plt.plot(x_values, accs_cnn_dropout, label="CNN w. dropout")
    plt.legend()
    plt.xlabel("num epochs")
    plt.ylabel("test accuracy")
    plt.show()
