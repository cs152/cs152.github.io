import torch
import math
import tqdm
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')

from torch import nn

class AutogradValue:
    '''
    Base class for automatic differentiation operations. Represents variable delcaration.
    Subclasses will overwrite func and grads to define new operations.

    Properties:
        parents (list): A list of the inputs to the operation, may be AutogradValue or float
        args    (list): A list of raw values of each input (as floats)
        grad    (float): The derivative of the final loss with respect to this value (dL/da)
        value   (float): The value of the result of this operation
    '''

    def __init__(self, *args):
        self.parents = list(args)
        self.args = [arg.value if isinstance(arg, AutogradValue) else arg for arg in self.parents]
        self.grad = 0.
        self.value = self.forward_pass()

    def func(self, input):
        '''
        Compute the value of the operation given the inputs.
        For declaring a variable, this is just the identity function (return the input).

        Args:
            input (float): The input to the operation
        Returns:
            value (float): The result of the operation
        '''
        return input

    def grads(self, input):
        '''
        Compute the derivative of the operation with respect to each input.
        In the base case the derivative of the identity function is just 1. (da/da = 1).

        Args:
            input (float): The input to the operation
        Returns:
            grads (tuple): The derivative of the operation with respect to each input
                            Here there is only a single input, so we return a length-1 tuple.
        '''
        return (1,)
    
    def forward_pass(self):
        # Calls func to compute the value of this operation 
        return self.func(*self.args)
    
    def __repr__(self):
        # Python magic function for string representation.
        return str(self.value)

class _add(AutogradValue):
    def func(self, a, b):
        return a + b
    
    def grads(self, a, b):
        return 1., 1.

class _sub(AutogradValue):
    def func(self, a, b):
        return a - b
    
    def grads(self, a, b):
        return 1., -1.

class _neg(AutogradValue):
    def func(self, a):
        return -a
    
    def grads(self, a):
        return (-1.,)
    
class _mul(AutogradValue):
    def func(self, a, b):
        return a * b
    
    def grads(self, a, b):
        return b, a

class _div(AutogradValue):
    def func(self, a, b):
        return a / b
    
    def grads(self, a, b):
        return 1 / b, -a / (b * b)
    
class _exp(AutogradValue):
    def func(self, a):
        return math.exp(a)
    
    def grads(self, a):
        return (math.exp(a),)

class _log(AutogradValue):
    def func(self, a):
        return math.log(a)
    
    def grads(self, a):
        return (1 / a,)
    
def exp(a):
    return _exp(a) if isinstance(a, AutogradValue) else math.exp(a)
def log(a):
    return _log(a) if isinstance(a, AutogradValue) else math.log(a)

AutogradValue.exp = lambda a: _exp(a)
AutogradValue.log = lambda a: _log(a)
AutogradValue.__add__ = lambda a, b: _add(a, b)
AutogradValue.__radd__ = lambda a, b: _add(b, a)
AutogradValue.__sub__ = lambda a, b: _sub(a, b)
AutogradValue.__rsub__ = lambda a, b: _sub(b, a)
AutogradValue.__neg__ = lambda a: _neg(a)
AutogradValue.__mul__ = lambda a, b: _mul(a, b)
AutogradValue.__rmul__ = lambda a, b: _mul(b, a)
AutogradValue.__truediv__ = lambda a, b: _div(a, b)
AutogradValue.__rtruediv__ = lambda a, b: _div(b, a)
 
def backward_pass(self):
    grads = self.grads(*self.args)
    for node, grad in zip(self.parents, grads):
        if isinstance(node, AutogradValue):
            node.grad += self.grad * grad

AutogradValue.backward_pass = backward_pass

def backward(self):
    # We call backward on the loss, so dL/dL = 1
    self.grad = 1.
    queue = [self]
    order = []
    counts = {}
    while len(queue) > 0:
        node = queue.pop()
        
        if isinstance(node, AutogradValue):
            if node in counts:
                counts[node] += 1
            else:
                counts[node] = 1

            order.append(node)
            queue.extend(node.parents)
    
    
    for node in order:
        counts[node] -= 1
        if counts[node] == 0:
            node.backward_pass()


AutogradValue.backward = backward


def sigmoid(x):
    # Computes the sigmoid function
    return 1. / (1. + np.exp(-x))

class LogisticRegression:
    def __init__(self, dims):
        '''
        Args:
            dims (int): d, the dimension of each input
        '''
        self.weights = np.zeros((dims + 1, 1))

    def prediction_function(self, X, w):
        '''
        Get the result of our base function for prediction (i.e. x^t w)

        Args:
            X (array): An N x d matrix of observations.
            w (array): A (d+1) x 1 vector of weights.
        Returns:
            pred (array): A length N vector of f(X).
        '''
        X = np.pad(X, ((0,0), (0,1)), constant_values=1.)
        return X.dot(w)

    def predict(self, X):
        '''
        Predict labels given a set of inputs.

        Args:
            X (array): An N x d matrix of observations.
        Returns:
            pred (array): An N x 1 column vector of predictions in {0, 1}
        '''
        return (self.prediction_function(X, self.weights) > 0)
    
    def predict_probability(self, X):
        '''
        Predict the probability of each class given a set of inputs

        Args:
            X (array): An N x d matrix of observations.
        Returns:
            probs (array): An N x 1 column vector of predicted class probabilities
        '''
        return sigmoid(self.prediction_function(X, self.weights))

    def accuracy(self, X, y):
        '''
        Compute the accuracy of the model's predictions on a dataset

        Args:
            X (array): An N x d matrix of observations.
            y (array): A length N vector of labels.
        Returns:
            acc (float): The accuracy of the classifier
        '''
        y = y.reshape((-1, 1))
        return (self.predict(X) == y).mean()

    def nll(self, X, y, w=None):
        '''
        Compute the negative log-likelihood loss.

        Args:
            X (array): An N x d matrix of observations.
            y (array): A length N vector of labels.
            w (array, optional): A (d+1) x 1 matrix of weights.
        Returns:
            nll (float): The NLL loss
        '''
        if w is None:
            w = self.weights

        y = y.reshape((-1, 1))
        xw = self.prediction_function(X, w)
        py = sigmoid((2 * y - 1) * xw)
        return -(np.log(py)).sum()
    
    def nll_gradient(self, X, y):
        '''
        Compute the gradient of the negative log-likelihood loss.

        Args:
            X (array): An N x d matrix of observations.
            y (array): A length N vector of labels.
        Returns:
            grad (array): A length (d + 1) vector with the gradient
        '''
        y = y.reshape((-1, 1))
        xw = self.prediction_function(X, self.weights)
        py = sigmoid((2 * y - 1) * xw)
        grad = ((1 - py) * (2 * y - 1)).reshape((-1, 1)) * np.pad(X, [(0,0), (0,1)], constant_values=1.)
        return -np.sum(grad, axis=0)
    
    def nll_and_grad_no_autodiff(self, X, y):
        # Compute nll_and_grad without automatic diferentiation
        return self.nll(X, y), self.nll_gradient(X, y)
    
class NeuralNetwork(LogisticRegression):
    def __init__(self, dims, hidden_sizes=[]):
        self.weights = [1. * np.random.normal(scale=1., size=(i + 1, o)) for (i, o) in zip([dims] + hidden_sizes, hidden_sizes + [1])]

def prediction_function(self, X, w):
    '''
    Get the result of our base function for prediction (i.e. x^t w)

    Args:
        X (array): An N x d matrix of observations.
        w (list of arrays): A list of weight matrices
    Returns:
        pred (array): An N x 1 matrix of f(X).
    '''
    for wi in w[:-1]:
        X = sigmoid(LogisticRegression.prediction_function(self, X, wi))
    return LogisticRegression.prediction_function(self, X, w[-1]).reshape((-1, 1))


NeuralNetwork.prediction_function = prediction_function

def plotRegression(x, y, xvalid=None, yvalid=None, loss_history=None, valid_loss_history=None, model=None):
    f, ax = plt.subplots(1, 3, figsize=(15, 4))

    if not (loss_history is None):
        ax[0].plot(loss_history, label='Training loss')
    if not (valid_loss_history is None):
        ax[0].plot(valid_loss_history, label='Validation loss')
    if not (loss_history is None and valid_loss_history is None):
        ax[0].set_title('Loss vs. iterations')
        ax[0].legend()
    
    ax[1].scatter(x, y, c=sns.palettes.SEABORN_PALETTES['deep'][0])
    
    xp = torch.linspace(-5, 5, 200).reshape((-1, 1))
    if model is not None:
       model.train()
       ax[1].plot(xp, model(xp).detach(), c=sns.palettes.SEABORN_PALETTES['deep'][3])
    ax[1].set_title('Training data')

    if not (xvalid is None and yvalid is None):
        ax[2].scatter(xvalid, yvalid, c=sns.palettes.SEABORN_PALETTES['deep'][1])
        ax[2].set_title('Validation data')
        if model is not None:
            model.eval()
            ax[2].plot(xp, model(xp).detach(), c=sns.palettes.SEABORN_PALETTES['deep'][3])
            model.train()
    sns.despine(f)

def get_dataset(name):
    data = np.load('data.npz')
    hhimages, hhlabels, hhlabel_names = data['hhimages'], data['hhlabels'], data['hhlabel_names']
    mnistimages, mnistlabels, mnistlabel_names = data['mnistimages'], data['mnistlabels'], data['mnistlabel_names']
    images, labels, label_names = None, None, list(map(str, range(10)))
    if name == 'ones_and_zeros':
        images, labels = mnistimages, mnistlabels
        images, labels = images[labels <= 1], labels[labels <= 1]
    if name == 'mnist':
        images, labels = mnistimages, mnistlabels
    elif name == 'horses_and_humans':
        images, labels = hhimages, hhlabels
        label_names = hhlabel_names

    f, subplots = plt.subplots(8, 8, figsize=(20, 20))
    i = 0
    for row in subplots:
        for subplot in row:
            subplot.imshow(images[i], cmap='gray')
            subplot.axis('off')
            i += 1
    plt.show()
    return images, labels, label_names

def isclose(x, y, mag=1e-1):
    return np.abs(x - y) < mag

def test_gradient_descent(func, mse_loss, x, y, xvalid, yvalid, l1=False):
    offset = 1. if l1 else 0.
    class LinearZeros(nn.Module):
        def __init__(self, in_dimensions, out_dimensions):
            super().__init__()
            self.weights = nn.Parameter(torch.zeros(in_dimensions, out_dimensions) + offset)
            self.bias = nn.Parameter(torch.zeros(out_dimensions))

        def forward(self, x):
            return torch.matmul(x, self.weights) + self.bias
        
    model = LinearZeros(1, 1)
    if l1:
        losses, valid_losses = func(model, mse_loss, x, y, xvalid, yvalid, steps=1, lr=0.1, l1_weight=10.)
        #assert isclose(losses[0], 25.523453, 1e-3), 'Loss computed incorrectly!'
        #assert isclose(model.weights.item(), -2.9291603565216064, 1e-3), 'Update applied incorrectly'
    else:
        losses, valid_losses = func(model, mse_loss, x, y, xvalid, yvalid, steps=1, lr=0.1)
        #assert isclose(losses[0], 14.633200645446777), 'Loss computed incorrectly!'
        assert model.weights.item() != 0., 'Gradient descent update not applied!'
        #assert isclose(model.weights.item(), -0.2489, 1e-3), 'Update applied incorrectly'
        assert model.weights.grad is None, 'Did not zero gradients!'
        #assert isclose(valid_losses[0], 13.963179588317871), 'Validation loss computed incorrectly!'
    print('Passed!')

def test_dropout(dropout):
    d = dropout(rate = 0.3)
    d.train()
    X = torch.ones((500, 500))
    dx = d(X)
    assert torch.sum(dx) != torch.sum(X), 'Dropout does not change input!'
    assert torch.all((dx == 0.) | (dx == 1.)), 'Dropout not randomly dropping inputs in training mode!'
    assert torch.abs(torch.mean(dx) - 0.7) < 1e-2, 'Dropout applied with incorrect rate!'

    d.eval()
    dx = d(X)
    assert torch.all((dx - 0.7) < 1e-4), 'Dropout not scaling at test time!'
    print('Passed!')

def test_build(model, layertype, dropout_type=None, type='zeros'):
    param_shapes = [p.shape for p in model.parameters()]
    assert len(param_shapes) > 9, 'Too few layers!'
    assert len(param_shapes) < 11, 'Too many layers!'
    assert all(s1.shape == s2 for  (s1, s2) in zip( model.parameters(), [torch.Size([1, 20]),
            torch.Size([20]),
            torch.Size([20, 20]),
            torch.Size([20]),
            torch.Size([20, 20]),
            torch.Size([20]),
            torch.Size([20, 20]),
            torch.Size([20]),
            torch.Size([20, 1]),
            torch.Size([1])])), 'Incorrect Layer shapes!'
    
    if type == 'zeros':
        assert all(p.sum() == 0. for p in model.parameters()), 'Parameters not initialized to 0.!'

    correct_layers = [layertype,
                nn.ReLU,
                layertype,
                nn.ReLU,
                layertype,
                nn.ReLU,
                layertype,
                nn.ReLU,
                layertype]
    
    if not (dropout_type is None):
        correct_layers = [dropout_type, layertype,
                nn.ReLU,
                dropout_type,
                layertype,
                nn.ReLU,
                dropout_type,
                layertype,
                nn.ReLU,
                dropout_type,
                layertype,
                nn.ReLU,
                dropout_type,
                layertype]
    
    assert isinstance(list(model.children())[-1], layertype), 'Last layer is incorrect type (could be an activation or wrong linear layer type)!'
    assert all(isinstance(l, t) for (l, t) in zip(model.children(), correct_layers)), 'Incorrect layer order!'
    print('Passed!')


def test_normal(layertype):
   weights = layertype(500, 500).weights
   assert not torch.all(weights == 0.), 'Weights initialized to 0!'
   assert torch.abs(torch.mean(weights)) < 0.1, 'Weights have incorrect mean!'
   assert torch.abs(torch.std(weights) - torch.tensor(1.)) <  1e-1, 'Weights initialized with incorrect variance!'
   print('Passed!')

def test_kaiming(layertype):
   weights = layertype(5, 1000).weights
   assert not torch.all(weights == 0.), 'Weights initialized to 0!'
   assert torch.abs(torch.mean(weights)) < 0.1, 'Weights have incorrect mean!'
   assert torch.abs(torch.std(weights) - (1. / torch.sqrt(torch.tensor(5.)))) <  1e-1, 'Weights initialized with incorrect variance!'
   print('Passed!')

def aslist(value):
    # Converts iterables to lists and single values to a single-element lists
    try:
        return list(value)
    except:
        return (value,)

def gradient_descent(model, X, y, lr=1e-6, steps=250):
    losses = []
    progress = tqdm.trange(steps)
    for i in progress:
        loss, g = model.nll_and_grad(X, y)
        if isinstance(model.weights, np.ndarray):
            model.weights = model.weights - lr * g
        else:
            model.weights = [wi - lr * gi for (wi, gi) in zip(aslist(model.weights), aslist(g))]
        accuracy = model.accuracy(X, y)

        losses.append(loss)
        progress.set_description('Loss %.2f, accuracy: %.2f' % (loss, accuracy))
        
    return losses

def plot_boundary(model, X, y):
    xrange = (-X[:, 0].min() + X[:, 0].max()) / 10
    yrange = (-X[:, y].min() + X[:, y].max()) / 10
    feature_1, feature_2 = np.meshgrid(
        np.linspace(X[:, 0].min() - xrange, X[:, 0].max() + xrange),
        np.linspace(X[:, 1].min() - yrange, X[:, 1].max() + yrange)
    )
    grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
    y_pred = np.reshape(model.predict(grid), feature_1.shape)
    display = DecisionBoundaryDisplay(
        xx0=feature_1, xx1=feature_2, response=y_pred
    )
    display.plot()
    display.ax_.scatter(
        X[:, 0], X[:, 1], c=y.flatten(), edgecolor="black"
    )
    plt.show()