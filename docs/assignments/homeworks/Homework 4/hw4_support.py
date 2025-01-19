import random
import math
import tqdm
import autograd.numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

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
        X = np.pad(X, ((0,0), (0,1)), constant_values=1., mode='constant')
        return np.dot(X, w)

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
        grad = ((1 - py) * (2 * y - 1)).reshape((-1, 1)) * np.pad(X, [(0,0), (0,1)], constant_values=1., mode='constant')
        return -np.sum(grad, axis=0)

    def nll_and_grad_no_autodiff(self, X, y):
        # Compute nll_and_grad without automatic diferentiation
        return self.nll(X, y), self.nll_gradient(X, y)

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


def test_backward_pass_numbers(AGClass):
    class BPTest(AGClass):
        def __init__(self, *args):
            self.parents = list(args)
            self.value = 1
            self.args = [arg.value if isinstance(arg, AGClass) else arg for arg in self.parents]
            self.grad = []

        def grads(self, *args):
            return self.value * len([a for a in self.parents if isinstance(a, AGClass)])
        
    class Test:
        def __init__(self, *args):
            self.parents = list(args)
            self.grad = []

    B, C, D = BPTest(1), BPTest(2), Test()
    A = BPTest(B, C)
    O = BPTest(B, D)
    Q = BPTest(C)

    A.grad = 5
    O.grad = 2
    O.value = 0
    Q.grad = 3
    
    A.backward_pass()
    O.backward_pass()
    Q.backward_pass()
    assert 'A' in B.grad, 'Did not add grad to parent grad!'
    assert ('A' in C.grad) and ('Q' in C.grad), 'Not all children in parents grad!'
    assert 'O' not in B.grad, 'Did not multiply by local derivative!'
    assert 'O' not in D.grad, 'Did not check that parent isinstance of AutogradValue!'
    print('Passed!')


def test_backward_pass(AGClass):
    class BPTest(AGClass):
        def __init__(self, *args):
            self.parents = list(args)
            self.value = 1
            self.args = [arg.value if isinstance(arg, AGClass) else arg for arg in self.parents]
            self.grad = []

        def grads(self, *args):
            return [self.value] * len([a for a in self.parents if isinstance(a, AGClass)])
        

    class Test:
        def __init__(self, *args):
            self.parents = list(args)
            self.grad = []

    B, C, D = BPTest(), BPTest(), Test()
    A = BPTest(B, C)
    O = BPTest(B, D)
    Q = BPTest(C)

    A.grad = ['A']
    O.grad = ['O']
    O.value = 0
    Q.grad = ['Q']
    
    A.backward_pass()
    O.backward_pass()
    Q.backward_pass()
    assert 'A' in B.grad, 'Did not add grad to parent grad!'
    assert ('A' in C.grad) and ('Q' in C.grad), 'Not all children in parents grad!'
    assert 'O' not in B.grad, 'Did not multiply by local derivative!'
    assert 'O' not in D.grad, 'Did not check that parent isinstance of AutogradValue!'
    print('Passed!')

        
def test_backward(AGClass):
    class BPTest(AGClass):
        def __init__(self, *args):
            self.parents = list(args)
            self.value = 1
            self.args = [arg.value if isinstance(arg, AGClass) else arg for arg in self.parents]
            self.grad = 0.
            self.visited = False

        def backward_pass(self, *args):
            assert not self.visited, 'Called backward pass on same value twice!'
            assert all([not p.visited for p in self.parents if isinstance(p, AGClass)]), 'Called backward_pass on a parent before its children!'
            self.visited = True

        def grads(self, *args):
            raise NotADirectoryError()

    class Test:
        def __init__(self, *args):
            self.visited = True

        def grads(self, *args):
            assert False, 'Visited non-autograd value node!'
        

    all_nodes = [BPTest(), Test(), BPTest(BPTest()), BPTest(BPTest(BPTest()))]
    for i in range(10):
        random.shuffle(all_nodes)
        all_nodes.append(BPTest(all_nodes[0], all_nodes[1]))

    output = BPTest()
    for node in all_nodes:
        output = BPTest(output, node)

    flag = False
    try:   
        output.backward()
    except NotADirectoryError:
        flag = True

    
    if flag or all([(not n.visited) for n in all_nodes]):
        print('Warning: No nodes visited, test results will be unreliable! \n\t Please call node.backward_pass() instead of backward_pass(node)')
    else:
        assert all([(not n.visited) for n in all_nodes]) or all([n.visited for n in all_nodes]), 'Not all nodes visited!'
        print('Passed!')

def test_operators(AGClass):
    a = AGClass(5.)
    b = AGClass(2.)

    av, bv = 5., 2.

    assert (a + b).value == 7, '_add gave incorrect value!'
    assert tuple((a + b).grads(av, av)) == (1., 1.), '_add gave incorrect derivatives!'

    assert (a - b).value == 3, '_sub gave incorrect value!'
    assert tuple((a - b).grads(av, bv)) == (1., -1.), '_sub gave incorrect derivatives!'

    assert (-b).value == -2, '_neg gave incorrect value!'
    assert tuple((-b).grads(bv)) == (-1.,), '_neg gave incorrect derivatives!'

    assert (a * b).value == 10, '_mul gave incorrect value!'
    assert tuple((a * b).grads(av, bv)) == (2., 5.), '_mul gave incorrect derivatives!'

    assert (a / b).value == 2.5, '_div gave incorrect value!'
    assert tuple((a / b).grads(av, bv)) == (1. / 2., -5 / 4.), '_div gave incorrect derivatives!'

    assert (a.exp()).value == math.exp(5.), '_exp gave incorrect value!'
    assert np.isclose(a.exp().grads(av)[0], math.exp(5)), '_exp gave incorrect derivatives!'

    assert (a.log()).value == math.log(5.), '_exp gave incorrect value!'
    assert np.isclose(a.log().grads(av)[0], 1 / 5.), '_log gave incorrect derivatives!'

    print('Passed!')

def test_autograd(AGClass):
    import autograd.numpy as anp
    import autograd

    def anp_grad(f):
        def fwrapped(args):
            return f(*args)

        def func(*args):
            args = [anp.array(arg) for arg in args]
            return [a.item() for a in autograd.grad(fwrapped)(args)]
        
        return func
        
    def mygrad(f):
        def func(*args):
            args = [AGClass(arg) for arg in args]
            result = f(*args)
            result.backward()
            return [arg.grad for arg in args]
        
        return func

    def test(f, *args):
        assert np.all(np.array(anp_grad(f)(a, b)) == np.array(mygrad(f)(a, b)))
        print('Passed!')

    a = 5.
    b = 2.

    f = lambda a, b: a + b
    test(f, a, b)

    f = lambda a, b: a - b
    test(f, a, b)

    f = lambda a, b: a - 3
    test(f, a, b)

    f = lambda a, b: 3 - a
    test(f, a, b)

    f = lambda a, b: a / 3
    test(f, a, b)

    f = lambda a, b: 3 / a
    test(f, a, b)

    f = lambda a, b: a / b
    test(f, a, b)

    f = lambda a, b: -a
    test(f, a, b)

    f = lambda a, b: a * b
    test(f, a, b)

    f = lambda a, b: b * a
    test(f, a, b)

def test_element_map(unary_map, binary_map, Matrix):
    def element_map(f, *args):
        if len(args) == 1:
            return unary_map(f, *args)
        return binary_map(f, *args)

    a = Matrix([[1, 2], [3, 4]])
    assert isinstance(element_map(lambda x: x ** 2, a), Matrix), 'Result should be a martrix!'
    assert element_map(lambda x: x ** 2, a).shape == a.shape, 'Result should have the same shape as input!'
    assert tuple(element_map(lambda x: x ** 2, a).value) == ((1, 4), (9, 16)), 'Function not applied to every element!'

    b = Matrix([[1, 1], [2, 2]])
    assert isinstance(element_map(lambda x, y: x + y, a, b), Matrix), 'Result should be a martrix!'
    assert element_map(lambda x, y: x + y, a, b).shape == a.shape, 'Result should have the same shape as input!'
    assert tuple(element_map(lambda x, y: x + y, a, b).value) == ((1, 3), (5, 6)), 'Function not applied to every pair of elements!'

    b = Matrix([[1], [2]])
    assert isinstance(element_map(lambda x, y: x + y, a, b), Matrix), 'Result should be a martrix!'
    assert element_map(lambda x, y: x + y, a, b).shape == a.shape, 'Result should broadcast rows of length 1!'
    assert tuple(element_map(lambda x, y: x + y, a, b).value) == ((1, 3), (5, 6)), 'Function not applied to every pair of elements in broadcast matricies!'

    b = Matrix([[1, 2]])
    assert isinstance(element_map(lambda x, y: x + y, a, b), Matrix), 'Result should be a martrix!'
    assert element_map(lambda x, y: x + y, a, b).shape == a.shape, 'Result should broadcast columns of length 1!'
    assert tuple(element_map(lambda x, y: x + y, a, b).value) == ((1, 4), (4, 6)), 'Function not applied to every pair of elements in broadcast matricies!'

    b = Matrix([[1]])
    assert isinstance(element_map(lambda x, y: x + y, a, b), Matrix), 'Result should be a martrix!'
    assert element_map(lambda x, y: x + y, a, b).shape == a.shape, 'Result should broadcast both dimensions if needed!'
    assert tuple(element_map(lambda x, y: x + y, a, b).value) == ((1, 3), (4, 5)), 'Function not applied to every pair of elements in broadcast matricies!'

    a = Matrix([[1, 2]])
    b = Matrix([[1], [2]])
    assert isinstance(element_map(lambda x, y: x + y, a, b), Matrix), 'Result should be a martrix!'
    assert element_map(lambda x, y: x + y, a, b).shape == (2,2), 'Result should broadcast dimensions of length 1!'
    assert tuple(element_map(lambda x, y: x + y, a, b).value) == ((1, 2), (2, 4)), 'Function not applied to every pair of elements in broadcast matricies!'

    a = Matrix([[1, 2], [3, 4]])
    b = 1
    assert isinstance(element_map(lambda x, y: x + y, a, b), Matrix), 'Inputs should be converted to matrix!'
    assert element_map(lambda x, y: x + y, a, b).shape == a.shape, 'Result should broadcast both dimensions if needed!'
    assert tuple(element_map(lambda x, y: x + y, a, b).value) == ((1, 3), (4, 5)), 'Function not applied to every pair of elements in broadcast matricies!'

    a = 2
    b = 1
    assert isinstance(element_map(lambda x, y: x + y, a, b), Matrix), 'Inputs should be converted to matrix!'
    assert element_map(lambda x, y: x + y, a, b).shape == (1, 1), 'Result should broadcast both dimensions if needed!'
    assert tuple(element_map(lambda x, y: x + y, a, b).value) == ((3,),), 'Function not applied to every pair of elements in broadcast matricies!'

    print('Passed!')


def test_dot(dot, Matrix):
    a = np.random.randn(3, 5)
    b = np.random.randn(5, 2)

    am, bm = Matrix(a), Matrix(b)
    cm = dot(am, bm)
    assert cm.shape == (3, 2), 'Result incorrect shape!'
    assert np.all(np.isclose(np.array(cm.value), np.dot(a, b))), 'Result has incorrect value'
    print('Passed!')

def test_sum(Matrix):
    a = np.random.randn(3, 5)
    assert np.all(np.isclose(a.sum(), Matrix(a).sum())), 'Result has incorrect value'
    print('Passed!')


def test_autograd_forward(AGClass):
    import autograd.numpy as anp
    import autograd

    def anp_grad(f):
        def fwrapped(args):
            return f(*args)

        def func(*args):
            args = [anp.array(arg) for arg in args]
            return [a.item() for a in autograd.grad(fwrapped)(args)]
        
        return func
        
    def mygrad(f, check=True):
        def func(*args):
            args = [AGClass(arg) for arg in args]
            result = f(*args)

            if check:
                assert hasattr(result, 'tangents'), 'Resulting value should have tangents dict'
                assert isinstance(result.tangents, dict), 'Tangents should be of type dict'
                assert all([(arg in result.tangents) for arg in args]), 'Missing derivatives with respect to some inputs!'
            return [result.tangents[arg] for arg in args]
        
        return func

    def test(f, *args):
        assert np.all(np.array(anp_grad(f)(*args)) == np.array(mygrad(f)(*args)))
        print('Passed!')

    a = 5.
    b = 2.

    f = lambda a, b: a + b
    test(f, a, b)

    f = lambda a, b: a - b
    test(f, a, b)

    f = lambda a: a - 3
    test(f, a)

    f = lambda a: 3 - a
    test(f, a)

    f = lambda a: a / 3
    test(f, a)

    f = lambda a: 3 / a
    test(f, a)

    f = lambda a, b: a / b
    test(f, a, b)

    f = lambda a: -a
    test(f, a)

    f = lambda a, b: a * b
    test(f, a, b)

    f = lambda a, b: b * a
    test(f, a, b)

def testnn(NeuralNetwork, Matrix):
    net = NeuralNetwork(3, [4, 5, 5])
    assert isinstance(net.weights[1], Matrix), 'Weights must be Matrix objects!'
    valid_weights = [
         ((4, 5), (6, 6), (7, 6), (7, 1)),
         ((5, 4), (6, 6), (6, 7), (7, 1)),
         ((5, 4), (6, 6), (6, 7), (1, 7)),
    ]
    unbiased_weights = [
         ((3, 5), (5, 6), (7, 6), (7, 1)),
         ((5, 4), (6, 6), (6, 7), (7, 1)),
         ((5, 4), (6, 6), (6, 7), (1, 7)),
    ]
    assert tuple([w.shape for w in net.weights]) in valid_weights, 'Weights not correct shape!'
    assert net.weights[2].value[1][1] - math.floor(net.weights[2].value[1][1]) != 0, 'Weights not random'

    np.random.seed(63)
    net.weights = [Matrix(0.2 * np.random.randn(2,3)), Matrix(np.random.randn(3,1))]
    output = net.prediction_function(np.random.randn(5, 2), net.weights)


def test_wrap_unwrap(wrap_array, unwrap_gradient, AutogradValue):
    a = np.random.randn(10, 7)
    try:
        aw = wrap_array(a)
    except:
        assert False, 'wrap_array should work on 2-dimensional (matrix) inputs!'
    assert isinstance(aw, np.ndarray), 'Wrapped array is not a numpy array!'
    assert aw.shape == a.shape, 'Wrapped array different shape than input!'
    assert np.all([isinstance(e, AutogradValue) for e in aw.flatten()]), 'Elements of wrapped array not of type AutogradValue!'
    assert np.all([np.isclose(e.value, et) for (e, et) in zip(aw.flatten(), a.flatten())]), 'Elements do not equal original array!'

    grads = np.random.randn(*a.shape).flatten()
    for agv, g in zip(aw.flatten(), grads):
        agv.grad = g

    uaw = unwrap_gradient(aw)
    assert isinstance(uaw, np.ndarray), 'Unwrapped array is not a numpy array!'
    assert uaw.shape == a.shape, 'Unwrapped array different shape than input!'
    assert np.all([(not isinstance(e, AutogradValue)) for e in uaw.flatten()]), 'Elements of unwrapped array are of type AutogradValue!'
    assert np.all([np.isclose(e, et) for (e, et) in zip(uaw.flatten(), grads.flatten())]), 'Elements do not equal original gradients!'
    
    print('Passed!')

def test_nn_constructor(NeuralNetwork, AutogradValue=None):
    net = NeuralNetwork(20, [40, 50, 50])

    shapes = [(20, 40), (40, 50), (50, 50), (50, 1)]

    for i, ((x, y), w) in enumerate(zip(shapes, net.weights)) :
        assert isinstance(w, np.ndarray), 'Weight %d is not a numpy array!' % i
        #assert not (isinstance(w.flatten()[0], AutogradValue)), 'Weight %d entries are AutogradValues (should be float except in nll_and_grad)!' % i
        possible_elems = [x * y, (x + 1) * y]
        assert int(np.prod(w.shape)) in possible_elems, 'Weight %d has incorrect number of elements!' % i
        assert len(np.unique(w.flatten())) == len(w.flatten()), 'Weight %d entries are not random!' % i
        assert np.abs(w.mean()) < 1., 'Mean of weight matrix %d is far from 0 (did you use np.random.normal?)!' % i
        assert w.std() > 1e-3, 'Std deviation of weight matrix %d is far from 0.01! (did you set scale?)!' % i
    if net.weights[-1].ndim == 1:
        print('Warning: final prediction weights may cause issues if 1 dimensional, try: self.weights[-1] = self.weights[-1].reshape((-1, 1))')
    print('Passed!')

def test_nn_prediction_function(NeuralNetwork):
    nn = NeuralNetwork(2, [4, 5, 4])
    for i in range(len(nn.weights)):
        wi = 0. * nn.weights[i]
        if wi.ndim == 1:
            wi[0] = i + 2
        else:
            wi[0,0] = i + 2
        nn.weights[i] = wi 
        
    output = nn.prediction_function(np.ones((3, 2)), nn.weights)
    if output.ndim < 2:
        print('Warning: output expected to be 2-dimensional (a column vector). This may be fine, but it could conflict with other assumptions made by your code!')
    assert not np.allclose(output, 0.99248508), 'Maybe applied sigmoid at end of prediction function! (The last call should just be phi(x) w_0)'
    assert np.allclose(output, 4.88332188, 1e-3), 'Incorrect output value!'
    
    print('Passed!')


def test_forward_mode(ForwardValue):
    a, b, c = ForwardValue(0.), ForwardValue(0.), ForwardValue(0.)

    x = ForwardValue(3.)
    x.forward_grads = {a: 1, b:7}
    y = ForwardValue(2.)
    y.forward_grads = {a: 2, b: -2, c: 6}

    w = ForwardValue(-3.)
    w.forward_grads = {c: -3}
    z = ForwardValue(5.)
    z.forward_grads = {b: 4}

    out1 = w + z
    assert (b in out1.forward_grads) and (c in out1.forward_grads), 'forward_grads dicts not combined!'
    assert (out1.forward_grads[b] == z.forward_grads[b]) and (out1.forward_grads[c] == w.forward_grads[c]), 'forward_grads values do not match parents!'

    out2 = x + z
    assert (b in out2.forward_grads) and (a in out2.forward_grads), 'forward_grads dicts not combined!'
    assert (out2.forward_grads[a] == x.forward_grads[a]) and (out2.forward_grads[b] == (x.forward_grads[b] + z.forward_grads[b])), 'Corresponding values in forward_grads not added!'

    final = ((x + z) * y) / w
    assert np.isclose(final.forward_grads[a], -6.0, 1e-3), 'Incorrect value in forward_grads!'
    assert np.isclose(final.forward_grads[b], -2.0, 1e-3), 'Incorrect value in forward_grads!'
    assert np.isclose(final.forward_grads[c], -10.666666666666668, 1e-3), 'Incorrect value in forward_grads!'
    print('Passed!')

        
