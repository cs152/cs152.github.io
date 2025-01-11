import random
import math
import tqdm
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


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
        X[:, 0], X[:, 1], c=y, edgecolor="black"
    )
    plt.show()


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
    A = BPTest(B, C, D)
    O = BPTest(B)
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
    assert 'A' not in D.grad, 'Did not check that parent isinstance of AutogradValue!'
    print('Passed!')

        
def test_backward(AGClass):
    class BPTest(AGClass):
        def __init__(self, *args):
            self.parents = list(args)
            self.value = 1
            self.args = [arg.value if isinstance(arg, AGClass) else arg for arg in self.parents]
            self.grad = []
            self.visited = False

        def backward_pass(self):
            assert not self.visited, 'Called backward pass on same value twice!'
            assert all([not p.visited for p in self.parents if isinstance(p, AGClass)]), 'Called backward_pass on a parent before its children!'
            self.visited = True

    class Test:
        def __init__(self, *args):
            self.visited = True

        def backward_pass(self):
            assert False, 'Visited non-autograd value node!'
        

    all_nodes = [BPTest(), Test(), BPTest(BPTest()), BPTest(BPTest(BPTest()))]
    for i in range(10):
        random.shuffle(all_nodes)
        all_nodes.append(BPTest(all_nodes[0], all_nodes[1]))

    output = BPTest()
    for node in all_nodes:
        output = BPTest(output, node)

    output.backward()
    assert all([n.visited for n in all_nodes]), 'Not all nodes visited!'
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