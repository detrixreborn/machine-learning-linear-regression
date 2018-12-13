import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt


mx.random.seed(1)
data_ctx = mx.cpu()
model_ctx = mx.cpu()


num_inputs = 2
num_outputs = 1
num_examples = 10000
epochs = 10
learning_rate = .0001
batch_size = 4
num_batches = num_examples/batch_size
losses = []


def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2


X = nd.random_normal(shape=(num_examples, num_inputs), ctx=data_ctx)
noise = .1 * nd.random_normal(shape=(num_examples,), ctx=data_ctx)
y = real_fn(X) + noise

train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                   batch_size=batch_size, shuffle=True)


w = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
b = nd.random_normal(shape=num_outputs, ctx=model_ctx)
params = [w, b]

for param in params:
    param.attach_grad()


def net(X):
    return mx.nd.dot(X, w) + b


def square_loss(yhat, y):
    return nd.mean((yhat - y) ** 2)


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def plot(losses, X, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             net(X[:sample_size, :]).asnumpy(), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             real_fn(X[:sample_size, :]).asnumpy(), '*g', label='Real')
    fg2.legend()

    plt.show()


for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx).reshape((-1, 1))
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += loss.asscalar()

    print("Epoch %s, batch %s. Mean loss: %s" % (e, i, cumulative_loss/num_batches))
    losses.append(cumulative_loss/num_batches)

plot(losses, X)
