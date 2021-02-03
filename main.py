import math
import tensorflow as tf

from mpi4py import MPI

minibatch_size = 64
max_epoch = 5


@tf.custom_gradient
def grad_op(x, w):
    def grad(dy):
        return None, dy

    return tf.identity(x), grad


class GradLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=input_shape, trainable=True)

    def call(self, inputs, **kwargs):
        return grad_op(inputs, self.kernel)


class Input(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(128, activation='relu')

    def call(self, inputs, **kwargs):
        x = self.flatten(inputs)
        x = self.dense(x)
        return x

    def get_config(self):
        raise NotImplementedError


class Block(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.grad_layer = GradLayer()
        self.dense_1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(128, activation='relu')

    def call(self, inputs, **kwargs):
        x = self.grad_layer(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x

    def get_config(self):
        raise NotImplementedError


class Head(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.grad_layer = GradLayer()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.grad_layer(inputs)
        x = self.dropout(x)
        x = self.dense(x)
        return x

    def get_config(self):
        raise NotImplementedError


class DistributedModel:

    def __init__(self,
                 comm,
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss_fn: tf.keras.losses.Loss):
        self._comm = comm
        self._size = comm.Get_size()
        self._rank = comm.Get_rank()
        self._first_rank = 0
        self._last_rank = self._size - 1
        self._next_rank = self._rank + 1 if self._rank + 1 <= self._last_rank else MPI.PROC_NULL
        self._prev_rank = self._rank - 1 if self._rank - 1 >= self._first_rank else MPI.PROC_NULL
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        if self.is_first_node():
            self._model = Input()
        elif self.is_last_node():
            self._model = Head()
        else:
            self._model = Block()

    def _get_microbatches(self, minibatch):
        assert minibatch_size % self._size == 0
        microbatch_size = minibatch_size // self._size
        microbatches = tf.data.Dataset \
            .from_tensor_slices(minibatch) \
            .batch(microbatch_size)
        return microbatches

    def _forward(self, minibatch):
        microbatches = self._get_microbatches(minibatch)
        predictions, tapes, losses = [], [], []
        for microbatch in microbatches:
            x, y_true = microbatch
            with tf.GradientTape() as tape:
                if self.is_first_node():
                    y_pred = self._model(x)
                    self._comm.send(y_true, dest=self._last_rank)
                    self._comm.send(y_pred, dest=self._next_rank)
                elif self.is_last_node():
                    y_true = self._comm.recv(source=self._first_rank)
                    recvd = self._comm.recv(source=self._prev_rank)
                    y_pred = self._model(recvd)
                    loss = self._loss_fn(y_true, y_pred)
                    losses.append(loss)
                else:
                    recvd = self._comm.recv(source=self._prev_rank)
                    y_pred = self._model(recvd)
                    self._comm.send(y_pred, dest=self._next_rank)
            predictions.append(y_pred)
            tapes.append(tape)
        return predictions, tapes, losses

    def _backward(self, predictions, tapes, losses):
        grads = []
        for i in range(self._size):
            if self.is_first_node():
                partial_error = self._comm.recv(source=self._next_rank)
                grad = tapes[i].gradient(predictions[i],
                                         self._model.trainable_weights,
                                         output_gradients=partial_error)
            elif self.is_last_node():
                grad = tapes[i].gradient(losses[i], self._model.trainable_weights)
                self._comm.send(grad[0], dest=self._prev_rank)
            else:
                partial_error = self._comm.recv(source=self._next_rank)
                grad = tapes[i].gradient(predictions[i],
                                         self._model.trainable_weights,
                                         output_gradients=partial_error)
                self._comm.send(grad[0], dest=self._prev_rank)
            grads.append(grad)
        grads = list(map(list, zip(*grads)))
        grads = [tf.reduce_mean(grad, axis=0) for grad in grads]
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_weights))

    def train_on_batch(self, minibatch):
        predictions, tapes, losses = self._forward(minibatch)
        self._backward(predictions, tapes, losses)
        loss = tf.reduce_mean(losses)
        return loss

    def predict_on_batch(self, minibatch):
        microbatches = self._get_microbatches(minibatch)
        predictions = []
        for microbatch in microbatches:
            x, _ = microbatch
            if self.is_first_node():
                y_pred = self._model(x)
                self._comm.send(y_pred, dest=self._next_rank)
            elif self.is_last_node():
                recvd = self._comm.recv(source=self._prev_rank)
                y_pred = self._model(recvd)
            else:
                recvd = self._comm.recv(source=self._prev_rank)
                y_pred = self._model(recvd)
                self._comm.send(y_pred, dest=self._next_rank)
            predictions.extend(y_pred.numpy())
        return predictions

    def is_first_node(self) -> bool:
        return self._rank == self._first_rank

    def is_last_node(self) -> bool:
        return self._rank == self._last_rank


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    n_train, n_test = len(x_train), len(x_test)
    n_train_minibatch, n_test_minibatch = math.floor(n_train / minibatch_size), math.ceil(n_test / minibatch_size)

    x_train = tf.data.Dataset \
        .from_tensor_slices((x_train, y_train)) \
        .batch(minibatch_size, drop_remainder=True) \
        .shuffle(n_train)
    x_test = tf.data.Dataset \
        .from_tensor_slices((x_test, y_test)) \
        .batch(minibatch_size)

    comm = MPI.COMM_WORLD
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    model = DistributedModel(comm, optimizer, loss_fn)

    if model.is_last_node():
        print('Training')
    for i in range(max_epoch):
        if model.is_last_node():
            print(f'Epoch {i + 1}/{max_epoch}')
            progbar = tf.keras.utils.Progbar(n_train_minibatch, stateful_metrics=['loss'])
        for idx, minibatch in enumerate(x_train):
            loss = model.train_on_batch(minibatch)
            if model.is_last_node():
                # noinspection PyUnboundLocalVariable
                progbar.add(1, values=[('loss', loss)])

    if model.is_last_node():
        print('Testing')
        progbar = tf.keras.utils.Progbar(n_test_minibatch, stateful_metrics=['acc'])
    for idx, minibatch in enumerate(x_test):
        _, y_true = minibatch
        y_pred = model.predict_on_batch(minibatch)
        accuracy.update_state(y_true, y_pred)
        if model.is_last_node():
            # noinspection PyUnboundLocalVariable
            progbar.add(1, values=[('acc', accuracy.result())])


if __name__ == '__main__':
    main()
