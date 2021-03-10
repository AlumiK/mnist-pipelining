import math
import tensorflow as tf

from mpi4py import MPI
from typing import Sequence, Dict, Optional

batch_size = 64
max_epoch = 5


# noinspection PyUnusedLocal
@tf.custom_gradient
def partial_error_op(x, kernel):
    def partial_error(dy):
        return None, dy

    x = tf.identity(x)
    return x, partial_error


class GradLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=input_shape, trainable=True)

    def call(self, inputs, **kwargs):
        return partial_error_op(inputs, self.kernel)


class Input(tf.keras.Model):

    def __init__(self, hidden_dims: Sequence):
        super().__init__()
        self._hidden_dims = hidden_dims
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.Sequential()
        for hidden_dim in hidden_dims:
            self.hidden.add(tf.keras.layers.Dense(hidden_dim, activation='relu'))

    def call(self, inputs, **kwargs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return x

    def get_config(self):
        return {
            'hidden_dims': self._hidden_dims
        }


class Block(tf.keras.Model):

    def __init__(self, hidden_dims):
        super().__init__()
        self._hidden_dims = hidden_dims
        self.grad_layer = GradLayer()
        self.hidden = tf.keras.Sequential()
        for hidden_dim in hidden_dims:
            self.hidden.add(tf.keras.layers.Dense(hidden_dim, activation='relu'))

    def call(self, inputs, **kwargs):
        x = self.grad_layer(inputs)
        x = self.hidden(x)
        return x

    def get_config(self):
        return {
            'hidden_dims': self._hidden_dims
        }


class Head(tf.keras.Model):

    def __init__(self, dropout_rate: float):
        super().__init__()
        self._dropout_rate = dropout_rate
        self.grad_layer = GradLayer()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.grad_layer(inputs)
        x = self.dropout(x)
        x = self.dense(x)
        return x

    def get_config(self):
        return {
            'dropout_rate': self._dropout_rate
        }


class PipelineModel:

    def __init__(self,
                 comm,
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss_fn: tf.keras.losses.Loss,
                 configs: Optional[Dict] = None):
        self._comm = comm
        self._size = comm.Get_size()
        self._rank = comm.Get_rank()
        self._first_rank = 0
        self._last_rank = self._size - 1
        self._next_rank = self._rank + 1 if self._rank + 1 <= self._last_rank else MPI.PROC_NULL
        self._prev_rank = self._rank - 1 if self._rank - 1 >= self._first_rank else MPI.PROC_NULL
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._model = self._get_model(configs or {})

    def _get_model(self, configs: Dict):
        if self.is_first_node():
            return Input(configs.get('input_hidden_dims', [128]))
        if self.is_last_node():
            return Head(configs.get('dropout_rate', 0.2))
        return Block(configs.get('block_hidden_dims', [64, 64]))

    def _get_microbatches(self, batch):
        assert batch_size % self._size == 0
        microbatch_size = batch_size // self._size
        microbatches = tf.data.Dataset \
            .from_tensor_slices(batch) \
            .batch(microbatch_size)
        return microbatches

    def _forward(self, batch):
        microbatches = self._get_microbatches(batch)
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
        gradients = []
        for i in range(self._size):
            if self.is_first_node():
                partial_error = self._comm.recv(source=self._next_rank)
                gradient = tapes[i].gradient(predictions[i],
                                             self._model.trainable_weights,
                                             output_gradients=partial_error)
            elif self.is_last_node():
                gradient = tapes[i].gradient(losses[i], self._model.trainable_weights)
                self._comm.send(gradient[0], dest=self._prev_rank)
            else:
                partial_error = self._comm.recv(source=self._next_rank)
                gradient = tapes[i].gradient(predictions[i],
                                             self._model.trainable_weights,
                                             output_gradients=partial_error)
                self._comm.send(gradient[0], dest=self._prev_rank)
            gradients.append(gradient)
        gradients = list(map(list, zip(*gradients)))
        gradients = [tf.reduce_mean(gradient, axis=0) for gradient in gradients]
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_weights))

    def train_on_batch(self, batch):
        predictions, tapes, losses = self._forward(batch)
        self._backward(predictions, tapes, losses)
        loss = tf.reduce_mean(losses)
        return loss

    def predict_on_batch(self, x):
        microbatches = self._get_microbatches(x)
        predictions = []
        for x in microbatches:
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
    n_train_batch, n_test_batch = math.floor(n_train / batch_size), math.ceil(n_test / batch_size)

    x_train = tf.data.Dataset \
        .from_tensor_slices((x_train, y_train)) \
        .batch(batch_size, drop_remainder=True) \
        .shuffle(n_train)
    x_test = tf.data.Dataset \
        .from_tensor_slices((x_test, y_test)) \
        .batch(batch_size)

    comm = MPI.COMM_WORLD
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    model = PipelineModel(comm, optimizer, loss_fn)

    if model.is_last_node():
        print('Training')
    for i in range(max_epoch):
        if model.is_last_node():
            print(f'Epoch {i + 1}/{max_epoch}')
            progbar = tf.keras.utils.Progbar(n_train_batch, stateful_metrics=['loss'])
        for batch in x_train:
            loss = model.train_on_batch(batch)
            if model.is_last_node():
                # noinspection PyUnboundLocalVariable
                progbar.add(1, values=[('loss', loss)])

    if model.is_last_node():
        print('Testing')
        progbar = tf.keras.utils.Progbar(n_test_batch, stateful_metrics=['acc'])
    for batch in x_test:
        x, y_true = batch
        y_pred = model.predict_on_batch(x)
        accuracy.update_state(y_true, y_pred)
        if model.is_last_node():
            # noinspection PyUnboundLocalVariable
            progbar.add(1, values=[('acc', accuracy.result())])


if __name__ == '__main__':
    main()
