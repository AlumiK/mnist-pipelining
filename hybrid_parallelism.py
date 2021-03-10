import math
import tensorflow as tf

from mpi4py import MPI
from nkmpi4py import NKMPI
from typing import Sequence, Tuple, Dict, Any, Optional

batch_size = 64
epochs = 10


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

    def __init__(self, hidden_dims: Sequence):
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


class CommEngine:
    class Comm:

        def __init__(self, comm):
            self.comm = comm
            self.size: int = comm.Get_size()
            self.rank: int = comm.Get_rank()
            self.first_rank: int = 0
            self.last_rank: int = self.size - 1
            self.prev_rank: int = self.rank - 1 if self.rank - 1 >= self.first_rank else MPI.PROC_NULL
            self.next_rank: int = self.rank + 1 if self.rank + 1 <= self.last_rank else MPI.PROC_NULL
            self.is_first: bool = self.rank == self.first_rank
            self.is_last: bool = self.rank == self.last_rank

    def __init__(self, comm, model_size: int, data_dims: Sequence):
        self.world = CommEngine.Comm(comm)
        assert self.world.size % model_size == 0
        cart_comm = comm.Create_cart(dims=[model_size, self.world.size // model_size],
                                     periods=[False, False],
                                     reorder=False)
        model_comm, data_comm = cart_comm.Sub([True, False]), cart_comm.Sub([False, True])
        data_comm = NKMPI.Comm(data_comm, new_dims=data_dims)
        self.model = CommEngine.Comm(model_comm)
        self.data = CommEngine.Comm(data_comm)


class HyParModel:

    def __init__(self,
                 comm_engine: CommEngine,
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss_fn: tf.keras.losses.Loss,
                 configs: Optional[Dict] = None):
        self._ce = comm_engine
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._model = self._get_model(configs or {})
        self._built = False

    def _get_model(self, configs: Dict[str, Any]) -> tf.keras.Model:
        if self._ce.model.is_first:
            return Input(configs.get('input_hidden_dims', [128]))
        if self._ce.model.is_last:
            return Head(configs.get('dropout_rate', 0.2))
        return Block(configs.get('block_hidden_dims', [64, 64]))

    def _get_microbatches(self, batch) -> tf.data.Dataset:
        assert batch_size % self._ce.model.size == 0
        microbatch_size = batch_size // self._ce.model.size
        microbatches = tf.data.Dataset \
            .from_tensor_slices(batch) \
            .batch(microbatch_size)
        return microbatches

    def _build(self, batch):
        if self._ce.model.is_first:
            microbatches = self._get_microbatches(batch)
            x = next(iter(microbatches))[0]
            y_pred = self._model(x)
            self._ce.model.comm.send(y_pred, dest=self._ce.model.next_rank)
        elif self._ce.model.is_last:
            recvd = self._ce.model.comm.recv(source=self._ce.model.prev_rank)
            self._model(recvd)
        else:
            recvd = self._ce.model.comm.recv(source=self._ce.model.prev_rank)
            y_pred = self._model(recvd)
            self._ce.model.comm.send(y_pred, dest=self._ce.model.next_rank)
        weights = None
        if self._ce.data.is_first:
            weights = self._model.get_weights()
        weights = self._ce.data.comm.bcast(weights)
        self._model.set_weights(weights)
        self._built = True

    # noinspection PyUnboundLocalVariable
    def _forward(self, batch) -> Tuple[Sequence, ...]:
        if self._ce.model.is_first:
            microbatches_iter = iter(self._get_microbatches(batch))
        predictions, tapes, losses = [], [], []
        for _ in range(self._ce.model.size):
            if self._ce.model.is_first:
                x, y_true = next(microbatches_iter)
            with tf.GradientTape() as tape:
                if self._ce.model.is_first:
                    y_pred = self._model(x)
                    self._ce.model.comm.send(y_true, dest=self._ce.model.last_rank)
                    self._ce.model.comm.send(y_pred, dest=self._ce.model.next_rank)
                elif self._ce.model.is_last:
                    y_true = self._ce.model.comm.recv(source=self._ce.model.first_rank)
                    recvd = self._ce.model.comm.recv(source=self._ce.model.prev_rank)
                    y_pred = self._model(recvd)
                    loss = self._loss_fn(y_true, y_pred)
                    losses.append(loss)
                else:
                    recvd = self._ce.model.comm.recv(source=self._ce.model.prev_rank)
                    y_pred = self._model(recvd)
                    self._ce.model.comm.send(y_pred, dest=self._ce.model.next_rank)
            predictions.append(y_pred)
            tapes.append(tape)
        return predictions, tapes, losses

    def _backward(self, predictions: Sequence, tapes: Sequence, losses: Sequence):
        gradients = []
        for i in range(self._ce.model.size):
            if self._ce.model.is_first:
                partial_error = self._ce.model.comm.recv(source=self._ce.model.next_rank)
                gradient = tapes[i].gradient(predictions[i],
                                             self._model.trainable_weights,
                                             output_gradients=partial_error)
            elif self._ce.model.is_last:
                gradient = tapes[i].gradient(losses[i], self._model.trainable_weights)
                self._ce.model.comm.send(gradient[0], dest=self._ce.model.prev_rank)
            else:
                partial_error = self._ce.model.comm.recv(source=self._ce.model.next_rank)
                gradient = tapes[i].gradient(predictions[i],
                                             self._model.trainable_weights,
                                             output_gradients=partial_error)
                self._ce.model.comm.send(gradient[0], dest=self._ce.model.prev_rank)
            gradients.append(gradient)
        gradients = list(map(list, zip(*gradients)))
        gradients = [tf.reduce_mean(gradient, axis=0) for gradient in gradients]
        gradients = [self._ce.data.comm.allreduce(gradient, op=MPI.SUM) for gradient in gradients]
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_weights))

    def train_on_batch(self, batch) -> Optional[float]:
        if not self._built:
            self._build(batch)
        predictions, tapes, losses = self._forward(batch)
        self._backward(predictions, tapes, losses)
        loss = None
        if self._ce.model.is_last:
            loss = self._ce.data.comm.reduce(tf.reduce_mean(losses), op=MPI.SUM, root=self._ce.data.last_rank)
            if self._ce.world.is_last:
                loss /= self._ce.data.size
        return loss

    # noinspection PyUnboundLocalVariable
    def predict_on_batch(self, x) -> Sequence:
        if not self._built:
            raise RuntimeError('The model has not been built yet.')
        n_microbatch = None
        if self._ce.model.is_first:
            microbatches = self._get_microbatches(x)
            microbatches_iter = iter(microbatches)
            n_microbatch = microbatches.cardinality().numpy()
        n_microbatch = self._ce.model.comm.bcast(n_microbatch)
        predictions = []
        for _ in range(n_microbatch):
            if self._ce.model.is_first:
                x = next(microbatches_iter)
                y_pred = self._model(x)
                self._ce.model.comm.send(y_pred, dest=self._ce.model.next_rank)
            elif self._ce.model.is_last:
                recvd = self._ce.model.comm.recv(source=self._ce.model.prev_rank)
                y_pred = self._model(recvd)
            else:
                recvd = self._ce.model.comm.recv(source=self._ce.model.prev_rank)
                y_pred = self._model(recvd)
                self._ce.model.comm.send(y_pred, dest=self._ce.model.next_rank)
            predictions.extend(y_pred.numpy())
        if self._ce.model.is_last:
            predictions = self._ce.data.comm.gather(predictions, root=self._ce.data.last_rank)
        else:
            predictions = []
        return predictions


def dispatch_data(comm_engine: CommEngine) -> Tuple[Sequence, ...]:
    sendobj = []
    if comm_engine.world.is_first:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        n_train, n_test = len(x_train), len(x_test)
        n_train_proc, n_test_proc = n_train // comm_engine.data.size, n_test // comm_engine.data.size
        for i in range(comm_engine.data.size):
            x_train_proc = x_train[i * n_train_proc:(i + 1) * n_train_proc, :, :]
            y_train_proc = y_train[i * n_train_proc:(i + 1) * n_train_proc]
            if i == comm_engine.data.last_rank:
                x_test_proc = x_test[i * n_test_proc:, :, :]
                y_test_proc = y_test[i * n_test_proc:]
            else:
                x_test_proc = x_test[i * n_test_proc:(i + 1) * n_test_proc, :, :]
                y_test_proc = y_test[i * n_test_proc:(i + 1) * n_test_proc]
            sendobj.append([x_train_proc, y_train_proc, x_test_proc, y_test_proc])
    x_train, y_train, x_test, y_test = comm_engine.data.comm.scatter(sendobj)
    return x_train, y_train, x_test, y_test


# noinspection PyUnboundLocalVariable
def main():
    ce = CommEngine(MPI.COMM_WORLD, model_size=4, data_dims=[2, 2, 2])

    n_train_batch, n_test_batch = None, None
    if ce.model.is_first:
        x_train, y_train, x_test, y_test = dispatch_data(ce)
        n_train, n_test = len(x_train), len(x_test)
        n_train_batch, n_test_batch = math.floor(n_train / batch_size), math.ceil(n_test / batch_size)
        x_train = tf.data.Dataset \
            .from_tensor_slices((x_train, y_train)) \
            .batch(batch_size, drop_remainder=True) \
            .shuffle(n_train)
        x_test = tf.data.Dataset \
            .from_tensor_slices((x_test, y_test)) \
            .batch(batch_size)
    n_train_batch, n_test_batch = ce.model.comm.bcast((n_train_batch, n_test_batch))

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    model = HyParModel(comm_engine=ce, optimizer=optimizer, loss_fn=loss_fn)

    if ce.world.is_last:
        print('Training')
    for i in range(epochs):
        if ce.world.is_last:
            print(f'Epoch {i + 1}/{epochs}')
            progbar = tf.keras.utils.Progbar(n_train_batch, stateful_metrics=['loss'])
        if ce.model.is_first:
            x_train_iter = iter(x_train)
        for _ in range(n_train_batch):
            batch = next(x_train_iter) if ce.model.is_first else None
            loss = model.train_on_batch(batch)
            if ce.world.is_last:
                progbar.add(1, values=[('loss', loss)])

    if ce.world.is_last:
        print('Testing')
        progbar = tf.keras.utils.Progbar(n_test_batch, stateful_metrics=['acc'])
    if ce.model.is_first:
        x_test_iter = iter(x_test)
    for _ in range(n_test_batch):
        x, y_true = next(x_test_iter) if ce.model.is_first else (None, None)
        y_pred = model.predict_on_batch(x)
        if ce.model.is_first:
            ce.model.comm.send(y_true, dest=ce.model.last_rank)
        if ce.model.is_last:
            y_true = ce.model.comm.recv(source=ce.model.first_rank)
            y_true = ce.data.comm.gather(y_true, root=ce.data.last_rank)
        if ce.world.is_last:
            accuracy.update_state(y_true, y_pred)
            progbar.add(1, values=[('acc', accuracy.result())])


if __name__ == '__main__':
    main()
