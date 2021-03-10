import math
import tensorflow as tf

from mpi4py import MPI
from nkmpi4py import NKMPI
from typing import Sequence, Tuple, Dict, Optional

batch_size = 64
max_epoch = 10


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


class CommEngine:
    class Entry:

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
        size = comm.Get_size()
        cart2d = comm.Create_cart(dims=[model_size, size // model_size], periods=[False, False], reorder=False)
        remain_dims = [True, False]
        model_comm = cart2d.Sub(remain_dims)
        remain_dims = [False, True]
        data_comm = cart2d.Sub(remain_dims)
        data_comm = NKMPI.Comm(data_comm, new_dims=data_dims)
        self.model = CommEngine.Entry(model_comm)
        self.data = CommEngine.Entry(data_comm)

    @property
    def is_coordinator(self) -> bool:
        return MPI.COMM_WORLD.Get_rank() == MPI.COMM_WORLD.Get_size() - 1

    @property
    def is_dispatcher(self) -> bool:
        return MPI.COMM_WORLD.Get_rank() == 0


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

    def _get_model(self, configs: Dict) -> tf.keras.Model:
        if self._ce.model.is_first:
            return Input(configs.get('input_hidden_dims', [128]))
        if self._ce.model.is_last:
            return Head(configs.get('dropout_rate', 0.2))
        return Block(configs.get('block_hidden_dims', [64, 64]))

    def _get_microbatches(self, batch):
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
        weights = self._model.get_weights()
        weights = self._ce.data.comm.bcast(weights)
        self._model.set_weights(weights)

    # noinspection PyUnboundLocalVariable
    def _forward(self, batch):
        if self._ce.model.is_first:
            microbatches = iter(self._get_microbatches(batch))
        predictions, tapes, losses = [], [], []
        for _ in range(self._ce.model.size):
            if self._ce.model.is_first:
                x, y_true = next(microbatches)
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

    def _backward(self, predictions, tapes, losses):
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

    def train_on_batch(self, batch):
        if not self._built:
            self._build(batch)
            self._built = True
        predictions, tapes, losses = self._forward(batch)
        self._backward(predictions, tapes, losses)
        if self._ce.model.is_last:
            loss = self._ce.data.comm.reduce(tf.reduce_mean(losses), op=MPI.SUM, root=self._ce.data.last_rank)
            if self._ce.data.is_last:
                loss /= self._ce.data.size
        else:
            loss = None
        return loss

    # noinspection PyUnboundLocalVariable
    def predict_on_batch(self, x):
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


def get_distributed_data(comm_engine: CommEngine) -> Tuple:
    sendobj = []
    if comm_engine.is_dispatcher:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        n_train, n_test = len(x_train), len(x_test)
        n_train_per_proc = n_train // comm_engine.data.size
        n_test_per_proc = n_test // comm_engine.data.size
        for i in range(comm_engine.data.size):
            data_proc = []
            x_train_proc = x_train[i * n_train_per_proc:(i + 1) * n_train_per_proc, :, :]
            y_train_proc = y_train[i * n_train_per_proc:(i + 1) * n_train_per_proc]

            if i == comm_engine.data.last_rank:
                x_test_proc = x_test[i * n_test_per_proc:, :, :]
                y_test_proc = y_test[i * n_test_per_proc:]
            else:
                x_test_proc = x_test[i * n_test_per_proc:(i + 1) * n_test_per_proc, :, :]
                y_test_proc = y_test[i * n_test_per_proc:(i + 1) * n_test_per_proc]
            data_proc.append(x_train_proc)
            data_proc.append(y_train_proc)
            data_proc.append(x_test_proc)
            data_proc.append(y_test_proc)
            sendobj.append(data_proc)
    x_train, y_train, x_test, y_test = comm_engine.data.comm.scatter(sendobj)
    return x_train, y_train, x_test, y_test


# noinspection PyUnboundLocalVariable
def main():
    comm_engine = CommEngine(MPI.COMM_WORLD, model_size=4, data_dims=[2, 2, 2])
    n_train_batch, n_test_batch = None, None
    if comm_engine.model.is_first:
        x_train, y_train, x_test, y_test = get_distributed_data(comm_engine)
        n_train, n_test = len(x_train), len(x_test)
        n_train_batch, n_test_batch = math.floor(n_train / batch_size), math.ceil(n_test / batch_size)

        x_train = tf.data.Dataset \
            .from_tensor_slices((x_train, y_train)) \
            .batch(batch_size, drop_remainder=True) \
            .shuffle(n_train)
        x_test = tf.data.Dataset \
            .from_tensor_slices((x_test, y_test)) \
            .batch(batch_size)
    n_train_batch, n_test_batch = comm_engine.model.comm.bcast((n_train_batch, n_test_batch))

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    model = HyParModel(comm_engine, optimizer, loss_fn)

    if comm_engine.is_coordinator:
        print('Training')
    for i in range(max_epoch):
        if comm_engine.is_coordinator:
            print(f'Epoch {i + 1}/{max_epoch}')
            progbar = tf.keras.utils.Progbar(n_train_batch, stateful_metrics=['loss'])
        if comm_engine.model.is_first:
            x_train_iter = iter(x_train)
        for _ in range(n_train_batch):
            batch = next(x_train_iter) if comm_engine.model.is_first else None
            loss = model.train_on_batch(batch)
            if comm_engine.is_coordinator:
                # noinspection PyUnboundLocalVariable
                progbar.add(1, values=[('loss', loss)])

    if comm_engine.is_coordinator:
        print('Testing')
        progbar = tf.keras.utils.Progbar(n_test_batch, stateful_metrics=['acc'])
    if comm_engine.model.is_first:
        x_test_iter = iter(x_test)
    for _ in range(n_test_batch):
        x, y_true = next(x_test_iter) if comm_engine.model.is_first else (None, None)
        y_pred = model.predict_on_batch(x)
        if comm_engine.model.is_first:
            comm_engine.model.comm.send(y_true, dest=comm_engine.model.last_rank)
        if comm_engine.model.is_last:
            y_true = comm_engine.model.comm.recv(source=comm_engine.model.first_rank)
            y_true = comm_engine.data.comm.gather(y_true, root=comm_engine.data.last_rank)
        if comm_engine.is_coordinator:
            accuracy.update_state(y_true, y_pred)
            # noinspection PyUnboundLocalVariable
            progbar.add(1, values=[('acc', accuracy.result())])


if __name__ == '__main__':
    main()
