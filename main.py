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
        self.dummy = None

    def build(self, input_shape):
        self.dummy = self.add_weight(name='dummy', shape=input_shape, initializer='ones', trainable=True)

    def call(self, inputs, **kwargs):
        return grad_op(inputs, self.dummy)


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
        self.dense = tf.keras.layers.Dense(128, activation='relu')

    def call(self, inputs, **kwargs):
        x = self.grad_layer(inputs)
        x = self.dense(x)
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


class Trainer:

    def __init__(self,
                 comm,
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss_fn: tf.keras.losses.Loss):
        self._comm = comm
        self._size = comm.Get_size()
        self._rank = comm.Get_rank()
        self._first_rank = 0
        self._last_rank = self._size - 1
        self._next_rank = self._rank + 1 if self._rank + 1 <= self._last_rank else MPI.PROC_NULL
        self._prev_rank = self._rank - 1 if self._rank - 1 >= self._first_rank else MPI.PROC_NULL
        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn

    def _is_first_node(self) -> bool:
        return self._rank == self._first_rank

    def _is_last_node(self) -> bool:
        return self._rank == self._last_rank

    def _forward_pass(self, minibatch):
        assert minibatch_size % self._size == 0
        microbatch_size = minibatch_size // self._size
        microbatches = tf.data.Dataset \
            .from_tensor_slices(minibatch) \
            .batch(microbatch_size)
        predictions = []
        tapes = []
        losses = []
        for microbatch in microbatches:
            x, y = microbatch
            with tf.GradientTape() as tape:
                if self._is_first_node():
                    prediction = self._model(x)
                    self._comm.send(y, dest=self._)
                    self._comm.send(prediction, dest=self._next_rank)
                elif self._is_last_node():
                    y = self._comm.recv(source=0)
                    recvd = self._comm.recv(source=self._prev_rank)
                    prediction = self._model(recvd)
                    loss = self._loss_fn(y, prediction)
                    losses.append(loss)
                else:
                    recvd = self._comm.recv(source=self._prev_rank)
                    prediction = self._model(recvd)
                    self._comm.send(prediction, dest=self._next_rank)
            predictions.append(prediction)
            tapes.append(tape)
        return predictions, tapes, losses

    def _backward_pass(self, predictions, tapes, losses):
        grads = []
        for i in range(self._size):
            if self._is_first_node():
                errors = self._comm.recv(source=self._next_rank)
                grad = tapes[i].gradient(predictions[i],
                                         self._model.trainable_weights,
                                         output_gradients=errors)
            elif self._is_last_node():
                grad = tapes[i].gradient(losses[i], self._model.trainable_weights)
                self._comm.send(grad[0], dest=self._prev_rank)
            else:
                errors = self._comm.recv(source=self._next_rank)
                grad = tapes[i].gradient(predictions[i],
                                         self._model.trainable_weights,
                                         output_gradients=errors)
                self._comm.send(grad[0], dest=self._prev_rank)
            grads.append(grad)
        grads = list(map(list, zip(*grads)))
        grads = [tf.reduce_mean(grad, axis=0) for grad in grads]
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_weights))

    def _predict(self, minibatch):
        assert minibatch_size % self._size == 0
        microbatch_size = minibatch_size // self._size
        microbatches = tf.data.Dataset \
            .from_tensor_slices(minibatch) \
            .batch(microbatch_size)
        predictions = []
        for microbatch in microbatches:
            x, y = microbatch
            if self._is_first_node():
                prediction = self._model(x)
                self._comm.send(prediction, dest=self._next_rank)
            elif self._is_last_node():
                recvd = self._comm.recv(source=self._prev_rank)
                prediction = self._model(recvd)
            else:
                recvd = self._comm.recv(source=self._prev_rank)
                prediction = self._model(recvd)
                self._comm.send(prediction, dest=self._next_rank)
            predictions.extend(list(prediction.numpy()))
        return predictions

    def train_minibatch(self, minibatch):
        predictions, tapes, losses = self._forward_pass(minibatch)
        self._backward_pass(predictions, tapes, losses)
        return tf.reduce_mean(losses)

    def predict(self, minibatch):
        predictions = self._predict(minibatch)
        return predictions


def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print(y_train)
    n_train = len(x_train)
    n_minibatch = n_train // minibatch_size
    n_test = len(x_test)
    n_test_minibatch = n_test // minibatch_size
    x_train = tf.data.Dataset \
        .from_tensor_slices((x_train, y_train)) \
        .batch(minibatch_size, drop_remainder=True) \
        .shuffle(n_train)
    x_test = tf.data.Dataset \
        .from_tensor_slices((x_test, y_test)) \
        .batch(minibatch_size)

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    if rank == 0:
        model = Input()
    elif rank == size - 1:
        model = Head()
    else:
        model = Block()
    trainer = Trainer(comm, model, optimizer, loss_fn)

    if rank == size - 1:
        print('Training')
    for i in range(max_epoch):
        if rank == size - 1:
            print(f'Epoch {i + 1}/{max_epoch}')
            progbar = tf.keras.utils.Progbar(n_minibatch, stateful_metrics=['loss'])
        for idx, minibatch in enumerate(x_train):
            loss = trainer.train_minibatch(minibatch)
            if rank == size - 1:
                # noinspection PyUnboundLocalVariable
                progbar.add(1, values=[('loss', loss)])

    if rank == size - 1:
        print('Testing')
        progbar = tf.keras.utils.Progbar(n_test_minibatch, stateful_metrics=['acc'])
    for idx, minibatch in enumerate(x_test):
        _, y_true = minibatch
        y_pred = trainer.predict(minibatch)
        metric.update_state(y_true, y_pred)
        if rank == size - 1:
            # noinspection PyUnboundLocalVariable
            progbar.add(1, values=[('acc', metric.result())])


if __name__ == '__main__':
    main()
