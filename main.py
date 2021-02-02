import tensorflow as tf
import matplotlib.pyplot as plt

from mpi4py import MPI

minibatch_size = 64
n_microbatch = 4
assert minibatch_size % n_microbatch == 0
microbatch_size = minibatch_size // n_microbatch


class Tail(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, **kwargs):
        x = self.flatten(inputs)
        return x


class Block(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(128, activation='relu')

    def call(self, inputs, **kwargs):
        x = self.dense(inputs)
        return x


class Head(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.dropout(inputs)
        x = self.dense(x)
        return x


def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = tf.data.Dataset\
        .from_tensor_slices((x_train, y_train))\
        .batch(minibatch_size, drop_remainder=True)\
        .shuffle(len(x_train))
    x_test = tf.data.Dataset\
        .from_tensor_slices((x_test, y_test))\
        .batch(minibatch_size)

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    block = Block()
    block_tapes = []
    block_outputs = []
    if rank == 0:
        tail = Tail()
    if rank == size - 1:
        head = Head()
        head_tapes = []
        head_outputs = []
        losses = []

    for minibatch_idx, minibatch in enumerate(x_train):
        microbatches = tf.data.Dataset.from_tensor_slices(minibatch).batch(microbatch_size)
        for microbatch_idx, microbatch in enumerate(microbatches):
            x, y = microbatch
            if rank == 0:
                x = tail(x)
            else:
                x = comm.recv(source=rank - 1)
            with tf.GradientTape() as block_tape:
                x = block(x)
            block_outputs.append(x)
            block_tapes.append(block_tape)
            if rank != size - 1:
                comm.send(x, dest=rank + 1)
            else:
                with tf.GradientTape(persistent=True) as head_tape:
                    head_tape.watch(y)
                    head_tape.watch(x)
                    x = head(x)
                    loss = loss_fn(y, x)
                head_outputs.append(x)
                head_tapes.append(head_tape)
                losses.append(loss)

        block_grads = []
        if rank == size - 1:
            head_grads = []
        for _, _ in enumerate(microbatches):
            if rank == size - 1:
                loss, head_tape, head_output = losses.pop(), head_tapes.pop(), head_outputs.pop()
                print(f'{loss=}, {head_output=}')
                delta = head_tape.gradient(loss, head_output)
                print(f'{head.trainable_weights[0]=}, {delta=}')
                delta = tf.matmul(head.trainable_weights[0], delta, transpose_a=True)
                grad = head_tape.gradient(loss, head.trainable_weights)
                head_tape.reset()
                head_grads.append(grad)
            else:
                delta = comm.recv(source=rank + 1)
            block_output, block_tape = block_outputs.pop(), block_tapes.pop()
            grad = block_tape.gradient(tf.matmul(delta, block_output), block.trainable_weights)
            delta = tf.matmul(block.trainable_weights[0], delta, transpose_a=True)
            block_grads.append(grad)
            if rank != 0:
                comm.send(delta, dest=rank - 1)

        if rank == 0:
            print(f'{block_grads=}, {head_grads=}')
        block_grad = tf.reduce_mean(block_grads, axis=-1)
        if rank == size - 1:
            head_grad = tf.reduce_mean(head_grads, axis=-1)


if __name__ == '__main__':
    main()
