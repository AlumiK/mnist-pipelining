# MNIST Hybrid Parallelism

MNIST hybrid parallelism using pipelining and hierarchical Allreduce.

This version is implemented with TensorFlow 2 and MPI.

## Run

Run the code with

```
$ mpirun -np 32 python hybrid_parallelism.py
```

`32` is the number of processes.
