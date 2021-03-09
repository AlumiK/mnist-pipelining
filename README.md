# MNIST Hybrid Parallelism

MNIST hybrid parallelism using pipelining and hierarchical Allreduce.

This version is implemented with TensorFlow 2 and MPI.

## Run

Run the code with

```
# 4 is the number of processes
mpirun -np 4 python main.py
```
