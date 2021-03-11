# MNIST Hybrid Parallelism

MNIST hybrid parallelism using pipelining and hierarchical Allreduce.

This version is implemented with TensorFlow 2 and MPI.

## Run

Run the code with

```
$ mpirun -np 16 python hybrid_parallelism.py
```

`16` is the number of processes.

## Model Structure

The script uses 4-way model parallelism and 4-way data parallelism by default.

![model structure](https://raw.githubusercontent.com/AlumiK/images/main/hybrid-parallelism/model_structure.svg)
