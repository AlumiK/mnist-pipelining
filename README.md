# Hybrid Parallelism

TensorFlow 2 hybrid parallelism using pipelining and hierarchical Allreduce.

MPI is used as the communication runtime.

## Run

Install dependencies by

```
$ pip install -r requirements.txt
```

Then run the script by

```
$ mpirun -n 16 python main.py
```

`16` is the number of processes.

## Distributed Model

The script uses 4-way model parallelism and 4-way data parallelism by default.

![model structure](https://raw.githubusercontent.com/AlumiK/images/main/hybrid-parallelism/model_structure.svg)

A brief flowchart of the distributed training process is shown below.

![flowchart](https://raw.githubusercontent.com/AlumiK/images/main/hybrid-parallelism/flowchart.svg)
