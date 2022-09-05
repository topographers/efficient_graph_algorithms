# Gaussian Kernel Test

To run the script on the sample lemon dataset:

```sh
python scripts/experiments/gaussian_kernel_test.py 
```

This script file will evaluate the brute force method using memory_profiler and line_profiler to take record of the memory and time useage respectively. 

Evaluation result on sample data is as follows:

```sh
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    23    120.0 MiB    120.0 MiB           1   @profile
    24                                         def evaluate_brute_force_memory(adjacency_lists, kernel_function, graph_field):
    25    634.8 MiB    514.8 MiB           1       brute_force = BruteForce(adjacency_lists, kernel_function)
    26    635.1 MiB      0.3 MiB           1       Mx = brute_force.model_top_field(graph_field)
```

```sh
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    28                                           def evaluate_brute_force_time(adjacency_lists, kernel_function, graph_field):
    29         1   12538104.0 12538104.0     97.2      brute_force = BruteForce(adjacency_lists, kernel_function)
    30         1     363631.0 363631.0      2.8      Mx = brute_force.model_top_field(graph_field)
```
