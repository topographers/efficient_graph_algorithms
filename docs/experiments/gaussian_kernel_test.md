# Gaussian Kernel Test

To run the script on the sample lemon dataset:

```sh
python scripts/experiments/gaussian_kernel_test.py 
```

`--method`: current version only supports BruteForce matrix calculation in time O(n^2).

`--kernel`: this specifies the function f used in f(dist(i,j)). Current version only supports gaussian kernel. 

`--visualize_mesh`: if this is set as True, the dataset mesh be visualized with open3d.
