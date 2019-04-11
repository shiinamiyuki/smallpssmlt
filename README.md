#  smallpssmlt

## Robust Global Illumination in 99 lines of C++

A modification to smallpt by adding primary sample space metropolis light transport.

Usage: ./smallpssmlt [mutations per pixel]:

```
./smallpssmlt 1024
```

Build:

```
g++ -O3 smallpssmlt.cpp -fopenmp
```





