# About

This repository demonstrates how implementing a custom Canny edge detector using both NumPy/SciPy and CuPy based backends, can allow one to take advantage of speedups offered by GPGPU processing while still being able to enjoy vectorized python ergonomics.

# Motivating Factors that Drove Implementation

- Use operations built into the libraries themselves as much as possible without reducing the overall codebase down to a trivial amount of code
- Have the GPU and CPU code paths be unified for fair comparison
- Vectorize as many operations as possible without sacrificing correctness and while maintaining reasonable amount of performance
    - This means very few or no Python native loops allowed in implementation
- Mimic the high level approach used by the reference scikit-image implementation as much as possible, with same vectorized logic used by scikit-image permissible in implementation while adhering to licensing standards
    - However, anything that corresponds to parts of the reference implementation that isn't trivially vectorizable will be instead implemented differently for this custom Canny implementation

# Non-Goals

- Implementing or claiming that the implementation is even remotely close to state-of-art implementations of Canny like those found in OpenCV library
- non CPU and GPU backend support

