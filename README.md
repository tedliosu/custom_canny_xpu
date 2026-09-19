# About

This repository demonstrates how implementing a custom Canny edge detector using both NumPy/SciPy and CuPy based backends, can allow one to take advantage of speedups offered by GPGPU processing while still being able to enjoy vectorized python ergonomics.

# Motivating Factors that Drove Implementation

- Balance between amount of code written from first principles vs reusing existing primitives
- Have the GPU and CPU code paths be unified for fair comparison
- Version ALPHA only: vectorize as many operations as possible without sacrificing correctness, to map decently to GPU execution naively
    - This means very few or no "top level" Python loops allowed in implementation
- Mimic the high level approach used by the reference scikit-image implementation as much as possible, with same vectorized logic used by scikit-image permissible in implementation while adhering to licensing standards

# Non-Goals

- Implementing or claiming that the implementation is even remotely close to state-of-art implementations of Canny like those found in OpenCV library
- non CPU and GPU backend support

# LLM Assistance Usage Disclaimer

- All code and documentation in this repository were drafted with assistance from ChatGPT models publicly available circa 2025 - 2026. Architecture, support boundaries, semantics, interfaces, invariants, acceptance or rejection decisions, and final integration approval remain human-owned.
- Human review is contract-first and risk-weighted. It includes deeper inspection of critical behavior and selective implementation review supported by tests and behavioral summaries; it does not claim exhaustive line-by-line inspection of every agent-assisted change.
- This disclosure provides transparency about the development process; it is not by itself proof of originality, complete provenance, or license compliance.
- Reports of suspected similarity to third-party material or licensing concerns are welcome so that affected code or documentation can be reviewed and, when appropriate, replaced.

