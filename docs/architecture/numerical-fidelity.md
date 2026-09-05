# Numerical Fidelity

The goal is original weights, dtype, architecture, and operator semantics. Bitwise identity is not promised. Tiling, BLAS kernels, CPU/GPU differences, and accumulation order can change low-order bits. Tests use `torch.allclose`/`torch.testing.assert_close` with documented tolerances where appropriate.
