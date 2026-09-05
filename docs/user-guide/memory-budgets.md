# Memory Budgets

The budget limits runtime-managed reservations, not total process RSS. Managed categories include weights, activations, temporary memory, cache, and the future prefetch category. Before each reservation, current managed bytes plus requested bytes must fit. Python, PyTorch allocators, shared libraries, OS page cache, and GPU allocations are outside this guarantee.
