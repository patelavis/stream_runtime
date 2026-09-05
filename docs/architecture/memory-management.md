# Memory Management

`MemoryManager.reserve` checks the budget and returns a context-compatible `Reservation`; `release` decrements category counters. `MemoryStats` tracks current, peak, weights, activations, cache, and temporary bytes. This is logical managed memory, not RSS. The current demo's PyTorch output tensor is not fully integrated into category reservations; improving this accounting is planned.
