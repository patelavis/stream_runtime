# Profiling

Performance profiling is not the first milestone. Use Python profiling around `StreamingEngine.run`, record `bytes_read` and read count, and separately measure PyTorch compute. Do not interpret OS page-cache behavior as controlled managed memory.
