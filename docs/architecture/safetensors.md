# Safetensors

The reader reads the 8-byte little-endian header length, then only the JSON header. Each descriptor records dtype, shape, and data offsets. `tensor.read(offset,length)` uses a seek/read range; `iter_chunks` yields ranges. `read_log`, `bytes_read`, and `reads` support streaming tests. No complete tensor or file is loaded by metadata inspection.
