# Tensor Storage

`TensorStore` abstracts storage from operators. `SafeTensorStream` is the current backend. Future backends may implement a common range-read interface for sharded, mmap, compressed, or remote storage; remote storage is not used by the current offline server.
