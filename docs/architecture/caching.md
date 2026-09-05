# Caching

`TensorChunkCache` is a bounded LRU implementation and its byte count can be integrated into managed memory. The server's `--cache-size` option is accepted, but active serving does not yet populate this cache. This is **experimental/planned**.
