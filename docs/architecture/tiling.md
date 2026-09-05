# Tiling

`StreamingLinear` computes `X @ W.T` one output-row tile at a time. Tile rows are derived from available planner bytes and weight-row size. The logical node remains a single node. Temporary conversion and output accounting are areas for future tightening. Tiled accumulation may differ slightly because floating-point order can change.
