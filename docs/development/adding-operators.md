# Adding Operators

Add an operator module with a narrow interface: input activation, declared node weights, manager/store access, selected strategy, output, and explicit release. Provide a direct reference implementation and a streaming/tiled implementation. Preserve dtype unless the operator explicitly documents a safe conversion. Include minimum-working-set failure behavior, range-read assertions, and `torch.testing.assert_close` tests. Never open safetensors files directly from an operator; use `TensorStore`.
