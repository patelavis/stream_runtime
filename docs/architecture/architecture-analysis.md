# Architecture Analysis

`ArchitectureAdapter` is the extension point. `ArchitectureAdapter.analyze(reader)` returns nodes and tensor metadata. The current generic implementation is appropriate only for simple sequential weight/bias naming. A production Hugging Face adapter must inspect local config/module hierarchy and add confidence and limitations.
