# Execution Model

Preparation creates metadata and a graph. At inference, dependencies are checked in order, the loader identifies current-node tensors, the executor streams required ranges, computes, and releases node resources. The current engine operates synchronously per node; async storage and queue/server integration are separate layers.
