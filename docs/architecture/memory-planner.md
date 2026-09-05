# Memory Planner

`MemoryPlanner` estimates each node from declared weight bytes plus a safety reserve. Nodes fit directly when the estimate is within the budget; otherwise linear execution chooses an output-row tile. It is deterministic and conservative, but activation-aware planning is still experimental.
