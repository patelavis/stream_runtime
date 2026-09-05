
 You are a senior systems engineer, ML runtime engineer, Python/PyTorch developer, and storage/memory-management expert.

 I want you to design and IMPLEMENT a working prototype of a **disk-streaming AI model inference runtime**.

 Do not merely explain the concept. **Write the actual complete project code.**

---

 # 1\. PROJECT GOAL

 The long-term goal is:

 > Run very large AI/ML models locally even when the model is much larger than available RAM and/or VRAM.

 The complete model should remain on local HDD/SSD.

 Only the minimum data required for the current computation should be loaded into RAM/VRAM.

 The runtime should execute the model sequentially and release memory before loading the next required data.

 Example:

```
              LOCAL HDD / SSD
             ┌─────────────────┐
             │   Huge Model    │
             │                 │
             │ 10 GB           │
             │ 100 GB          │
             │ 1 TB            │
             └────────┬────────┘
                      │
                stream required
                     data
                      │
                      ▼
             ┌─────────────────┐
             │      RAM        │
             │                 │
             │ User-defined    │
             │ maximum budget  │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ CPU / GPU       │
             │ computation     │
             └────────┬────────┘
                      │
                      ▼
                 next node
```

 The model size should NOT determine the required RAM size.

 The user's configured RAM budget determines the maximum managed working memory.

---

 # 2\. ABSOLUTE PRIORITY

 The development priorities are:

```
1. Correctness
2. Memory safety / memory budgeting
3. Ability to execute models larger than RAM
4. Architecture-aware streaming
5. Operator tiling
6. Activation management
7. GPU/VRAM support
8. Performance optimization
```

 For the first version:

 > **SPEED DOES NOT MATTER.**

 If a model takes an extremely long time because of HDD I/O, that is acceptable.

 First prove that the architecture works.

---

 # 3\. USER-DEFINED RAM BUDGET

 The user must specify the maximum RAM budget.

 Example:

```
python -m stream_runtime run model.safetensors --ram-budget 512M
```

 or:

```
python -m stream_runtime run model.safetensors --ram-budget 1G
```

 or:

```
python -m stream_runtime run model.safetensors --ram-budget 1048576
```

 The runtime must interpret this as a hard maximum for **managed model/runtime memory**.

 Supported sizes:

```
K
M
G
KB
MB
GB
```

 and raw bytes.

 Examples:

```
512K
1M
256M
1G
2GB
```

---

 # 4\. VERY IMPORTANT MEMORY RULE

 The runtime must NEVER intentionally load the complete model into RAM.

 For example:

```
Model = 10 GB
RAM budget = 512 MB
```

 This must remain possible:

```
Disk:
10 GB model

RAM:
≤ 512 MB managed working set
```

 The runtime should stream the model.

---

 # 5\. IMPORTANT DISTINCTION: RAM BUDGET VS PHYSICAL RAM

 The user-defined budget is a **runtime managed-memory limit**.

 Do not claim that Python's entire process RSS will be exactly equal to this value.

 Python itself, the operating system, PyTorch, CUDA, shared libraries, filesystem cache, etc. consume memory.

 The runtime must clearly document:

```
User RAM budget
=
maximum memory available to model/runtime-managed buffers
```

 It must NOT claim:

```
whole operating system process can never exceed this value
```

 unless such behavior is explicitly implemented using OS-level mechanisms.

---

 # 6\. MODEL STORAGE

 Initial supported model format:

```
.safetensors
```

 The runtime must support a model much larger than the RAM budget.

 Example:

```
model.safetensors = 5 GB
RAM budget = 512 MB
```

 The complete safetensors file must stay on disk.

---

 # 7\. DO NOT USE FULL safetensors LOADING

 DO NOT do:

```
from safetensors.torch import load_file

state = load_file("model.safetensors")
```

 for the complete model.

 That defeats the purpose.

 The runtime must implement a streaming safetensors reader that:

 1. Reads only the safetensors header.
2. Reads tensor metadata.
3. Determines:
   - tensor name
   - dtype
   - shape
   - byte offsets
   - byte length
4. Reads only requested ranges from the file.
5. Does not load all tensors.

 Use appropriate file APIs such as:

```
os.pread()
```

 where appropriate.

---

 # 8\. SAFETENSORS READER

 Implement:

```
class SafeTensorStream:
    ...
```

 Example:

```
reader = SafeTensorStream("model.safetensors")

print(reader.tensor_names())

tensor = reader.get_tensor("model.layers.0.self_attn.q_proj.weight")

chunk = tensor.read_chunk(
    offset=0,
    size=256 * 1024,
)
```

 Also support:

```
for chunk in tensor.iter_chunks(256 * 1024):
    ...
```

 The tensor object should contain metadata without containing the complete tensor data.

---

 # 9\. ARCHITECTURE-AWARE CHUNKING — CRITICAL

 DO NOT split the model into arbitrary byte ranges.

 Incorrect:

```
0–400 KB
400–800 KB
800–1200 KB
```

 These byte ranges do not necessarily correspond to computational units.

 Instead:

 > **The primary logical chunk boundary must be a valid computational architecture boundary.**

 Example:

```
Input
 ↓
Embedding
 ↓
Transformer Block 0
 ↓
Transformer Block 1
 ↓
Transformer Block 2
 ↓
Final LayerNorm
 ↓
LM Head
 ↓
Output
```

 Logical chunks should be:

```
Node 0 = Embedding
Node 1 = Transformer Block 0
Node 2 = Transformer Block 1
Node 3 = Transformer Block 2
Node 4 = Final LayerNorm
Node 5 = LM Head
```

 NOT:

```
random 400 KB byte chunks
```

---

 # 10\. CHUNK MUST END AT A NODE/COMPUTATIONAL BOUNDARY

 The default policy is:

```
one logical chunk = one complete computational node/block
```

 A node should contain the weights and computation necessary for that architectural unit.

 Example:

```
TransformerBlock 0
 ├── LayerNorm
 ├── Attention
 ├── Q projection
 ├── K projection
 ├── V projection
 ├── output projection
 ├── residual
 ├── LayerNorm
 ├── MLP
 └── residual
```

 This entire block is one logical node.

 The runtime must NOT arbitrarily split the block between two normal architectural chunks.

---

 # 11\. IMPORTANT: LOGICAL NODE VS INTERNAL TILING

 These are different concepts.

 Logical architecture:

```
Block 0
 ↓
Block 1
 ↓
Block 2
```

 If Block 1 is too large for the RAM budget:

 DO NOT redefine the architecture as:

```
Half Block 1
 ↓
Half Block 1
```

 Instead:

```
Block 1
   │
   ├── internal tile 0
   ├── internal tile 1
   ├── internal tile 2
   └── internal tile 3
```

 The node remains:

```
TransformerBlock 1
```

 but its implementation uses memory-aware tiling.

 This distinction is mandatory.

---

 # 12\. HUGGING FACE ARCHITECTURE ANALYSIS

 Hugging Face models should be the primary source for determining architecture and node boundaries.

 Use available:

```
Hugging Face model config
Hugging Face model architecture
PyTorch module hierarchy
model.named_modules()
model.named_parameters()
state_dict tensor names
model class
```

 during the preparation/analysis stage.

 The runtime should identify:

```
architecture
model type
modules
layers
blocks
submodules
execution order
tensor → module relationships
```

 Do NOT assume every model is a Transformer.

 The system must be extensible.

---

 # 13\. SEPARATE PREPARATION FROM INFERENCE

 Create two stages.

 ## Preparation

```
Hugging Face model
       ↓
Architecture analyzer
       ↓
identify nodes
       ↓
identify execution order
       ↓
map tensors to nodes
       ↓
read safetensors metadata
       ↓
create manifest
```

 ## Inference

```
load manifest
       ↓
load current node's required data
       ↓
execute
       ↓
release data
       ↓
next node
```

 Inference must NOT instantiate the entire original Hugging Face model.

---

 # 14\. ARCHITECTURE ADAPTER SYSTEM

 Implement:

```
class ArchitectureAdapter:
    def can_handle(self, config):
        ...

    def analyze(self, model):
        ...

    def build_graph(self):
        ...

    def estimate_memory(self, node):
        ...
```

 Create a registry:

```
ARCHITECTURE_ADAPTERS = [...]
```

 Initially support a generic/simple architecture.

 The system should be designed so we can later add:

```
Llama
Mistral
Qwen
Gemma
Phi
GPT-2
T5
BERT
Diffusion models
CNNs
etc.
```

 Do not hard-code the whole runtime around one model.

---

 # 15\. MODEL MANIFEST

 Create a manifest describing the prepared model.

 It should include:

```
format version
architecture
model type
execution order
nodes
node type
node dependencies
tensor names
tensor offsets
tensor sizes
dtype
shape
estimated memory
```

 Example:

```
{
  "format_version": 1,
  "architecture": "example_transformer",
  "nodes": [
    {
      "id": 0,
      "name": "embedding",
      "type": "Embedding",
      "inputs": ["input"],
      "outputs": ["hidden"],
      "weights": [
        "model.embed_tokens.weight"
      ]
    },
    {
      "id": 1,
      "name": "layers.0",
      "type": "TransformerBlock",
      "inputs": ["hidden"],
      "outputs": ["hidden"],
      "weights": [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight"
      ]
    }
  ]
}
```

 Improve this format where appropriate.

---

 # 16\. AUTOMATIC MEMORY PLANNER

 The user should specify only:

```
RAM budget
```

 The runtime should automatically calculate how much memory can be used for:

```
weights
activations
temporary buffers
cache
workspace
```

 Example:

```
RAM budget = 512 MB

Runtime reservation = 20 MB
Activation = 100 MB
Cache = 50 MB
Temporary workspace = 50 MB

Available for current weights:
292 MB
```

 The planner must account for all managed allocations.

---

 # 17\. NODE MEMORY ESTIMATION

 For each node calculate approximately:

```
weights
+
input activation
+
output activation
+
temporary workspace
+
cache
```

 Example:

```
Node:
Transformer Block 7

Weights:
300 MB

Input:
40 MB

Output:
40 MB

Temporary:
80 MB

Total:
460 MB
```

 With:

```
RAM budget = 512 MB
```

 the node can potentially execute directly.

 If:

```
Total = 900 MB
RAM = 512 MB
```

 the runtime must select an internal tiling strategy.

---

 # 18\. AUTOMATIC RAM-BASED CHUNK/WORKING-SET SELECTION

 Do NOT require the user to specify:

```
chunk_size=400KB
```

 The user specifies:

```
--ram-budget 512M
```

 The runtime determines the maximum working set automatically.

 Conceptually:

```
RAM budget
    ↓
memory planner
    ↓
available memory
    ↓
node requirements
    ↓
select execution strategy
```

 For example:

```
Node A = 100 MB
Node B = 400 MB
Node C = 2 GB

RAM = 512 MB

A → direct
B → direct
C → tiled
```

---

 # 19\. INTERNAL OPERATOR TILING

 Large operators must eventually be tiled.

 Example:

```
Y = X @ W
```

 If:

```
W = 1 GB
RAM = 256 MB
```

 do not load W completely.

 Instead:

```
W tile 0
 ↓
partial result

W tile 1
 ↓
partial result

W tile 2
 ↓
partial result
```

 Conceptually:

```
result = initialize_result()

for weight_tile in stream_weights():
    partial = compute(input, weight_tile)
    accumulate(result, partial)
```

 Make sure:

```
weight_tile
+
input
+
partial result
+
temporary memory
```

 fit within the available budget.

---

 # 20\. ACTIVATION MEMORY

 Weights are not the only memory problem.

 Activations can be huge.

 Implement:

```
class ActivationStore:
    ...
```

 Initially support RAM-backed activations.

 Design it so we can later add:

```
disk-backed activations
```

 If an activation is larger than the RAM budget, the architecture should eventually support activation tiling/streaming.

---

 # 21\. MEMORY MANAGER

 Implement:

```
class MemoryManager:
    ...
```

 Example:

```
memory = MemoryManager(
    budget_bytes=512 * 1024 * 1024
)
```

 Track:

```
current usage
peak usage
available memory
weight memory
activation memory
cache memory
temporary memory
```

 Every managed allocation must go through this system.

---

 # 22\. HARD MEMORY LIMIT

 If:

```
current = 400 MB
request = 200 MB
budget = 512 MB
```

 the allocation must fail or trigger eviction/replanning.

 It must NOT silently allocate 600 MB.

 Example:

```
raise MemoryBudgetExceeded(...)
```

 Do not rely solely on Python's garbage collector.

 Explicitly manage buffers.

---

 # 23\. CACHE

 Implement a bounded LRU cache.

 Example:

```
cache = TensorChunkCache(
    max_bytes=64 * 1024 * 1024
)
```

 Cache memory counts toward the RAM budget.

 The cache must never cause:

```
RAM budget = 512 MB
managed memory = 800 MB
```

---

 # 24\. CPU FIRST

 Version 1 should run on CPU.

 Architecture:

```
Disk
 ↓
RAM streaming buffer
 ↓
CPU
 ↓
result
```

 Do not require CUDA for the first milestone.

---

 # 25\. GPU / VRAM SUPPORT

 Design the architecture for future GPU support.

 Create something like:

```
class DeviceMemoryManager:
    ...
```

 and:

```
CPUMemoryManager
GPUMemoryManager
```

 Eventually support:

```
--ram-budget 512M
--vram-budget 256M
```

 Architecture:

```
Disk
 ↓
RAM
 ↓
VRAM
 ↓
GPU computation
 ↓
RAM
 ↓
next node
```

 Do not load the entire model into VRAM.

---

 # 26\. DISK-BACKED ACTIVATIONS

 Eventually support:

```
weights → disk
activations → disk
```

 This is necessary when activations are larger than RAM.

 Design an interface now:

```
class ActivationStore:
    def write(...)
    def read(...)
    def release(...)
```

 Then implement:

```
RamActivationStore
DiskActivationStore
```

 later.

---

 # 27\. STORAGE MANAGER

 Create:

```
class TensorStore:
    ...
```

 It should abstract where tensors come from:

```
safetensors file
sharded safetensors
future storage formats
```

 Example:

```
tensor_store.read(
    tensor_name,
    offset,
    length
)
```

 This lets the inference engine remain independent of the storage format.

---

 # 28\. MODEL PREPARATION COMMAND

 Implement:

```
python -m stream_runtime prepare \
    model_directory \
    --output prepared_model
```

 Preparation should:

```
inspect Hugging Face model
        ↓
identify architecture
        ↓
identify nodes
        ↓
map tensors to nodes
        ↓
read safetensors metadata
        ↓
generate manifest
```

 Do not arbitrarily split tensors merely to make equal-size files.

 Prefer architecture-aware grouping.

---

 # 29\. MODEL INSPECTION COMMAND

 Implement:

```
python -m stream_runtime inspect model.safetensors
```

 Display:

```
Model:
File size:

Architecture:
Tensor count:

Tensor:
    name
    dtype
    shape
    bytes
    offset

Total parameter bytes:
Largest tensor:
```

 Do not load complete tensors.

---

 # 30\. MEMORY PLAN COMMAND

 Implement:

```
python -m stream_runtime plan \
    prepared_model \
    --ram-budget 512M
```

 Output:

```
========== MEMORY PLAN ==========

RAM budget: 512 MB

Node 0: Embedding
  strategy: direct
  working set: 120 MB

Node 1: TransformerBlock
  strategy: direct
  working set: 450 MB

Node 2: TransformerBlock
  strategy: tiled
  tile size: 64 MB
  working set: 500 MB

Peak planned memory: 508 MB

==================================
```

 The plan must be deterministic.

---

 # 31\. RUN COMMAND

 Implement:

```
python -m stream_runtime run \
    prepared_model \
    --ram-budget 512M
```

 The runtime should:

```
load manifest
 ↓
initialize memory manager
 ↓
initialize storage
 ↓
execute node 0
 ↓
release node 0
 ↓
execute node 1
 ↓
release node 1
 ↓
...
```

---

 # 32\. MEMORY REPORT

 At runtime display something like:

```
========== STREAM RUNTIME ==========

RAM budget:          512 MB
Current managed:     320 MB
Peak managed:        488 MB
Available:           24 MB

Weights:             280 MB
Activations:          80 MB
Cache:                20 MB
Temporary:            108 MB

Disk model:           5 GB
Bytes read:           18 GB
Chunks processed:     1200

Status: SUCCESS

=====================================
```

 Track:

```
current memory
peak memory
total disk bytes read
number of reads
number of chunks
weight memory
activation memory
cache memory
temporary memory
```

---

 # 33\. TEST MODEL

 Create a script:

```
python examples/create_test_model.py
```

 It should create a small `.safetensors` test model.

 For example:

```
Input
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
Output
```

 Make the test model configurable so we can make it larger than the RAM budget.

---

 # 34\. STREAMING CORRECTNESS TEST

 Run the same model two ways:

```
normal in-memory execution
```

 and:

```
streaming execution
```

 Compare outputs:

```
torch.allclose(...)
```

 with an appropriate tolerance.

 The streaming implementation must produce numerically equivalent results within expected floating-point tolerance.

---

 # 35\. MEMORY TESTS

 Test:

```
budget = 1 MB
request = 512 KB
PASS
```

 and:

```
current = 800 KB
request = 300 KB
FAIL
```

 Test:

```
model = 5 MB
budget = 1 MB
```

 and prove that the runtime does not intentionally load all 5 MB.

---

 # 36\. IMPORTANT BUGS TO TEST

 Review the code specifically for:

```
accidental full-model loading
accidental full-tensor loading
tensor duplication
cache exceeding budget
activation exceeding budget
temporary buffers exceeding budget
incorrect safetensors offsets
incorrect dtype
incorrect shape
memory accounting errors
buffer lifetime errors
file descriptor leaks
chunk boundary errors
incorrect tensor-to-node mapping
```

---

 # 37\. PERFORMANCE IS NOT THE FIRST GOAL

 Do not prematurely optimize.

 This is acceptable:

```
Model = 5 GB
RAM = 512 MB
Runtime = extremely slow
```

 if:

```
the model executes correctly
```

 The first objective is:

 > Prove that a model can execute while remaining larger than the configured memory budget.

 Later we will optimize:

```
SSD/NVMe
prefetching
parallel reads
memory mapping
I/O overlap
caching
GPU
CUDA
quantization
compression
```

---

 # 38\. PREFETCHING

 Do not implement aggressive prefetching initially.

 Later support:

```
compute node N
while reading node N+1
```

 but all prefetched data must count against the RAM budget.

 Never allow prefetching to violate the memory constraint.

---

 # 39\. MEMORY-MAPPED FILES

 The design may later support:

```
mmap
```

 but be careful:

 OS page cache is not equivalent to controlled application memory.

 Do not claim `mmap` automatically solves the memory budget.

 The runtime must still have explicit memory accounting and controlled working sets.

---

 # 40\. GENERIC FALLBACK

 If the runtime encounters an architecture it does not have a dedicated adapter for:

 1. Try generic PyTorch module inspection.
2. Determine module hierarchy.
3. Determine parameter relationships.
4. Build a best-effort graph.
5. Clearly report confidence/limitations.
6. Never silently claim universal support.

---

 # 41\. FAILURE HANDLING

 If a node cannot fit:

 Example:

```
Node:
TransformerBlock 12

RAM budget:
512 MB

Estimated minimum working memory:
1.8 GB

Weight tiling:
available

Activation tiling:
available

Current operator implementation:
insufficient

Result:
CANNOT EXECUTE UNDER CURRENT IMPLEMENTATION
```

 Do not silently exceed the memory budget.

 Do not secretly load the whole node.

---

 # 42\. VERY IMPORTANT: "ANY SIZE" INTERPRETATION

 The long-term objective is to support models that are much larger than available RAM.

 Do NOT claim mathematically that literally every possible model can run with arbitrarily tiny memory.

 The correct engineering statement is:

 > Any model whose computational graph and operators can be decomposed into working sets that fit within the configured memory budget can theoretically be executed using storage-backed streaming.

 If an operation has an unavoidable working set larger than the budget and there is no supported tiling strategy, report the limitation.

---

 # 43\. PROJECT STRUCTURE

 Create:

```
stream_runtime/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── build_zip.py
│
├── stream_runtime/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── exceptions.py
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── safetensors_reader.py
│   │   ├── tensor.py
│   │   ├── tensor_store.py
│   │   └── cache.py
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   └── buffer.py
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── node.py
│   │   └── graph.py
│   │
│   ├── architecture/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   └── generic.py
│   │
│   ├── operators/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── linear.py
│   │   └── activation.py
│   │
│   ├── planner/
│   │   ├── __init__.py
│   │   └── memory_planner.py
│   │
│   ├── activation/
│   │   ├── __init__.py
│   │   └── store.py
│   │
│   └── runtime/
│       ├── __init__.py
│       └── engine.py
│
├── tests/
│   ├── test_memory.py
│   ├── test_safetensors_reader.py
│   ├── test_cache.py
│   ├── test_linear_streaming.py
│   ├── test_runtime.py
│   └── test_memory_budget.py
│
└── examples/
    ├── create_test_model.py
    ├── inspect_model.py
    └── run_streaming.py
```

 You may improve the structure if necessary.

---

 # 44\. DEPENDENCIES

 Keep dependencies minimal.

 Prefer:

```
Python
PyTorch
safetensors
numpy
pytest
```

 Do not introduce unnecessary frameworks.

---

 # 45\. ZIP FILE

 Create:

```
build_zip.py
```

 Running:

```
python build_zip.py
```

 must generate:

```
stream_runtime.zip
```

 containing the complete project.

---

 # 46\. README

 The README must explain:

 1. Project purpose.
2. Architecture.
3. Disk streaming.
4. RAM budget.
5. Difference between storage and working memory.
6. Safetensors streaming.
7. Architecture-aware node boundaries.
8. Hugging Face architecture analysis.
9. Automatic memory planning.
10. Internal operator tiling.
11. Activation memory.
12. Cache.
13. CPU execution.
14. Future GPU/VRAM support.
15. Limitations.
16. Installation.
17. Test model generation.
18. Inspection.
19. Memory planning.
20. Running inference.
21. Creating the ZIP.
22. Future roadmap.

---

 # 47\. FUTURE ROADMAP

 Design the code so the project can evolve through:

```
PHASE 1
✓ safetensors streaming
✓ disk-backed model
✓ hard managed RAM budget
✓ CPU execution
✓ architecture-aware nodes
✓ basic memory planner
✓ basic operator tiling
✓ correctness tests

PHASE 2
✓ disk-backed activations
✓ better cache
✓ memory mapping
✓ asynchronous I/O
✓ prefetching
✓ automatic memory optimization

PHASE 3
✓ CUDA
✓ VRAM budget
✓ RAM → VRAM streaming
✓ GPU tiled operators

PHASE 4
✓ Transformer support
✓ Llama
✓ Mistral
✓ Qwen
✓ Gemma
✓ Phi
✓ attention tiling
✓ KV-cache management

PHASE 5
✓ quantization
✓ INT8
✓ INT4
✓ compressed weights

PHASE 6
✓ performance optimization
✓ SSD/NVMe optimization
✓ parallel I/O
✓ compute/I/O overlap
✓ automatic execution scheduling
```

---

 # 48\. FINAL SUCCESS CRITERION

 The first major milestone is:

```
Model:
    5 MB

RAM budget:
    1 MB

Model stored:
    HDD/SSD

                  ↓

Architecture analyzed

                  ↓

Valid computational nodes identified

                  ↓

Memory plan generated

                  ↓

Node 0 streamed

                  ↓

Node 0 executed

                  ↓

Node 0 memory released

                  ↓

Node 1 streamed

                  ↓

Node 1 executed

                  ↓

...

                  ↓

Output
```

 At no point should the runtime intentionally load the complete 5 MB model into RAM.

 Then test:

```
Model = 100 MB
RAM = 10 MB
```

 Then eventually:

```
Model = 10 GB
RAM = 512 MB
```

 The same architecture should apply.

---

 # 49\. DO NOT JUST GIVE PSEUDOCODE

 This is critical.

 I want:

```
actual Python source code
actual classes
actual functions
actual tests
actual CLI
actual safetensors parsing
actual memory manager
actual streaming execution
actual example model
actual README
actual ZIP builder
```

 Do not respond with only an architectural explanation.

 Write the complete project.

 After writing it, inspect your own implementation and fix obvious issues before presenting it.

 If some requested functionality cannot safely be implemented in version 1, implement the working subset and clearly mark the limitation rather than producing fake functionality.

 The first goal is a **real, runnable proof-of-concept** of architecture-aware, disk-backed, memory-budgeted model execution.











##################################################################
##################################################################

##################################################################
##################################################################













 Now move from specification to implementation.

 Do NOT give me another high-level design.

 I want you to actually build the first working version of the runtime described above.

 # 1\. IMPLEMENTATION OBJECTIVE

 Build a runnable Python project that demonstrates this exact pipeline:

```
Hugging Face / Safetensors Model
            ↓
      Model Analyzer
            ↓
   Architecture-aware Graph
            ↓
      Memory Planner
            ↓
   Disk-backed Tensor Store
            ↓
   Async/Sync Scheduler
            ↓
      Node-by-node execution
            ↓
       Serialized output
```

 The first implementation must prioritize correctness over performance.

---

 # 2\. FIRST TARGET

 Do NOT try to support every Hugging Face architecture immediately.

 Build the runtime so that it can first support a small, well-defined model architecture and prove the complete architecture.

 Use a simple Transformer-like model or a small PyTorch model with:

```
Input
 ↓
Embedding
 ↓
Block 0
 ↓
Block 1
 ↓
Block 2
 ↓
Linear / Head
 ↓
Output
```

 The architecture adapter system must be extensible so additional Hugging Face architectures can be added later.

---

 # 3\. DO NOT CHEAT WITH MODEL LOADING

 The streaming runtime must NOT do:

```
AutoModel.from_pretrained(...)
```

 and then keep the complete model in memory.

 Do not use:

```
load_file(...)
```

 to load the entire safetensors file.

 Do not create a complete `state_dict` containing all weights.

 Do not use an implementation that secretly loads all model weights.

 The streaming runtime must work from:

```
manifest
+
safetensors metadata
+
disk ranges
```

---

 # 4\. SAFETENSORS STORAGE LAYER

 Implement a real streaming reader.

 Required API:

```
reader = SafeTensorStream(path)

metadata = reader.metadata()

tensor = reader.tensor("some.weight")

print(tensor.shape)
print(tensor.dtype)
print(tensor.nbytes)
print(tensor.offset)
```

 Support:

```
tensor.read(offset, length)
```

 and:

```
for chunk in tensor.iter_chunks(chunk_size):
    ...
```

 The implementation must read only the requested byte ranges.

 Use appropriate low-level file operations where possible.

 Do not copy the entire safetensors file into RAM.

---

 # 5\. TENSOR METADATA

 Each tensor descriptor should expose at least:

```
name
shape
dtype
nbytes
file_offset
file_length
```

 Example:

```
TensorDescriptor(
    name="layers.0.linear.weight",
    shape=(4096, 4096),
    dtype=torch.float32,
    nbytes=67108864,
    file_offset=123456,
)
```

 The descriptor itself must be tiny compared with the tensor.

---

 # 6\. ARCHITECTURE ANALYZER

 Implement:

```
class ArchitectureAnalyzer:
    ...
```

 It must inspect the prepared/reference model and determine:

```
model architecture
module hierarchy
execution order
computational nodes
tensor → node mapping
dependencies
```

 Do NOT determine nodes solely from file size.

---

 # 7\. NODE MODEL

 Implement something similar to:

```
@dataclass
class ModelNode:
    id: int
    name: str
    node_type: str
    inputs: list
    outputs: list
    weights: list
    dependencies: list
```

 Example:

```
Node 0
  name: embedding
  type: Embedding

Node 1
  name: block.0
  type: TransformerBlock

Node 2
  name: block.1
  type: TransformerBlock

Node 3
  name: block.2
  type: TransformerBlock

Node 4
  name: head
  type: Linear
```

---

 # 8\. ARCHITECTURAL BOUNDARY RULE

 This is mandatory.

 A logical node should end at a valid computational boundary.

 Do not produce arbitrary:

```
400 KB chunk
```

 boundaries.

 Instead:

```
Block 0
Block 1
Block 2
```

 should be the logical execution units.

 If a node is too large for RAM, preserve the logical node and use internal tiling.

---

 # 9\. MEMORY MANAGER

 Implement:

```
class MemoryManager:
    ...
```

 It must support:

```
reserve(size)
release(size)
available()
used()
peak()
```

 and context-manager style usage:

```
with memory.reserve(size):
    ...
```

 If the allocation would exceed the budget:

```
MemoryBudgetExceeded
```

 must occur.

---

 # 10\. MEMORY ACCOUNTING

 Track separately:

```
weights
activations
temporary
cache
prefetch
```

 Example:

```
RAM budget = 512 MB

weights       250 MB
activations    80 MB
temporary      60 MB
cache          40 MB
prefetch       20 MB
--------------------
total         450 MB
```

 The runtime must know its managed memory usage.

---

 # 11\. MEMORY SAFETY

 Before every allocation:

```
requested memory
+
current managed memory
<=
budget
```

 must be checked.

 Never intentionally exceed the budget.

 Do not rely on:

```
gc.collect()
```

 as the memory management mechanism.

 Explicitly release buffers.

---

 # 12\. NODE LOADING

 Implement:

```
class NodeLoader:
    ...
```

 Given:

```
node = graph.node(3)
```

 it should determine exactly which tensors are required.

 It must then read those tensors from disk.

 Do not load unrelated tensors.

---

 # 13\. NODE EXECUTION

 Implement:

```
class NodeExecutor:
    ...
```

 Conceptually:

```
weights = loader.load(node)

output = executor.execute(
    node,
    input_activation,
    weights,
)

loader.release(node)
```

 The implementation must make sure weights are released after they are no longer required.

---

 # 14\. INTERNAL TILING

 If a node fits:

```
load
 ↓
execute
 ↓
release
```

 If it does not:

```
tile 0
 ↓
compute
 ↓
release

tile 1
 ↓
compute
 ↓
release

tile 2
 ↓
compute
 ↓
release
```

 The logical node remains unchanged.

 Implement at least one real tiled operator.

 For example:

```
Linear / Matrix Multiplication
```

 Implement something conceptually like:

```
for weight_tile in weight_stream:
    partial = input @ weight_tile
    accumulate(partial)
```

 The tile size must be determined from the available memory budget.

---

 # 15\. AUTOMATIC TILE SIZE

 The user should not need to specify:

```
tile_size=400KB
```

 Instead:

```
tile_size = planner.choose_tile_size(...)
```

 The planner must consider:

```
RAM budget
current usage
input activation
output
temporary workspace
weight tile
```

---

 # 16\. MEMORY PLAN

 Implement:

```
class MemoryPlanner:
    ...
```

 API:

```
plan = planner.plan(
    graph,
    ram_budget=512 * 1024 * 1024,
)
```

 The plan should contain:

```
node
strategy
estimated memory
tile size
prefetch decision
```

 Example:

```
Node 0:
    strategy = direct

Node 1:
    strategy = direct

Node 2:
    strategy = tiled
    tile_size = 8 MB

Node 3:
    strategy = direct
```

---

 # 17\. ASYNC STORAGE

 Implement asynchronous storage support.

 Use Python mechanisms such as:

```
asyncio
ThreadPoolExecutor
async file operations through worker threads
```

 where appropriate.

 Because ordinary file I/O is blocking, do not pretend that:

```
async def read_file():
    file.read()
```

 is truly asynchronous.

 Use an executor/thread pool or another appropriate mechanism.

---

 # 18\. ASYNC PREFETCH

 Implement a basic prefetch mechanism.

 While:

```
Node N
```

 is computing, optionally start:

```
Node N+1
```

 disk reads.

 Example:

```
Compute Node N
      │
      ├──────────────► async read Node N+1
      │
      ▼
finish Node N
      │
      ▼
wait for Node N+1 if necessary
      │
      ▼
execute Node N+1
```

 However:

 > Prefetched data counts toward the memory budget.

 If there is insufficient memory, prefetch must not happen.

---

 # 19\. NODE SYNCHRONIZATION

 Node dependencies are authoritative.

 For:

```
Node 0 → Node 1 → Node 2
```

 Node 1 cannot consume the output of Node 0 before Node 0 has completed.

 Implement explicit synchronization.

 Use:

```
await
```

 or equivalent synchronization mechanisms where appropriate.

---

 # 20\. TOKEN GENERATION

 Implement an interface like:

```
async def generate(...):
    ...
```

 For each generated token:

```
Token N
   ↓
execute graph
   ↓
logits
   ↓
sample/select
   ↓
Token N+1
```

 Token ordering must remain serialized.

 Do not allow asynchronous tasks to reorder generated tokens.

---

 # 21\. ASYNC TOKEN INTERNALS

 Although token output is serialized, internal I/O can be asynchronous.

 For example:

```
Token N
 ├── Node 0 compute
 │
 ├── async prefetch Node 1
 │
 ├── Node 1 compute
 │
 ├── async prefetch Node 2
 │
 └── ...
```

 The public interface still emits:

```
Token 0
Token 1
Token 2
Token 3
```

 in order.

---

 # 22\. OUTPUT FIDELITY

 This is extremely important.

 Do NOT solve memory limitations by automatically:

```
quantizing
compressing
changing dtype
approximating
```

 The default runtime should preserve:

```
original weights
original dtype
original architecture
original operator semantics
```

 as closely as possible.

---

 # 23\. REFERENCE VS STREAMING

 Create:

```
reference_output(...)
streaming_output(...)
```

 and compare them.

 Example:

```
torch.testing.assert_close(
    streaming_output,
    reference_output,
    rtol=...,
    atol=...,
)
```

 The test must document why the selected tolerance is appropriate.

---

 # 24\. DETERMINISM

 Where possible, make the test deterministic:

```
torch.manual_seed(...)
```

 Use identical:

```
weights
inputs
dtype
operations
```

 for reference and streaming modes.

---

 # 25\. IMPORTANT FLOATING-POINT RULE

 If tiled computation changes floating-point accumulation order, small differences may occur.

 Do NOT falsely claim:

```
bit-for-bit identical
```

 unless the test actually proves it.

 Report:

```
max absolute difference
mean absolute difference
relative difference
allclose result
```

---

 # 26\. TEST WITH MODEL LARGER THAN RAM

 Create a test where:

```
model size > configured RAM budget
```

 For example:

```
model = 10 MB
RAM budget = 2 MB
```

 The model must still execute if the operators can be tiled to fit.

 Then test:

```
model = 100 MB
RAM = 10 MB
```

 if practical.

---

 # 27\. DO NOT FAKE MEMORY TESTING

 Do not simply claim:

```
model is 100 MB
RAM = 10 MB
```

 while secretly loading 100 MB.

 Use instrumentation to demonstrate:

```
maximum managed allocation
peak managed memory
```

 and verify that it stays within the configured budget.

---

 # 28\. DISK I/O INSTRUMENTATION

 Track:

```
read count
bytes read
read offsets
read sizes
```

 This allows tests to verify that the runtime is actually streaming.

 Example:

```
Tensor A:
  read:
    offset=1234
    length=65536

Tensor B:
  read:
    offset=90000
    length=65536
```

 instead of reading the whole file.

---

 # 29\. NO UNNECESSARY COPIES

 Be careful with:

```
tensor.clone()
tensor.contiguous()
numpy conversion
CPU → GPU copies
```

 because these may temporarily duplicate large buffers.

 Memory accounting must include intentional temporary copies.

 Avoid unnecessary copies.

---

 # 30\. MODEL PREPARATION

 Implement:

```
python -m stream_runtime prepare \
    <model_directory> \
    --output <prepared_directory>
```

 The command should produce:

```
prepared_directory/
    manifest.json
    metadata.json
```

 and reference the original safetensors files where possible.

 Do not duplicate a huge model unnecessarily during preparation.

---

 # 31\. INSPECT COMMAND

 Implement:

```
python -m stream_runtime inspect <model>
```

 It should show:

```
architecture
tensor count
parameter count
total bytes
largest tensor
nodes
node sizes
```

---

 # 32\. PLAN COMMAND

 Implement:

```
python -m stream_runtime plan \
    <prepared_model> \
    --ram-budget 2M
```

 Show:

```
RAM budget
node list
node strategy
estimated memory
tile size
prefetch
peak planned memory
```

---

 # 33\. RUN COMMAND

 Implement:

```
python -m stream_runtime run \
    <prepared_model> \
    --ram-budget 2M
```

 The command should run the test model and report:

```
tokens/output
peak memory
bytes read
read operations
nodes executed
execution strategy
```

---

 # 34\. TRACE MODE

 Add:

```
--trace
```

 Example:

```
python -m stream_runtime run model \
    --ram-budget 2M \
    --trace
```

 Output:

```
[DISK] read Node 0
[RAM] +512 KB
[EXEC] Node 0
[RAM] -512 KB

[PREFETCH] Node 1
[EXEC] Node 1

[RAM] peak=1.82 MB
```

 This will make debugging much easier.

---

 # 35\. UNIT TESTS

 Write real tests for:

```
safetensors metadata parsing
range reads
tensor chunk iteration
memory manager
memory budget enforcement
cache eviction
architecture graph
node dependencies
memory planner
linear tiling
node execution
async prefetch
token serialization
reference/streaming equivalence
```

---

 # 36\. INTEGRATION TEST

 Create one complete integration test:

```
create model
 ↓
save safetensors
 ↓
analyze model
 ↓
create manifest
 ↓
plan memory
 ↓
run streaming inference
 ↓
compare reference output
 ↓
verify memory budget
```

 This test is the most important test in the project.

---

 # 37\. FAILURE TEST

 Create tests where the budget is deliberately too small.

 Example:

```
RAM = 1 KB
minimum operator working set = 100 KB
```

 The runtime must produce a useful error:

```
MemoryBudgetExceeded

Node:
    block.2

Required minimum working memory:
    100 KB

Configured budget:
    1 KB

Reason:
    No supported tiling strategy can reduce this operator below the
    configured working-set requirement.
```

---

 # 38\. CLI DESIGN

 Use a clean CLI.

 Commands:

```
prepare
inspect
plan
run
test
```

 Examples:

```
python -m stream_runtime inspect model.safetensors

python -m stream_runtime prepare ./model \
    --output ./prepared

python -m stream_runtime plan ./prepared \
    --ram-budget 512M

python -m stream_runtime run ./prepared \
    --ram-budget 512M \
    --trace
```

---

 # 39\. LOGGING

 Use Python's standard:

```
logging
```

 Support:

```
quiet
normal
verbose
trace
```

 Do not print thousands of lines by default.

---

 # 40\. PROJECT QUALITY

 Use:

```
type hints
dataclasses
docstrings
clear module boundaries
exceptions
unit tests
```

 Avoid one giant Python file.

---

 # 41\. NO PLACEHOLDER IMPLEMENTATIONS

 Do not write:

```
pass
```

 for core functionality.

 Do not write:

```
# TODO: implement later
```

 for the fundamental streaming pipeline.

 If a feature genuinely cannot be implemented in the first version, explicitly isolate it and explain why.

 The core path must actually run.

---

 # 42\. FIRST SUPPORTED OPERATOR SET

 Implement at least:

```
Embedding
Linear
ReLU or equivalent simple activation
basic residual/add operation
```

 Then construct a small test network.

 This lets us prove:

```
disk streaming
node boundaries
memory planning
tiling
execution
```

 before implementing complicated attention.

---

 # 43\. TRANSFORMER SUPPORT

 After the simple network works, structure the code so Transformer blocks can be added.

 Do not attempt to implement every Hugging Face Transformer immediately.

 The architecture should eventually support:

```
Embedding
Attention
MLP
LayerNorm
Residual
LM Head
```

 as node/operator components.

---

 # 44\. GPU DESIGN

 Do not make CUDA mandatory.

 Create abstractions so later we can add:

```
DeviceMemoryManager
CPUDevice
CUDADevice
```

 The first implementation can use:

```
CPU only
```

 but the architecture should not prevent:

```
Disk
 ↓
RAM
 ↓
VRAM
 ↓
GPU
```

 later.

---

 # 45\. RAM + VRAM FUTURE

 Eventually support:

```
--ram-budget 512M
--vram-budget 256M
```

 and allow the planner to determine:

```
what remains on RAM
what moves to VRAM
when transfers happen
```

 Do not implement fake GPU support merely to satisfy the interface.

---

 # 46\. PERFORMANCE ARCHITECTURE

 Separate:

```
Storage
Memory
Planning
Graph
Operators
Execution
Scheduling
CLI
```

 Do not make operators directly open safetensors files.

 Use:

```
Operator
    ↓
TensorStore
    ↓
Storage backend
```

 This is important for future optimization.

---

 # 47\. FUTURE STORAGE BACKENDS

 Design:

```
class StorageBackend:
    ...
```

 with the first implementation:

```
SafeTensorsStorage
```

 Future possibilities:

```
sharded safetensors
memory mapped storage
custom binary storage
compressed storage
remote storage
```

---

 # 48\. FUTURE SCHEDULERS

 Design:

```
class Scheduler:
    ...
```

 Possible future implementations:

```
SyncScheduler
AsyncIOScheduler
PrefetchScheduler
GPUAsyncScheduler
```

 The first implementation can be simple.

---

 # 49\. IMPORTANT: DON'T OVERENGINEER VERSION 1

 Build the smallest real system that proves the concept.

 The first milestone is NOT:

```
support every Hugging Face model
```

 The first milestone is:

```
model larger than RAM
+
architecture-aware nodes
+
safetensors on disk
+
streaming tensor reads
+
automatic memory planning
+
tiled operator
+
node-by-node execution
+
async prefetch
+
serialized token/output
+
reference output comparison
```

 If this works, the architecture is proven.

---

 # 50\. SELF-VALIDATION

 Before presenting the result, run through the implementation mentally and check:

```
Can a 10 MB model execute with 2 MB managed RAM?
Can the safetensors file remain on disk?
Can only a tensor range be read?
Can a node be loaded and released?
Can an oversized node be tiled?
Can prefetch happen without exceeding RAM?
Can Node N+1 wait for Node N?
Can token output remain ordered?
Can reference and streaming outputs be compared?
Can the runtime fail safely when the budget is impossible?
```

 Fix issues you identify.

---

 # 51\. FINAL DELIVERABLE

 Provide:

```
complete project tree
all source files
tests
README
requirements
pyproject.toml
example model generator
CLI
ZIP builder
```

 Then provide the exact commands to:

```
install
create test model
inspect
prepare
plan
run
run tests
build ZIP
```

 Do not stop at pseudocode.

 Build the actual working prototype.











##################################################################
##################################################################

##################################################################
##################################################################













 Now extend the existing streaming runtime into a completely LOCAL model-serving system.

 The goal is to make the system usable like Ollama / LM Studio from the perspective of client applications, but with our architecture-aware disk-streaming execution engine underneath.

 # 1\. PRIMARY OBJECTIVE

 I want to run a large model locally even when the model is larger than available RAM/VRAM.

 The model remains stored on:

```
HDD / SSD
```

 as:

```
.safetensors
```

 The runtime streams only the required model data into RAM/VRAM according to the configured memory budget.

 The user should be able to start a local server:

```
python -m stream_runtime serve \
    ./models/my-model \
    --host 127.0.0.1 \
    --port 8000 \
    --ram-budget 2G
```

 Then another application should be able to communicate with it over:

```
http://127.0.0.1:8000
```

 with NO Internet connection.

---

 # 2\. OFFLINE-FIRST REQUIREMENT

 The server must work with:

```
NO INTERNET
NO EXTERNAL API
NO CLOUD MODEL
NO TELEMETRY
NO REMOTE INFERENCE
```

 After the required Python packages and model files have been installed/downloaded manually, runtime inference must work completely offline.

 Do not make runtime requests to:

```
Hugging Face
OpenAI
Anthropic
Google
any cloud API
```

 or any other Internet service.

---

 # 3\. OPENAI-COMPATIBLE API

 Implement an OpenAI-compatible HTTP API wherever practical.

 At minimum support:

```
POST /v1/chat/completions
POST /v1/completions
GET  /v1/models
GET  /health
```

 The goal is that applications which already support an OpenAI-compatible endpoint can use this server.

 Example:

```
Base URL:

http://127.0.0.1:8000/v1
```

---

 # 4\. CHAT COMPLETIONS

 Support requests similar to:

```
{
  "model": "my-model",
  "messages": [
    {
      "role": "user",
      "content": "Hello"
    }
  ],
  "stream": false
}
```

 Return an OpenAI-compatible response structure.

 Do not require an Internet connection.

---

 # 5\. STREAMING RESPONSES

 This is extremely important.

 Support:

```
{
  "model": "my-model",
  "messages": [
    {
      "role": "user",
      "content": "Write a Python function."
    }
  ],
  "stream": true
}
```

 The server should stream generated tokens over HTTP using the standard streaming mechanism expected by OpenAI-compatible clients, normally SSE.

 Conceptually:

```
Client
  │
  │ HTTP request
  ▼
Local Server
  │
  ▼
Streaming Runtime
  │
  ├── Disk I/O
  ├── RAM
  ├── compute
  └── token generation
  │
  ▼
SSE stream
  │
  ├── token 1
  ├── token 2
  ├── token 3
  └── ...
  │
  ▼
Client
```

 Tokens must be emitted in correct order.

---

 # 6\. OPENAI CLIENT COMPATIBILITY

 Make it possible to use a standard OpenAI-compatible client by changing only the base URL.

 For example:

```
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local"
)

response = client.chat.completions.create(
    model="my-model",
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)

print(response.choices[0].message.content)
```

 The server must NOT validate the API key against an Internet service.

 For local mode, accept a configurable local API key or optionally disable authentication.

---

 # 7\. CURL TEST

 Provide a working test:

```
curl http://127.0.0.1:8000/v1/models
```

 and:

```
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local" \
  -d '{
    "model": "my-model",
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ],
    "stream": false
  }'
```

 Also provide a streaming example:

```
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local" \
  -d '{
    "model": "my-model",
    "messages": [
      {
        "role": "user",
        "content": "Explain Python generators."
      }
    ],
    "stream": true
  }'
```

---

 # 8\. MODEL REGISTRY

 Implement a local model registry.

 Example directory:

```
models/
    llama-test/
        manifest.json
        config.json
        tokenizer.json
        model.safetensors

    another-model/
        manifest.json
        config.json
        tokenizer.json
        model.safetensors
```

 The server should discover locally installed models.

 No Internet lookup should be required.

---

 # 9\. MODELS ENDPOINT

 `GET /v1/models`

 should return locally available models.

 Example concept:

```
{
  "object": "list",
  "data": [
    {
      "id": "llama-test",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

 Do not report cloud models.

 Only return models actually available locally.

---

 # 10\. MODEL LOADING

 IMPORTANT:

 The server must NOT load the entire model into RAM simply because a client requests it.

 Use:

```
model registry
      ↓
manifest
      ↓
architecture graph
      ↓
memory planner
      ↓
streaming executor
```

 The `.safetensors` file remains on disk.

 Only required tensors/nodes should enter memory.

---

 # 11\. MODEL LIFECYCLE

 Implement:

```
UNLOADED
   ↓
INITIALIZING
   ↓
READY
   ↓
RUNNING
   ↓
READY
   ↓
UNLOADED
```

 Do not permanently keep the entire model in RAM.

---

 # 12\. OPTIONAL NODE CACHE

 Implement a configurable cache.

 Example:

```
--cache-size 512M
```

 The cache may keep recently used node weights in RAM.

 But:

```
cache + active weights + activation memory + temporary memory
```

 must never exceed the user-configured RAM budget.

 The cache must be optional.

---

 # 13\. SERVER MEMORY BUDGET

 Add:

```
--ram-budget 2G
```

 and eventually:

```
--vram-budget 4G
```

 The server must enforce the budget.

 Example:

```
RAM budget:        2 GB
Active weights:   900 MB
Activations:      400 MB
Temporary:        300 MB
Cache:            200 MB
Prefetch:         100 MB
--------------------------------
Total:            1.9 GB
```

 Never intentionally exceed:

```
2 GB
```

 of managed runtime memory.

---

 # 14\. REQUEST QUEUE

 Implement a request manager.

 The server may receive:

```
Request A
Request B
Request C
```

 but the underlying streaming runtime has limited memory.

 Do NOT allow unlimited requests to simultaneously consume model memory.

 Implement:

```
HTTP requests
      ↓
Request Queue
      ↓
Scheduler
      ↓
Execution
```

---

 # 15\. SINGLE-REQUEST MODE FIRST

 The first implementation should prioritize:

```
one active generation
```

 at a time.

 Other requests should wait in a queue.

 This is acceptable for the first version.

 Later we can implement batching.

---

 # 16\. ASYNC SERVER

 Use an asynchronous HTTP framework such as:

```
FastAPI
+
Uvicorn
```

 or another appropriate local async HTTP server.

 The server must not block the HTTP event loop during long disk reads or CPU computation.

 Use worker threads/processes appropriately for blocking operations.

---

 # 17\. ASYNC EXECUTION PIPELINE

 The server should eventually look like:

```
HTTP Request
     │
     ▼
Async Request Handler
     │
     ▼
Request Queue
     │
     ▼
Execution Scheduler
     │
     ├──────────────┐
     ▼              ▼
Disk I/O       Computation
async             worker
     │              │
     └──────┬───────┘
            ▼
       Token stream
            │
            ▼
       HTTP SSE stream
```

---

 # 18\. TOKEN SERIALIZATION

 Internally:

```
Disk I/O = asynchronous
Prefetch = asynchronous
Node preparation = asynchronous where possible
```

 But:

```
Node dependencies = synchronized
Token order = serialized
HTTP output = ordered
```

 For example:

```
Token 1
Token 2
Token 3
Token 4
```

 must never arrive to the client in a different order.

---

 # 19\. CANCELLATION

 Implement request cancellation.

 If the HTTP client disconnects:

```
client disconnected
       ↓
cancel generation
       ↓
stop future token generation
       ↓
cancel unnecessary prefetch
       ↓
release node memory
       ↓
release request resources
```

 Do not leave model buffers allocated indefinitely.

---

 # 20\. REQUEST TIMEOUT

 Support optional:

```
--request-timeout
```

 and/or per-request timeout.

 A timeout must cleanly release resources.

---

 # 21\. HEALTH ENDPOINT

 Implement:

```
GET /health
```

 Return information such as:

```
{
  "status": "ok",
  "offline": true,
  "runtime": "stream-runtime"
}
```

---

 # 22\. RUNTIME STATUS

 Add:

```
GET /v1/status
```

 Return local runtime information:

```
loaded model
RAM budget
managed RAM
peak RAM
active request
queue length
nodes executed
disk bytes read
```

 Do not expose sensitive system information unnecessarily.

---

 # 23\. LOGGING

 Example:

```
[SERVER] listening on 127.0.0.1:8000
[MODEL] llama-test
[MEMORY] budget=2GB
[REQUEST] id=abc123
[EXEC] node=0
[DISK] read=1.2MB
[EXEC] node=1
[TOKEN] generated=1
[TOKEN] generated=2
[REQUEST] complete
```

 Provide:

```
--verbose
--trace
```

 for detailed diagnostics.

---

 # 24\. NO TELEMETRY

 There must be no:

```
analytics
telemetry
remote logging
usage reporting
model reporting
```

 The server must remain completely local.

---

 # 25\. NO INTERNET DEPENDENCY AT RUNTIME

 Test the application with network access disabled.

 The runtime should still work.

 Make sure no code performs automatic:

```
pip install
model download
Hugging Face download
API calls
DNS requests
```

 during inference.

---

 # 26\. TOKENIZER

 The tokenizer must be local.

 Use local files such as:

```
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

 where applicable.

 Do not download a tokenizer at runtime.

 If a tokenizer is missing, produce a clear error telling the user to provide it.

---

 # 27\. CHAT TEMPLATE

 Support model-specific chat templates where available.

 Prefer locally stored configuration.

 Do not fetch templates from the Internet.

 The runtime should have a clean abstraction:

```
class ChatTemplate:
    def apply(messages):
        ...
```

---

 # 28\. API COMPATIBILITY

 Prioritize compatibility with tools that support OpenAI-compatible local servers.

 The following should ideally work by changing:

```
base_url
```

 only:

```
Claude Code
OpenAI-compatible clients
local agent frameworks
Python applications
custom applications
```

 Do NOT claim compatibility with a particular application until you actually test the protocol it expects.

---

 # 29\. AGENT/CODE-ASSISTANT USE CASE

 The server should be usable as a local inference backend for coding agents.

 Example:

```
Coding Agent
     │
     │ OpenAI-compatible HTTP
     ▼
127.0.0.1:8000
     │
     ▼
Streaming Runtime
     │
     ▼
Large local model
     │
     ▼
HDD/SSD
```

 The agent must not need direct access to the model files.

 Only the server accesses the model.

---

 # 30\. SECURITY

 By default bind only to:

```
127.0.0.1
```

 NOT:

```
0.0.0.0
```

 unless the user explicitly requests it.

 If the user wants LAN access, require an explicit:

```
--host 0.0.0.0
```

 option and clearly warn that this exposes the API to the network.

---

 # 31\. API KEY

 Support local authentication:

```
--api-key local-secret
```

 For localhost-only development, allow:

```
--no-auth
```

 Never send the API key anywhere externally.

---

 # 32\. MODEL SERVER CLI

 Implement:

```
python -m stream_runtime serve \
    --model ./models/my-model \
    --host 127.0.0.1 \
    --port 8000 \
    --ram-budget 2G
```

 Also:

```
python -m stream_runtime models
```

 and:

```
python -m stream_runtime status
```

 where appropriate.

---

 # 33\. SERVER STARTUP

 Startup should validate:

```
model exists
safetensors exists
manifest valid
architecture supported
tokenizer available
memory budget valid
```

 Then print:

```
Stream Runtime Server

Model:       my-model
Storage:     /models/my-model
RAM budget:  2 GB
VRAM budget: disabled
Host:        127.0.0.1
Port:        8000
API:         OpenAI-compatible
Internet:    disabled
Status:      READY
```

---

 # 34\. MODEL WARMUP

 Do NOT automatically load the entire model.

 If warmup is enabled:

```
--warmup
```

 only perform the minimum necessary initialization.

 Warmup must obey the memory budget.

---

 # 35\. SSE FORMAT

 Implement proper SSE streaming semantics.

 The server should emit data chunks in a form compatible with OpenAI-style streaming clients.

 Terminate the stream correctly.

 Handle:

```
client disconnect
generation error
server error
normal completion
```

 correctly.

---

 # 36\. ERROR FORMAT

 Return structured JSON errors.

 Example:

```
{
  "error": {
    "message": "RAM budget is too small for this model",
    "type": "memory_budget_error",
    "code": "MEMORY_BUDGET_EXCEEDED"
  }
}
```

 Do not expose Python stack traces to normal clients.

 Stack traces should only appear in server logs/debug mode.

---

 # 37\. TEST CLIENT

 Create:

```
examples/client.py
```

 which connects to:

```
http://127.0.0.1:8000/v1
```

 and performs:

```
model discovery
chat request
streaming chat request
```

---

 # 38\. OFFLINE INTEGRATION TEST

 Create an integration test that:

 1. Starts the local server.
2. Disables/blocks network access if practical.
3. Sends an HTTP request.
4. Generates output.
5. Confirms the output is received.
6. Confirms no external service is required.
7. Confirms memory budget is respected.
8. Confirms model data came from local storage.

---

 # 39\. MEMORY VERIFICATION

 During an HTTP request, expose internal metrics to the test framework:

```
peak managed RAM
current managed RAM
disk bytes read
number of node loads
number of node releases
```

 Verify:

```
peak managed RAM <= configured RAM budget
```

 within the clearly documented scope of what the memory manager controls.

 Do not falsely claim that Python/PyTorch process RSS can never exceed the budget unless the implementation actually guarantees that.

 This distinction is important.

---

 # 40\. PERFORMANCE IS NOT THE FIRST PRIORITY

 Do NOT optimize aggressively yet.

 The priorities are:

```
1. Correctness
2. Offline operation
3. Memory-budget enforcement
4. Output fidelity
5. API compatibility
6. Async architecture
7. Performance
```

---

 # 41\. IMPORTANT ARCHITECTURAL SEPARATION

 Keep these components independent:

```
HTTP Server
     ↓
API Adapter
     ↓
Request Manager
     ↓
Generation Engine
     ↓
Execution Scheduler
     ↓
Memory Planner
     ↓
Model Graph
     ↓
Node Executor
     ↓
Tensor Store
     ↓
Safetensors
     ↓
HDD / SSD
```

 The HTTP layer must NOT know how safetensors offsets work.

 The safetensors layer must NOT know anything about HTTP.

 The scheduler must not depend on FastAPI.

 This separation is mandatory.

---

 # 42\. FUTURE COMPATIBILITY

 Design the API so later we can add:

```
multiple models
model unload
model reload
batch requests
continuous batching
multiple concurrent users
GPU execution
CPU/GPU hybrid execution
KV cache management
speculative decoding
quantization
distributed execution
```

 But do not implement these prematurely.

---

 # 43\. OLLAMA/LM STUDIO COMPARISON

 Do NOT copy their internal implementation.

 We only want a similar user experience:

```
start local server
        ↓
localhost endpoint
        ↓
client application
        ↓
local model
```

 Our backend remains:

```
architecture-aware
disk-backed
memory-budgeted
streaming
async
offline
```

---

 # 44\. EXAMPLE END-TO-END WORKFLOW

 The final README should show:

```
# 1. Prepare model
python -m stream_runtime prepare ./my-model

# 2. Inspect
python -m stream_runtime inspect ./my-model

# 3. Plan
python -m stream_runtime plan \
    ./my-model \
    --ram-budget 512M

# 4. Start local server
python -m stream_runtime serve \
    --model ./my-model \
    --host 127.0.0.1 \
    --port 8000 \
    --ram-budget 512M

# 5. Test
curl http://127.0.0.1:8000/v1/models

# 6. Chat
curl http://127.0.0.1:8000/v1/chat/completions ...
```

---

 # 45\. DO NOT CHANGE THE CORE MEMORY PRINCIPLE

 The entire purpose of this project is:

```
Model size
    >
RAM size
```

 and still:

```
MODEL EXECUTES LOCALLY
```

 For example:

```
Model:       20 GB
RAM budget:  1 GB
VRAM:        0 GB
Storage:     500 GB SSD
```

 The runtime should attempt to execute the model through:

```
SSD
 ↓
small RAM working set
 ↓
compute
 ↓
release
 ↓
next node
```

 If a particular operator mathematically cannot be executed within the available memory, fail clearly rather than silently changing the model.

---

 # 46\. FINAL DELIVERABLE

 Update the existing project.

 Provide:

```
complete source tree
server implementation
OpenAI-compatible API
CLI
streaming SSE
local model registry
request scheduler
memory enforcement
async execution
tests
example client
curl examples
README
requirements
pyproject.toml
```

 Also create a ZIP archive containing the complete project.

 Before declaring completion, actually test:

```
server starts
/v1/models works
/health works
chat completion works
streaming works
client.py works
offline operation works
memory budget is respected
model remains on disk
node streaming occurs
tokens arrive in order
reference-vs-streaming test passes within documented tolerance
```

 Do not just provide pseudocode or an architectural proposal.

 Implement the working local server on top of the existing streaming runtime.












##################################################################
##################################################################

##################################################################
##################################################################












 Now create the complete documentation and contributor infrastructure for this project.

 Treat this as a serious open-source project that will eventually accept contributions from developers who have never seen the codebase before.

 The documentation must be sufficient for a new contributor to:

```
understand the project
↓
set up the development environment
↓
understand the architecture
↓
run the project locally
↓
run tests
↓
debug problems
↓
implement a feature
↓
modify the runtime safely
↓
add a new model architecture
↓
benchmark changes
↓
submit a commit
↓
open a pull request
```

 Do not create generic documentation.

 Documentation must describe the actual implementation that exists in this repository.

 If something is not implemented yet, clearly mark it as:

```
Planned
Not implemented
Experimental
Future work
```

 Never document a feature as implemented when it is only planned.

---

 # 1\. DOCUMENTATION PHILOSOPHY

 The documentation should make this project understandable at three levels.

 ## Level 1 — User

 A user should understand:

```
What is this?
Why does it exist?
What models can I run?
How do I install it?
How do I run a model?
How do I start the local server?
How do I connect an application?
How do I configure RAM/VRAM?
```

 ## Level 2 — Developer

 A developer should understand:

```
How does the runtime work?
How is a model represented?
How are safetensors read?
How does memory management work?
How does node execution work?
How does scheduling work?
How does async I/O work?
How does token generation work?
```

 ## Level 3 — Contributor

 A contributor should understand:

```
Where should I modify the code?
How do I add an operator?
How do I add an architecture?
How do I add a storage backend?
How do I change the scheduler?
How do I write tests?
How do I benchmark?
How do I submit a PR?
```

---

 # 2\. REQUIRED DOCUMENTATION TREE

 Create a professional documentation structure similar to a mature open-source project.

 At minimum:

```
README.md

CONTRIBUTING.md
CODE_OF_CONDUCT.md
SECURITY.md
LICENSE

CHANGELOG.md

docs/
    README.md

    getting-started/
        installation.md
        quickstart.md
        configuration.md
        first-model.md
        troubleshooting.md

    user-guide/
        model-management.md
        model-preparation.md
        memory-budgets.md
        local-server.md
        openai-api.md
        streaming.md
        offline-mode.md
        cli.md

    architecture/
        overview.md
        design-principles.md
        execution-model.md
        model-graph.md
        architecture-analysis.md
        node-system.md
        tensor-storage.md
        safetensors.md
        memory-management.md
        memory-planner.md
        tiling.md
        scheduler.md
        async-execution.md
        token-generation.md
        request-lifecycle.md
        api-server.md
        caching.md
        cpu-execution.md
        gpu-execution.md
        numerical-fidelity.md

    development/
        development-setup.md
        repository-layout.md
        coding-standards.md
        testing.md
        debugging.md
        profiling.md
        benchmarking.md
        adding-operators.md
        adding-model-architectures.md
        adding-storage-backends.md
        adding-schedulers.md

    api/
        README.md
        python-api.md
        server-api.md
        configuration-api.md

    reference/
        configuration-reference.md
        cli-reference.md
        environment-variables.md
        error-codes.md
        glossary.md

    design/
        adr/
```

 Adjust the structure if the actual repository requires something different, but keep the same level of organization.

---

 # 3\. README.md

 Rewrite/create the main README as the project's front door.

 It must contain:

```
Project name
One-line description
Problem being solved
Core idea
Architecture diagram
Key features
Current limitations
Supported models
Installation
Quick start
Local server
OpenAI-compatible API
Memory budget examples
Offline usage
Development
Testing
Contribution
License
```

 Explain the core concept clearly:

```
Model on SSD/HDD
        ↓
Architecture graph
        ↓
Complete logical nodes
        ↓
Memory planner
        ↓
Only required weights loaded
        ↓
Compute
        ↓
Release
        ↓
Next node
```

 Make clear that the project's purpose is NOT simply quantization.

 Explain the distinction between:

```
quantization
compression
offloading
streaming execution
architecture-aware execution
```

---

 # 4\. PROJECT MISSION

 Create a dedicated section explaining the project's mission.

 The central principle should be documented approximately as:

 > Available RAM/VRAM should primarily determine execution strategy and performance, rather than determining whether the model can be executed at all, provided the model's operators have a valid execution strategy within the available working memory.

 Explain that:

```
Model size > RAM
```

 is a supported design target.

 Example:

```
Model = 20 GB
RAM budget = 1 GB
VRAM = 0
SSD = 500 GB
```

 The runtime should attempt:

```
SSD
 ↓
RAM
 ↓
compute
 ↓
release
 ↓
next node
```

 If an operator cannot mathematically execute within the available budget, the runtime must fail clearly rather than silently altering the model.

---

 # 5\. ARCHITECTURE OVERVIEW

 Create a detailed architecture document.

 Include diagrams such as:

```
                    HTTP CLIENT
                        │
                        ▼
                 Local API Server
                        │
                        ▼
                  Request Manager
                        │
                        ▼
                 Generation Engine
                        │
                        ▼
                  Execution Scheduler
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
       Memory Planner       Async I/O
              │                   │
              └─────────┬─────────┘
                        ▼
                   Model Graph
                        │
                        ▼
                   Node Executor
                        │
                        ▼
                    Tensor Store
                        │
                        ▼
                  Safetensors
                        │
                        ▼
                     SSD/HDD
```

 Explain every component.

---

 # 6\. DATA FLOW

 Document a complete inference request.

 For example:

```
HTTP request
 ↓
tokenization
 ↓
generation state
 ↓
Node 0
 ↓
Node 1
 ↓
Node 2
 ↓
...
 ↓
logits
 ↓
sampling
 ↓
next token
 ↓
repeat
 ↓
HTTP stream
```

 Explain which parts are:

```
synchronous
asynchronous
serialized
parallelizable
```

---

 # 7\. MEMORY MODEL

 Create a dedicated memory-management document.

 Explain:

```
RAM budget
VRAM budget
active weights
activations
temporary buffers
cache
prefetch buffers
allocator overhead
```

 Explain the difference between:

```
logical managed memory
actual process RSS
OS page cache
GPU allocated memory
```

 This distinction is extremely important.

 Do not make false claims that setting:

```
--ram-budget 1G
```

 guarantees that the entire Python process can never exceed exactly 1 GB.

 Document exactly what the runtime controls.

---

 # 8\. MEMORY BUDGET EXAMPLES

 Include examples.

 Example:

```
RAM budget: 512 MB

Weights:      250 MB
Activations:   80 MB
Temporary:     60 MB
Cache:         40 MB
Prefetch:      20 MB
--------------------
Total:        450 MB
```

 Explain how the planner makes decisions.

---

 # 9\. ARCHITECTURE-AWARE NODE SYSTEM

 This needs its own detailed documentation.

 Explain why arbitrary byte chunks are NOT the primary logical execution unit.

 Bad:

```
400 KB chunk
400 KB chunk
400 KB chunk
```

 Preferred:

```
Embedding
Transformer Block 0
Transformer Block 1
Transformer Block 2
LM Head
```

 Then explain internal tiling:

```
Transformer Block 2
 ├── tile 0
 ├── tile 1
 ├── tile 2
 └── tile 3
```

 The logical graph remains:

```
Block 2
```

 even when physical execution uses tiles.

---

 # 10\. SAFETENSORS DOCUMENTATION

 Document exactly how `.safetensors` files are handled.

 Explain:

```
metadata
tensor offsets
tensor sizes
dtype
shape
range reads
chunk iteration
```

 Show examples of:

```
reader = SafeTensorStream(...)
tensor = reader.tensor(...)
```

 Explain how the implementation avoids loading the entire file.

 Also document limitations of the current implementation.

---

 # 11\. MODEL ARCHITECTURE SUPPORT

 Create a contributor guide explaining how to add support for a new Hugging Face architecture.

 Include:

```
1. Identify model architecture
2. Identify module hierarchy
3. Identify logical node boundaries
4. Map model tensors to nodes
5. Implement architecture adapter
6. Implement operator mappings
7. Add tokenizer/chat template handling
8. Add tests
9. Add reference-vs-streaming comparison
10. Add documentation
```

 Provide a concrete example using an already-supported architecture.

---

 # 12\. OPERATOR DEVELOPMENT GUIDE

 Create:

```
docs/development/adding-operators.md
```

 Explain:

```
operator interface
inputs
outputs
weights
memory requirements
tiling
device handling
dtype handling
reference implementation
streaming implementation
tests
```

 Explain what a contributor must do if an operator does not fit into the configured memory budget.

---

 # 13\. TILING DOCUMENTATION

 Document the tiling system thoroughly.

 Explain:

```
logical operator
physical tiles
tile size selection
memory calculation
accumulation
temporary buffers
dtype
output buffers
```

 Give a simple example using matrix multiplication.

 Explain why changing accumulation order may affect floating-point results.

---

 # 14\. NUMERICAL FIDELITY

 Create a dedicated document.

 Explain:

```
bitwise equality
numerical equivalence
approximate output
```

 The default goal is:

```
same weights
same dtype
same architecture
same semantics
```

 as closely as practical.

 Explain why:

```
tiling
parallel kernels
floating-point accumulation order
CPU vs GPU
different BLAS implementations
```

 can produce small differences.

 Document how tests use:

```
torch.testing.assert_close(...)
```

 and how tolerances are selected.

---

 # 15\. SCHEDULER DOCUMENTATION

 Explain the scheduler in detail.

 Logical dependency:

```
Node 0
 ↓
Node 1
 ↓
Node 2
```

 Physical execution:

```
Compute Node 0
      │
      ├── async read Node 1
      │
      ▼
Finish Node 0
      │
      ▼
Execute Node 1
```

 Explain:

```
prefetch
dependency management
memory reservation
task cancellation
backpressure
queueing
```

---

 # 16\. TOKEN GENERATION

 Document why:

```
node dependencies
```

 are synchronized while:

```
disk I/O
prefetch
internal preparation
```

 can be asynchronous.

 Document token serialization:

```
Token 0
Token 1
Token 2
Token 3
```

 and explain why asynchronous internal work must never reorder externally visible output.

---

 # 17\. LOCAL SERVER DOCUMENTATION

 Create a complete guide for running the server.

 Example:

```
python -m stream_runtime serve \
    --model ./models/my-model \
    --host 127.0.0.1 \
    --port 8000 \
    --ram-budget 1G
```

 Explain:

```
host
port
RAM budget
VRAM budget
cache
authentication
logging
trace mode
```

---

 # 18\. OPENAI-COMPATIBLE API

 Document all currently supported endpoints.

 At minimum:

```
GET /health
GET /v1/models
GET /v1/status
POST /v1/chat/completions
POST /v1/completions
```

 For each endpoint document:

```
HTTP method
URL
headers
request body
response body
streaming response
errors
examples
```

 Use realistic examples.

---

 # 19\. API COMPATIBILITY

 Document exactly what "OpenAI-compatible" means in this project.

 Do NOT claim full compatibility if only a subset is implemented.

 Create a compatibility matrix:

 | Feature | Status |
| --- | --- |
| `/v1/models` | Supported |
| `/v1/chat/completions` | Supported |
| Streaming | Supported |
| SSE | Supported |
| Tool calls | Planned/Supported |
| Embeddings | Planned/Supported |
| etc. | ... |

 Populate this from the actual implementation.

---

 # 20\. CODE-AGENT INTEGRATION

 Create a guide explaining how a generic OpenAI-compatible client can connect.

 Example:

```
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local"
)
```

 Also document generic configuration concepts for coding-agent applications.

 Do NOT claim that Claude Code, Cursor, Continue, Cline, or any other specific product works unless compatibility has actually been tested.

 If tested, document the exact configuration.

---

 # 21\. OFFLINE MODE

 Create a dedicated offline guide.

 Explain:

```
what is required before going offline
where model files live
where tokenizer files live
how dependencies are installed
how to verify no network is required
```

 Document:

```
No model downloads
No tokenizer downloads
No telemetry
No cloud inference
No remote API calls
```

---

 # 22\. INSTALLATION

 Document development installation.

 Support a clean environment such as:

```
git clone ...
cd ...
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

 If the project uses a different package manager, document the actual one.

 Include:

```
Linux
Windows
macOS
```

 where actually supported.

 Do not claim platform support that has not been tested.

---

 # 23\. REPOSITORY LAYOUT

 Create a detailed explanation of the repository.

 For example:

```
src/
    stream_runtime/
        api/
        cli/
        graph/
        memory/
        models/
        operators/
        scheduler/
        storage/
        generation/
```

 Explain the responsibility of every major package.

 This section should help a contributor quickly locate the code they need.

---

 # 24\. DEVELOPMENT WORKFLOW

 Document:

```
create branch
make change
run formatter
run linter
run type checker
run unit tests
run integration tests
run benchmarks
update docs
commit
push
open PR
```

 Use the actual project's tools.

---

 # 25\. CONTRIBUTING.md

 Create a professional contributor guide.

 It should include:

```
Welcome
Code of conduct
Before starting work
Issues
Feature requests
Bug reports
Architecture changes
Coding standards
Testing requirements
Documentation requirements
Commit requirements
Pull request requirements
Review process
```

---

 # 26\. CONTRIBUTOR PRINCIPLES

 Make these explicit:

```
Correctness before performance
Memory safety before performance
Output fidelity before optimization
No silent dtype changes
No silent quantization
No hidden model downloads
No hidden network access
No unnecessary full-model loading
```

---

 # 27\. COMMIT CONVENTION

 Define a commit convention.

 Prefer Conventional Commits:

```
feat:
fix:
docs:
test:
refactor:
perf:
build:
ci:
chore:
```

 Examples:

```
feat(storage): add safetensors range reader
fix(memory): prevent prefetch budget overflow
feat(scheduler): add async node prefetch
test(streaming): compare tiled linear output
docs(architecture): explain node boundaries
```

 Explain when each type should be used.

---

 # 28\. PULL REQUEST TEMPLATE

 Create:

```
.github/pull_request_template.md
```

 Include:

```
Summary
Motivation
What changed
Implementation details
Tests
Memory impact
Performance impact
Numerical/output impact
API impact
Documentation updated
Breaking changes
Checklist
```

 For runtime changes, require contributors to explicitly state:

```
RAM impact
VRAM impact
disk I/O impact
output fidelity impact
```

---

 # 29\. ISSUE TEMPLATES

 Create templates for:

```
bug report
feature request
architecture support request
performance issue
memory-budget issue
numerical correctness issue
security issue
```

 Do not expose security reports publicly if the project uses a private security-reporting process.

---

 # 30\. CODE OF CONDUCT

 Create a standard professional open-source Code of Conduct.

 Keep it concise and appropriate for a technical project.

---

 # 31\. SECURITY.md

 Document:

```
security scope
local server security
localhost default
LAN exposure
API authentication
dependency security
model file trust
malicious model considerations
reporting vulnerabilities
```

 Important:

 The default server should bind to:

```
127.0.0.1
```

 and documentation should explain why.

---

 # 32\. MODEL SECURITY

 Document that model files are untrusted input from the runtime's perspective.

 Discuss:

```
malformed safetensors
unexpected metadata
invalid tensor shapes
resource exhaustion
path traversal
unsafe model configuration
```

 The runtime should validate metadata before attempting allocation.

---

 # 33\. TESTING GUIDE

 Create a complete testing guide.

 Explain:

```
unit tests
integration tests
model tests
memory tests
numerical tests
API tests
offline tests
async tests
cancellation tests
failure tests
```

---

 # 34\. MEMORY TESTING

 Document how to test:

```
model > RAM
node > RAM
tile < RAM
prefetch + compute
cache eviction
budget exceeded
```

 Include expected behavior.

---

 # 35\. NUMERICAL TESTING

 Document:

```
reference execution
streaming execution
comparison
tolerances
dtype differences
CPU/GPU differences
```

---

 # 36\. PERFORMANCE / BENCHMARKING

 Create:

```
docs/development/benchmarking.md
```

 Document metrics:

```
tokens/sec
latency
disk throughput
bytes read
I/O operations
node load time
node compute time
prefetch hit rate
peak managed memory
cache hit rate
```

 Make clear that performance is secondary to correctness in the current stage.

---

 # 37\. DEBUGGING GUIDE

 Document common problems.

 Examples:

```
model does not fit
memory budget exceeded
unsupported architecture
missing tokenizer
invalid safetensors
numerical mismatch
server won't start
port already in use
client cannot connect
streaming response stops
slow disk
async cancellation issue
```

 For each provide:

```
symptom
likely cause
diagnostic command
solution
```

---

 # 38\. ERROR CODES

 Create a central error reference.

 Examples:

```
MODEL_NOT_FOUND
UNSUPPORTED_ARCHITECTURE
INVALID_SAFETENSORS
MEMORY_BUDGET_EXCEEDED
UNSUPPORTED_OPERATOR
TOKENIZER_NOT_FOUND
SERVER_ERROR
REQUEST_CANCELLED
```

 Use the actual exceptions implemented by the project.

---

 # 39\. ARCHITECTURE DECISION RECORDS

 Create an ADR system:

```
docs/design/adr/
```

 Include at least initial ADRs for:

```
ADR-0001: Architecture-aware execution nodes
ADR-0002: Safetensors disk streaming
ADR-0003: User-defined memory budgets
ADR-0004: Logical nodes vs physical tiles
ADR-0005: Async I/O with synchronized node dependencies
ADR-0006: OpenAI-compatible local API
ADR-0007: Offline-first runtime
```

 Each ADR should contain:

```
Status
Context
Decision
Alternatives considered
Consequences
```

 Use the actual project decisions.

---

 # 40\. GLOSSARY

 Create a glossary defining terms such as:

```
Node
Logical Node
Physical Tile
Tensor
Tensor Descriptor
Tensor Store
Safetensors
Memory Budget
Working Set
Prefetch
Cache
Activation
Scheduler
Execution Graph
Architecture Adapter
Reference Execution
Streaming Execution
Numerical Equivalence
Token
KV Cache
SSE
OpenAI-Compatible API
```

---

 # 41\. ROADMAP

 Create a realistic roadmap.

 Separate:

```
Completed
Current
Next
Future
Research
```

 Possible future areas:

```
more Hugging Face architectures
CUDA
CPU/GPU hybrid execution
better tiling
better prefetch
multiple concurrent requests
continuous batching
KV-cache optimization
quantization as an optional feature
additional storage formats
distributed execution
```

 Do not mark anything completed unless it exists.

---

 # 42\. CHANGELOG

 Create/update:

```
CHANGELOG.md
```

 Use a conventional structure.

 Example:

```
## [Unreleased]

### Added
### Changed
### Fixed
### Documentation
### Performance
### Breaking Changes
```

---

 # 43\. LICENSE

 If the repository already has a license, document it.

 If no license has been selected, do NOT invent a legal choice silently.

 Instead:

```
create a placeholder/documentation note
```

 and clearly tell me that a license decision is required.

---

 # 44\. GITHUB CONFIGURATION

 Create appropriate files under:

```
.github/
```

 such as:

```
pull_request_template.md

ISSUE_TEMPLATE/
    bug_report.yml
    feature_request.yml
    architecture_support.yml
    performance.yml
    memory_issue.yml
    numerical_correctness.yml

workflows/
    tests.yml
    lint.yml
```

 Only add CI workflows that match the actual project dependencies.

---

 # 45\. AUTOMATED DOCUMENTATION CHECKS

 Add CI checks where practical for:

```
tests
formatting
linting
type checking
documentation links
```

 Do not add complicated CI infrastructure unnecessarily.

---

 # 46\. DOCUMENTATION CONSISTENCY

 Make sure documentation agrees with code.

 Before finishing:

```
search documentation for commands
search documentation for class names
search documentation for CLI options
search documentation for API endpoints
```

 Verify that all documented examples match the actual implementation.

---

 # 47\. NO FAKE DOCUMENTATION

 This is mandatory.

 Never write:

```
"CUDA supported"
```

 if CUDA support is not implemented.

 Never write:

```
"all Hugging Face models supported"
```

 unless it is actually true.

 Never write:

```
"hard memory limit"
```

 if only managed allocations are limited.

 Never write:

```
"bit-for-bit identical"
```

 unless tests prove it.

 Documentation must distinguish:

```
Implemented
Experimental
Partial
Planned
```

---

 # 48\. CONTRIBUTOR QUICK START

 Create a short page:

```
docs/development/contributor-quickstart.md
```

 A new contributor should be able to follow:

```
git clone ...
cd project
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest

python -m stream_runtime --help
```

 Then explain:

```
Pick an issue
Create branch
Make change
Add tests
Run tests
Update docs
Commit
Push
Open PR
```

---

 # 49\. "WHERE DO I CHANGE THIS?" GUIDE

 Create a contributor navigation table:

 | I want to... | Look here |
| --- | --- |
| Change safetensors reading | storage/ |
| Change memory accounting | memory/ |
| Change node planning | graph/ / planner/ |
| Add an operator | operators/ |
| Add model architecture | models/ |
| Change async scheduling | scheduler/ |
| Change token generation | generation/ |
| Change HTTP API | api/ |
| Change CLI | cli/ |
| Add tests | tests/ |

 Use the actual repository paths rather than guessing them.

---

 # 50\. DOCUMENT CODE EXAMPLES

 For important public classes/functions, add docstrings containing:

```
purpose
parameters
return values
exceptions
memory behavior
async behavior
examples
```

 Do not over-document trivial private functions.

---

 # 51\. API REFERENCE GENERATION

 If practical, configure automatic Python API documentation.

 Possible tools:

```
MkDocs
Material for MkDocs
mkdocstrings
Sphinx
```

 Choose an appropriate lightweight solution.

 Do not add a documentation framework merely for appearance.

 It must be maintainable.

---

 # 52\. DOCUMENTATION SITE

 If practical, create:

```
mkdocs.yml
```

 with navigation such as:

```
Home
Getting Started
User Guide
Architecture
Development
API Reference
Design Decisions
Reference
Contributing
```

 If the project is not ready for a documentation site, structure Markdown so it can easily be added later.

---

 # 53\. EXAMPLES DIRECTORY

 Create:

```
examples/
```

 where appropriate.

 Include examples such as:

```
basic_inference.py
streaming_chat.py
openai_client.py
memory_budget.py
inspect_model.py
```

 Every example must work with the current implementation.

---

 # 54\. DOCUMENT THE CORE DESIGN IN ONE PAGE

 Create:

```
docs/architecture/one-page-overview.md
```

 A contributor should be able to read this in approximately five minutes and understand:

```
what problem the project solves
how data moves
how memory is controlled
how nodes work
how disk streaming works
how async execution works
how token generation works
how the server exposes the runtime
```

---

 # 55\. ARCHITECTURE DIAGRAMS

 Use Mermaid diagrams where appropriate.

 Example:

 Mermaid flowchart: HTTP Client, Local API Server, Request Manager, Generation Engine, Execution Scheduler, Memory Planner, Async Storage, Model Graph, Node Executor, Safetensors, SSD/HDD

 Use diagrams for:

```
architecture
memory flow
node execution
async prefetch
token generation
request lifecycle
```

---

 # 56\. DOCUMENT THE DESIGN PRINCIPLE

 Create a prominent architecture principle:

```
LOGICAL MODEL ≠ PHYSICAL STORAGE
```

 The logical model is:

```
Embedding
Block 0
Block 1
Block 2
Head
```

 Physical execution may be:

```
SSD
 ↓
tile
 ↓
RAM
 ↓
CPU/GPU
 ↓
release
 ↓
next tile
```

 This distinction is fundamental to the project.

---

 # 57\. DOCUMENT MEMORY AS A PLANNING PROBLEM

 Explain that the runtime is solving:

```
Given:

Model graph
+
available RAM
+
available VRAM
+
operator requirements

Determine:

node loading strategy
tile size
prefetch strategy
cache strategy
execution placement
```

 The model itself should not be modified merely because memory is limited.

---

 # 58\. DOCUMENT FAILURE BOUNDARIES

 Explain that "any model size" does NOT mean mathematically impossible workloads can always run.

 For example:

```
If an operator requires at least 100 MB of working memory
and the configured budget is 10 MB,
```

 the runtime cannot magically execute it without changing the algorithm or model semantics.

 The correct behavior is:

```
clear failure
+
required memory
+
available memory
+
possible mitigation
```

 Do not hide this limitation.

---

 # 59\. DOCUMENT FUTURE RESEARCH

 Create:

```
docs/design/future-research.md
```

 Discuss possible future techniques without pretending they are implemented:

```
operator-level streaming
out-of-core matrix multiplication
multi-level caching
SSD-aware scheduling
direct I/O
memory mapping
CUDA unified memory
GPU staging
KV cache paging
continuous batching
multi-model scheduling
compression
quantization
distributed storage
```

---

 # 60\. FINAL CONTRIBUTOR EXPERIENCE

 After documentation is complete, simulate a new contributor.

 Pretend you know nothing about the project.

 Try to answer:

```
What is this project?
How do I install it?
How do I run it?
Where is the model loader?
Where is memory management?
Where is node execution?
Where is the scheduler?
Where is the HTTP server?
How do I add a model architecture?
How do I add an operator?
How do I run tests?
How do I benchmark?
How do I report a bug?
How do I submit a PR?
```

 If any answer cannot be found easily, improve the documentation.

---

 # 61\. FINAL QUALITY CHECK

 Before finishing, verify:

```
[ ] README is accurate
[ ] Installation works
[ ] Quickstart works
[ ] CLI examples work
[ ] Server examples work
[ ] API documentation matches implementation
[ ] Architecture documentation matches code
[ ] Memory documentation is technically accurate
[ ] Safetensors documentation matches implementation
[ ] Contributor guide is complete
[ ] Testing guide is complete
[ ] Security guide exists
[ ] Code of Conduct exists
[ ] Changelog exists
[ ] Issue templates exist
[ ] PR template exists
[ ] CI documentation is accurate
[ ] ADRs exist
[ ] Glossary exists
[ ] Roadmap exists
[ ] No undocumented major architectural behavior
[ ] No documentation claims unsupported features
```

---

 # 62\. IMPORTANT FINAL INSTRUCTION

 Do not just write documentation files.

 First inspect the actual repository and implementation.

 Then generate documentation based on what actually exists.

 If you discover architectural problems, inconsistencies, missing interfaces, undocumented behavior, or implementation/documentation mismatches:

 1. Identify them.
2. Fix documentation where documentation is wrong.
3. Fix code only when necessary for the documented public behavior.
4. Clearly report any architectural issue that requires a larger implementation change.

 The final repository should feel like a serious open-source project where an external developer can clone it, understand it, modify it, test it, and contribute safely.

 At the end provide:

```
1. Complete documentation tree
2. Files created
3. Files modified
4. Documentation architecture
5. Contributor workflow
6. CI/documentation checks
7. Any unresolved gaps
8. Exact commands to build/test the documentation
9. Exact commands to run the project
10. Exact commands for a contributor to create a branch, test changes, commit, and prepare a PR
```

 Do not stop at an outline.

 Create the actual documentation files in the repository.