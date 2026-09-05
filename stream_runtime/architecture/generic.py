import re
from ..graph.node import Node


class ArchitectureAdapter:
    def can_handle(self, config):
        return True

    def analyze(self, reader):
        names = reader.tensor_names()
        groups = {}
        for name in names:
            m = re.match(r"(.+?)(?:\.weight|\.bias)$", name)
            key = m.group(1) if m else name.rsplit(".", 1)[0]
            groups.setdefault(key, []).append(name)
        nodes = []
        for i, (key, weights) in enumerate(sorted(groups.items())):
            kind = "Linear" if any(n.endswith(".weight") for n in weights) else "Tensor"
            nodes.append(
                Node(
                    i,
                    key,
                    kind,
                    ["hidden" if i else "input"],
                    ["hidden"],
                    weights,
                    dependencies=[] if i == 0 else [i - 1],
                )
            )
        tensors = {
            n: {
                "dtype": reader.metadata(n).dtype,
                "shape": list(reader.metadata(n).shape),
                "offset": reader.metadata(n).data_start,
                "bytes": reader.metadata(n).nbytes,
            }
            for n in names
        }
        return nodes, tensors
