import yaml
import random
from collections import defaultdict, deque

YAML_FILE = "config.yaml"

def generate_connected_random_graph(n, extra_edges=1):
    edges = set()

    # 1️⃣ Tạo spanning tree (đảm bảo liên thông)
    nodes = list(range(n))
    random.shuffle(nodes)

    for i in range(1, n):
        a = nodes[i]
        b = random.choice(nodes[:i])
        edges.add(tuple(sorted((a, b))))

    # 2️⃣ Thêm random edges
    attempts = n * extra_edges
    for _ in range(attempts):
        a, b = random.sample(range(n), 2)
        edges.add(tuple(sorted((a, b))))

    return [list(e) for e in edges]


with open(YAML_FILE, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

n = cfg["network"]["nodes"]

if cfg["network"].get("topology") == "random":
    cfg["network"]["additional_connections"] = generate_connected_random_graph(
        n, extra_edges=2
    )

with open(YAML_FILE, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f, sort_keys=False)

print(f"Đã sửa YAML – graph random nhưng LIÊN THÔNG với {n} nodes")
