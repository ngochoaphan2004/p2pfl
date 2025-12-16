import random
from collections import defaultdict, deque

N = 50      # số node
K = 4       # số neighbor / node (3–5 là hợp lý)
SEED = 42   # để reproducible

random.seed(SEED)

edges = set()
adj = defaultdict(set)

# 1. Tạo backbone ring để ĐẢM BẢO LIÊN THÔNG
for i in range(N):
    j = (i + 1) % N
    edges.add((i, j))
    adj[i].add(j)
    adj[j].add(i)

# 2. Thêm random edges cho đủ K neighbors
for i in range(N):
    while len(adj[i]) < K:
        j = random.randrange(N)
        if j != i and j not in adj[i]:
            a, b = sorted((i, j))
            edges.add((a, b))
            adj[i].add(j)
            adj[j].add(i)

# 3. (Optional) kiểm tra liên thông
visited = set()
q = deque([0])
visited.add(0)
while q:
    u = q.popleft()
    for v in adj[u]:
        if v not in visited:
            visited.add(v)
            q.append(v)

assert len(visited) == N, "Graph KHÔNG liên thông — code có vấn đề"

# 4. In ra YAML-ready
print("additional_connections:")
for a, b in sorted(edges):
    print(f"  - [{a}, {b}]")
