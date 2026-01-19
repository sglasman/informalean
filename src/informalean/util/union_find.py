from typing import Callable


class UnionFind:
    def __init__(self, n: int):
        self.parents = list(range(n))
        self.n = n
        self.sizes = [1 for _ in range(n)]

    def find(self, i: int) -> int:
        if self.parents[i] == i:
            return i
        else:
            root = self.find(self.parents[i])
            self.parents[i] = root
            return root

    def union(self, i: int, j: int) -> None:
        i_root = self.find(i)
        j_root = self.find(j)
        if i_root == j_root:
            return
        if self.sizes[i_root] < self.sizes[j_root]:
            self.parents[i_root] = j_root
            self.sizes[j_root] = self.sizes[j_root] + self.sizes[i_root]
        else:
            self.parents[j_root] = i_root
            self.sizes[i_root] = self.sizes[i_root] + self.sizes[j_root]

    def run(self, get_partners: Callable[[int], list[int]]) -> None:
        for i in range(self.n):
            for j in get_partners(i):
                self.union(i, j)
        for i in range(self.n):
            # Ensure path compression is done for all i
            self.find(i)

    def n_components(self) -> int:
        return len([i for i in range(self.n) if self.parents[i] == i])

    def singleton_fraction(self) -> float:
        return (
            float(
                len(
                    [
                        i
                        for i in range(self.n)
                        if self.sizes[i] == 1 and self.parents[i] == i
                    ]
                )
            )
            / self.n
        )

    def top_components(self, k: int) -> list[(int, int)]:
        all_components = [(self.sizes[i], i) for i in range(self.n) if self.parents[i] == i]
        return sorted(all_components, reverse=True)[:k]
    
    def component_of_root(self, root: int) -> list[int]:
        return [i for i in range(self.n) if self.parents[i] == root]
