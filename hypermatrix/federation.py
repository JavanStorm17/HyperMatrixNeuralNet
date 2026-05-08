"""
Federation: the network of networks.

A Federation owns a set of StemClusters and the Relations between them. It
routes incoming (input, task_signal) pairs to the most relevant cluster — or
spawns a brand-new cluster when the input is novel. Pathways used together
strengthen together (Hebbian-style on the relations).

This is the part that gives the system its "many networks, related by tasks"
shape. The federation is *grown*; it is not pre-designed.
"""

from .novelty import NoveltyDetector
from .numeric import project, vadd, vscale
from .relation import Relation
from .stem import StemCluster


class Federation:
    def __init__(
        self,
        dim_in,
        dim_out,
        hidden=16,
        affinity_threshold=0.4,
        max_clusters=64,
    ):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden = hidden
        self.max_clusters = max_clusters

        self.clusters = []
        self.relations = []  # list[Relation]
        self.novelty = NoveltyDetector(affinity_threshold=affinity_threshold)
        self.history = []  # log of what happened, for inspection

    # ---------------- growth ----------------

    def _spawn(self, task_signal):
        c = StemCluster(self.dim_in, self.dim_out, hidden=self.hidden)
        c.differentiate(task_signal)
        self.clusters.append(c)

        # Wire the new cluster to its nearest neighbor (if any). This is the
        # "white matter" forming on cluster birth: a new region wires into the
        # existing brain through its closest topological neighbor.
        neighbor, score = self.novelty.best_match(self.clusters[:-1], task_signal)
        if neighbor is not None:
            self.relations.append(Relation(neighbor, c))
            self.relations.append(Relation(c, neighbor))
        return c

    def _ensure_capacity(self):
        if len(self.clusters) > self.max_clusters:
            # Soft cap: prune the least-used cluster (and its relations).
            victim = min(self.clusters, key=lambda c: c.use_count)
            self.clusters.remove(victim)
            self.relations = [
                r for r in self.relations if r.source is not victim and r.target is not victim
            ]

    # ---------------- inference ----------------

    def _incoming_to(self, target, x_for_source):
        """Sum messages arriving at `target` from connected sources."""
        total = [0.0] * target.dim_in
        for r in self.relations:
            if r.target is not target:
                continue
            src_out = r.source.forward(x_for_source)
            msg = r.transmit(src_out)
            msg = project(msg, target.dim_in)
            total = vadd(total, msg)
        return total

    def predict(self, x, task_signal):
        """
        Forward pass. If no cluster matches, spawn one (so the system can never
        be 'stuck' — novelty *creates* the substrate it needs).
        """
        novel, match, score = self.novelty.is_novel(self.clusters, task_signal)
        if novel:
            match = self._spawn(task_signal)
            self._ensure_capacity()

        incoming = self._incoming_to(match, x)
        # Combine direct input with relational context (white-matter signal).
        x_eff = [a + 0.5 * b for a, b in zip(x, incoming)]
        out = match.forward(x_eff)
        return out, match, score

    # ---------------- learning ----------------

    def step(self, x, task_signal, target, lr=0.05):
        """
        One on-line learning step. Routes, possibly grows, reinforces the
        chosen cluster, and strengthens the relations that fed into it.
        """
        out, cluster, score = self.predict(x, task_signal)
        loss = cluster.reinforce(x, target, lr=lr)

        # Reinforce relations that contributed by handing them the residual
        # (what the target cluster needed in *its* input space). This makes
        # frequently co-firing pathways stronger — the architecture's
        # Hebbian seam.
        residual_in = list(x)  # crude: we treat x itself as the desired input
        for r in self.relations:
            if r.target is cluster:
                src_out = r.source.forward(x)
                r.reinforce(src_out, residual_in, lr=lr)

        self.history.append(
            {
                "cluster": cluster.id,
                "kind": cluster.kind,
                "loss": loss,
                "match_score": score,
                "n_clusters": len(self.clusters),
                "n_relations": len(self.relations),
            }
        )
        return loss, cluster

    # ---------------- introspection ----------------

    def topology(self):
        return {
            "clusters": [c.describe() for c in self.clusters],
            "relations": [r.describe() for r in self.relations],
        }
