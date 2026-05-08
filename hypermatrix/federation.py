"""
Federation: the network of networks.

A Federation owns a set of StemClusters and the Relations between them. It
routes incoming (input, task_signal) pairs to the most relevant cluster — or
spawns a brand-new cluster when the input is novel. Spawning a cluster also
wires it (in both directions) to its nearest existing neighbor, so the graph
of relations grows organically.

This iteration keeps relations as structural artifacts that strengthen with
use (Hebbian-style), but does NOT inject relational signals into the matched
cluster's input during forward. Random/under-trained adapters were corrupting
the gradient path. Functional relational composition is a future step; here
the goal is clean per-cluster learning + correct topology growth.
"""

from .novelty import NoveltyDetector
from .relation import Relation
from .stem import StemCluster


class Federation:
    def __init__(
        self,
        dim_in,
        dim_out,
        hidden=16,
        affinity_threshold=0.5,
        max_clusters=64,
    ):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden = hidden
        self.max_clusters = max_clusters

        self.clusters = []
        self.relations = []
        self.novelty = NoveltyDetector(affinity_threshold=affinity_threshold)
        self.history = []

    # ---------------- growth ----------------

    def _spawn(self, task_signal):
        c = StemCluster(self.dim_in, self.dim_out, hidden=self.hidden)
        c.differentiate(task_signal)
        self.clusters.append(c)

        # Wire the new cluster to its nearest neighbor (if any) — bidirectional.
        # Relations are first-class objects; their existence is the topology.
        neighbor, _ = self.novelty.best_match(self.clusters[:-1], task_signal)
        if neighbor is not None:
            self.relations.append(Relation(neighbor, c))
            self.relations.append(Relation(c, neighbor))
        return c

    def _ensure_capacity(self):
        if len(self.clusters) > self.max_clusters:
            victim = min(self.clusters, key=lambda c: c.use_count)
            self.clusters.remove(victim)
            self.relations = [
                r for r in self.relations
                if r.source is not victim and r.target is not victim
            ]

    def _route(self, task_signal):
        novel, match, score = self.novelty.is_novel(self.clusters, task_signal)
        if novel:
            match = self._spawn(task_signal)
            self._ensure_capacity()
            score = 1.0  # we just spawned for this exact signal
        return match, score

    # ---------------- inference ----------------

    def predict(self, x, task_signal):
        match, score = self._route(task_signal)
        out = match.forward(x)
        return out, match, score

    # ---------------- learning ----------------

    def step(self, x, task_signal, target, lr=0.05):
        """
        One online learning step. Routes (or grows), reinforces the chosen
        cluster, and Hebbian-updates the relations that connect to it (so
        co-firing pathways strengthen even though they don't yet inject
        signal during forward).
        """
        match, score = self._route(task_signal)
        loss = match.reinforce(x, target, lr=lr, task_signal=task_signal)

        # Hebbian relation use-count: every relation touching the matched
        # cluster increments its myelination. Adapters also nudge toward
        # representing the input the target just saw, so they're ready for
        # future functional use.
        for r in self.relations:
            if r.target is match:
                src_out = r.source.forward(x)
                r.reinforce(src_out, x, lr=lr * 0.5)

        self.history.append({
            "cluster": match.id,
            "kind": match.kind,
            "loss": loss,
            "match_score": score,
            "n_clusters": len(self.clusters),
            "n_relations": len(self.relations),
        })
        return loss, match

    # ---------------- introspection ----------------

    def topology(self):
        return {
            "clusters": [c.describe() for c in self.clusters],
            "relations": [r.describe() for r in self.relations],
        }
