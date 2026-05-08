"""
NoveltyDetector: decides whether an incoming (input, task) pair belongs to
existing clusters or warrants spawning a new one.

In a normal deep net, novel inputs just get pushed through fixed weights and
silently mis-handled. Here, novelty is the trigger for *structural* growth.
"""

from .numeric import cosine, vadd, vnorm


class NoveltyDetector:
    def __init__(self, affinity_threshold=0.4):
        # If the best matching cluster's affinity is below this, the input is
        # treated as novel and a new cluster is spawned for it.
        self.threshold = affinity_threshold

    def best_match(self, clusters, task_signal):
        """Return (cluster, score) for the existing cluster most aligned with the signal."""
        if not clusters:
            return None, -1.0
        scored = [(c, c.affinity(task_signal)) for c in clusters]
        scored.sort(key=lambda t: -t[1])
        return scored[0]

    def is_novel(self, clusters, task_signal):
        cluster, score = self.best_match(clusters, task_signal)
        if cluster is None:
            return True, None, score
        return score < self.threshold, cluster, score

    @staticmethod
    def task_signal(x, task_id_vec):
        """Combine an input with a task-id vector into a single signal."""
        return vnorm(vadd(vnorm(x), vnorm(task_id_vec)))
