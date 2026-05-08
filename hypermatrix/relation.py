"""
Relation: a directed inter-cluster connection — the "white matter" of the
federation. Learns a small linear adapter between source and target spaces and
strengthens (myelinates) with use.

Relations are first-class objects, separate from the clusters they connect.
This is the structural difference from a normal deep net: the topology of
connections is itself a growable, learned artifact.
"""

import math

from .numeric import matrand, matvec, project, vscale, vsub


class Relation:
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.adapter = matrand(target.dim_in, source.dim_out, scale=0.05)
        self.bandwidth = 0.1  # fraction of signal that crosses; grows with use
        self.use_count = 0

    def transmit(self, src_out):
        x = matvec(self.adapter, src_out)
        return [self.bandwidth * v for v in x]

    def reinforce(self, src_out, residual, lr=0.05):
        """
        Strengthen the relation. `residual` is the target-space signal we
        wanted this relation to deliver. Adapter moves toward producing it,
        and bandwidth grows.
        """
        self.use_count += 1
        self.bandwidth = min(1.0, 0.1 + math.log1p(self.use_count) * 0.05)

        produced = matvec(self.adapter, src_out)
        err = vsub(residual, [self.bandwidth * v for v in produced])
        for i in range(len(self.adapter)):
            for j in range(len(self.adapter[i])):
                self.adapter[i][j] += lr * err[i] * src_out[j] * self.bandwidth

    def describe(self):
        return (
            f"Rel({self.source.id}->{self.target.id}, "
            f"bw={self.bandwidth:.2f}, uses={self.use_count})"
        )
