"""
StemCluster: an undifferentiated computational substrate that takes shape from
the task it is first asked to perform — like a stem cell differentiating
based on its neighborhood.

Differentiation picks an internal "kind" (different architectural dynamics,
not just different weights). Use reinforces the cluster: weights move toward
fitting the task, the signature drifts toward the inputs it sees, and a
"myelination" factor scales effective bandwidth.
"""

import math

from .numeric import (
    cosine,
    matrand,
    matvec,
    relu,
    sigmoid,
    tanh,
    topk_mask,
    vadd,
    vnorm,
    vrand,
    vscale,
    vsub,
    vzeros,
)


KINDS = ("dense", "sparse", "gated", "recurrent")


class StemCluster:
    _next_id = 0

    def __init__(self, dim_in, dim_out, hidden=None):
        self.id = StemCluster._next_id
        StemCluster._next_id += 1

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden = hidden if hidden is not None else max(8, dim_in)

        self.W1 = matrand(self.hidden, dim_in)
        self.b1 = vzeros(self.hidden)
        self.W2 = matrand(dim_out, self.hidden)
        self.b2 = vzeros(dim_out)

        self.kind = None
        self.signature = vrand(dim_in)
        self.use_count = 0
        self.myelination = 1.0
        self._state = vzeros(self.hidden)  # used only by 'recurrent' kind

    def differentiate(self, task_signal):
        """Take shape from a task signal. Picks an architectural kind."""
        h = sum(abs(x) for x in task_signal)
        self.kind = KINDS[int(h * 1000) % len(KINDS)]
        self.signature = vnorm(list(task_signal))

    def _hidden_activation(self, z):
        if self.kind == "dense" or self.kind is None:
            return tanh(z)
        if self.kind == "sparse":
            return topk_mask(relu(z), max(1, self.hidden // 4))
        if self.kind == "gated":
            half = self.hidden // 2
            value = tanh(z[:half]) if half else []
            gate = sigmoid(z[half:]) if half else []
            gated = [v * g for v, g in zip(value, gate)]
            # pad back to hidden width
            return gated + [0.0] * (self.hidden - len(gated))
        if self.kind == "recurrent":
            mixed = [0.5 * a + 0.5 * b for a, b in zip(z, self._state)]
            out = tanh(mixed)
            self._state = out
            return out
        return tanh(z)

    def forward(self, x):
        z1 = [s + b for s, b in zip(matvec(self.W1, x), self.b1)]
        h = self._hidden_activation(z1)
        out = [s + b for s, b in zip(matvec(self.W2, h), self.b2)]
        return [v * self.myelination for v in out]

    def reinforce(self, x, target, lr=0.05):
        """Crude online update toward target. Strengthens with use."""
        self.use_count += 1
        self.myelination = 1.0 + math.log1p(self.use_count) * 0.05

        z1 = [s + b for s, b in zip(matvec(self.W1, x), self.b1)]
        h = self._hidden_activation(z1)
        out = [s + b for s, b in zip(matvec(self.W2, h), self.b2)]
        err = vsub(target, out)

        for i in range(self.dim_out):
            for j in range(self.hidden):
                self.W2[i][j] += lr * err[i] * h[j]
            self.b2[i] += lr * err[i]

        # backprop one step into W1 with simple linear approximation
        h_err = matvec(list(zip(*self.W2)), err) if self.W2 else vzeros(self.hidden)
        h_err = list(h_err)
        for j in range(self.hidden):
            for k in range(self.dim_in):
                self.W1[j][k] += lr * 0.5 * h_err[j] * x[k]
            self.b1[j] += lr * 0.5 * h_err[j]

        # signature drifts toward inputs the cluster is being asked to handle
        self.signature = vnorm(vadd(vscale(self.signature, 0.95), vscale(vnorm(x), 0.05)))
        return sum(e * e for e in err)

    def affinity(self, x):
        """How well this cluster 'fits' an input. Used for routing."""
        return cosine(self.signature, x)

    def describe(self):
        return (
            f"Stem#{self.id}(kind={self.kind}, uses={self.use_count}, "
            f"myel={self.myelination:.2f})"
        )
