"""
StemCluster: an undifferentiated computational substrate that takes shape from
the task it is first asked to perform — like a stem cell differentiating
based on its neighborhood.

Differentiation picks an internal "kind" (different architectural dynamics,
not just different weights). Use reinforces the cluster: weights move toward
fitting the task via proper backprop, the signature drifts slowly toward the
task signals it serves, and a "myelination" counter records pathway use.
"""

import math

from .numeric import (
    cosine,
    matrand,
    matvec,
    vadd,
    vnorm,
    vrand,
    vscale,
    vsub,
    vzeros,
)


KINDS = ("dense", "sparse", "gated", "recurrent")


def _sigmoid_scalar(z):
    if z >= 0.0:
        ez = math.exp(-min(30.0, z))
        return 1.0 / (1.0 + ez)
    ez = math.exp(max(-30.0, z))
    return ez / (1.0 + ez)


class StemCluster:
    _next_id = 0

    def __init__(self, dim_in, dim_out, hidden=None):
        self.id = StemCluster._next_id
        StemCluster._next_id += 1

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden = hidden if hidden is not None else max(8, dim_in)

        # Xavier-ish init: scale by 1/sqrt(fan_in) so activations stay in range.
        self.W1 = matrand(self.hidden, dim_in, scale=1.0 / math.sqrt(max(1, dim_in)))
        self.b1 = vzeros(self.hidden)
        self.W2 = matrand(dim_out, self.hidden, scale=1.0 / math.sqrt(max(1, self.hidden)))
        self.b2 = vzeros(dim_out)

        self.kind = None
        self.signature = vrand(dim_in)
        self.use_count = 0
        self.myelination = 1.0
        self._state = vzeros(self.hidden)  # 'recurrent' kind only

    # ---------------- differentiation ----------------

    def differentiate(self, task_signal):
        """Take shape from a task signal. Picks an architectural kind."""
        # Hash that depends on each component's position, so distinct one-hot
        # signals don't all collapse to the same kind.
        h = sum((i + 1) * x for i, x in enumerate(task_signal))
        self.kind = KINDS[abs(int(h * 997)) % len(KINDS)]
        self.signature = vnorm(list(task_signal))

    # ---------------- forward / activation ----------------

    def _activate(self, z, update_state):
        """
        Apply this cluster's activation. Returns (h, dh_or_meta) where:
          dh_or_meta is per-element dh/dz vector for element-wise kinds, or
          a tuple ('gated', half, value, gate) for the gated kind so that
          backprop can route the gradient through the multiplicative gate.
        """
        if self.kind == "dense" or self.kind is None:
            h = [math.tanh(zi) for zi in z]
            dh = [1.0 - hi * hi for hi in h]
            return h, dh

        if self.kind == "sparse":
            relu_h = [zi if zi > 0.0 else 0.0 for zi in z]
            k = max(1, self.hidden // 2)
            idx = sorted(range(len(relu_h)), key=lambda i: -abs(relu_h[i]))[:k]
            keep = set(idx)
            h = [relu_h[i] if i in keep else 0.0 for i in range(len(relu_h))]
            dh = [1.0 if (i in keep and z[i] > 0.0) else 0.0 for i in range(len(z))]
            return h, dh

        if self.kind == "gated":
            half = self.hidden // 2
            value = [math.tanh(z[j]) for j in range(half)]
            gate = [_sigmoid_scalar(z[j + half]) for j in range(half)]
            gated = [value[j] * gate[j] for j in range(half)]
            h = gated + [0.0] * (self.hidden - half)
            return h, ("gated", half, value, gate)

        if self.kind == "recurrent":
            mixed = [0.5 * z[j] + 0.5 * self._state[j] for j in range(self.hidden)]
            h = [math.tanh(m) for m in mixed]
            # truncated backprop through time: treat state as constant w.r.t. z.
            dh = [0.5 * (1.0 - hi * hi) for hi in h]
            if update_state:
                self._state = list(h)
            return h, dh

        h = [math.tanh(zi) for zi in z]
        dh = [1.0 - hi * hi for hi in h]
        return h, dh

    def _compute(self, x, update_state):
        z1 = [s + b for s, b in zip(matvec(self.W1, x), self.b1)]
        h, dh_meta = self._activate(z1, update_state=update_state)
        out = [s + b for s, b in zip(matvec(self.W2, h), self.b2)]
        return z1, h, out, dh_meta

    def reset_state(self):
        self._state = vzeros(self.hidden)

    def forward(self, x, reset_state=True):
        # Default: each forward is independent. Sequence callers can pass
        # reset_state=False to let the recurrent kind carry memory across calls.
        if reset_state:
            self.reset_state()
        _, _, out, _ = self._compute(x, update_state=True)
        return out

    # ---------------- backprop ----------------

    def reinforce(self, x, target, lr=0.05, task_signal=None, reset_state=True):
        """
        One online gradient step: forward, compute MSE-like error, backprop
        with the correct activation derivative for this cluster's kind.

        Returns sum-of-squared-errors on the output (matching probe.py).
        """
        self.use_count += 1
        self.myelination = 1.0 + math.log1p(self.use_count) * 0.05

        if reset_state:
            self.reset_state()
        _, h, out, dh_meta = self._compute(x, update_state=True)
        err = vsub(target, out)  # negative gradient on out (loss = sum(err^2))

        # Update output layer: dL/dW2[i][j] = -2 * err[i] * h[j]
        for i in range(self.dim_out):
            ei = err[i]
            row = self.W2[i]
            for j in range(self.hidden):
                row[j] += lr * ei * h[j]
            self.b2[i] += lr * ei

        # Gradient on hidden post-activation: err_h[j] = sum_i W2[i][j] * err[i]
        err_h = [0.0] * self.hidden
        for i in range(self.dim_out):
            ei = err[i]
            row = self.W2[i]
            for j in range(self.hidden):
                err_h[j] += row[j] * ei

        # Convert to pre-activation gradient err_z using the right derivative.
        if isinstance(dh_meta, tuple) and dh_meta[0] == "gated":
            _, half, value, gate = dh_meta
            err_z = [0.0] * self.hidden
            for j in range(half):
                # h[j] = tanh(z[j]) * sigmoid(z[j+half]) = value[j] * gate[j]
                # dh[j]/dz[j]      = (1 - value[j]^2) * gate[j]
                # dh[j]/dz[j+half] = value[j] * gate[j] * (1 - gate[j])
                eh = err_h[j]
                err_z[j] = eh * (1.0 - value[j] * value[j]) * gate[j]
                err_z[j + half] = eh * value[j] * gate[j] * (1.0 - gate[j])
        else:
            dh = dh_meta  # element-wise vector
            err_z = [err_h[j] * dh[j] for j in range(self.hidden)]

        # Update hidden layer: dL/dW1[j][k] = -2 * err_z[j] * x[k]
        for j in range(self.hidden):
            ez = err_z[j]
            if ez == 0.0:
                continue
            row = self.W1[j]
            for k in range(self.dim_in):
                row[k] += lr * ez * x[k]
            self.b1[j] += lr * ez

        # Gentle signature drift toward the task this cluster is serving.
        # Drift toward task_signal (not raw input) so routing identity stays
        # task-aligned. Rate is small; birth signature dominates.
        if task_signal is not None:
            self.signature = vnorm(
                vadd(vscale(self.signature, 0.99), vscale(vnorm(task_signal), 0.01))
            )

        return sum(e * e for e in err)

    # ---------------- introspection ----------------

    def affinity(self, signal):
        return cosine(self.signature, signal)

    def describe(self):
        return (
            f"Stem#{self.id}(kind={self.kind}, uses={self.use_count}, "
            f"myel={self.myelination:.2f})"
        )
