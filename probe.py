"""
Honest performance probe. Measures:

  1. Loss curve per task (does the federation actually learn each one?)
  2. Spawn vs reuse rate (is the novelty detector well-calibrated?)
  3. Held-out test loss after training (did it generalize, or just memorize?)
  4. Forgetting check: revisit early task at the end and see if it still works.
"""

import random
import statistics

from hypermatrix import Federation


DIM = 4


def task_a(x): return list(x)                # identity
def task_b(x): return [-v for v in x]        # negate
def task_c(x): return [2.0 * v for v in x]   # double
def task_d(x): return x[1:] + x[:1]          # shift

TASKS = {
    "identity": (task_a, [1.0, 0.0, 0.0, 0.0]),
    "negate":   (task_b, [0.0, 1.0, 0.0, 0.0]),
    "double":   (task_c, [0.0, 0.0, 1.0, 0.0]),
    "shift":    (task_d, [0.0, 0.0, 0.0, 1.0]),
}


def rand_x():
    return [random.uniform(-1.0, 1.0) for _ in range(DIM)]


def evaluate(fed, fn, task_id, n=100):
    losses = []
    for _ in range(n):
        x = rand_x()
        y = fn(x)
        signal = list(task_id)
        out, _, _ = fed.predict(x, signal)
        losses.append(sum((a - b) ** 2 for a, b in zip(out, y)))
    return statistics.mean(losses)


def baseline_zero(fn, n=200):
    """Trivial baseline: always predict zeros. Tells us what 'doing nothing' costs."""
    losses = []
    for _ in range(n):
        x = rand_x()
        y = fn(x)
        losses.append(sum(v * v for v in y))
    return statistics.mean(losses)


def main():
    random.seed(7)
    fed = Federation(dim_in=DIM, dim_out=DIM, hidden=16, affinity_threshold=0.5)

    schedule = [
        ("identity", 80),
        ("negate",   80),
        ("double",   80),
        ("shift",    80),
    ]

    # Track spawns per phase by watching cluster count.
    spawn_log = {}
    reuse_log = {}
    total = 0
    for phase, n in schedule:
        fn, task_id = TASKS[phase]
        before = len(fed.clusters)
        spawned = 0
        reused = 0
        for _ in range(n):
            x = rand_x()
            target = fn(x)
            signal = list(task_id)
            n_before = len(fed.clusters)
            fed.step(x, signal, target, lr=0.08)
            if len(fed.clusters) > n_before:
                spawned += 1
            else:
                reused += 1
        spawn_log[phase] = spawned
        reuse_log[phase] = reused
        total += n

    print("=" * 64)
    print(f"Trained on {total} examples across {len(schedule)} task phases.\n")

    print("Spawn vs reuse per task:")
    for phase, _ in schedule:
        s, r = spawn_log[phase], reuse_log[phase]
        print(f"  {phase:>10}  spawned={s:3d}  reused={r:3d}  "
              f"spawn_rate={s/(s+r):.2%}")

    print(f"\nFinal: {len(fed.clusters)} clusters, {len(fed.relations)} relations.")

    print("\nHeld-out test loss (lower = better, 0 = perfect):")
    print(f"  {'task':>10}  {'baseline':>10}  {'fed':>10}  {'reduction':>10}")
    for phase, _ in schedule:
        fn, task_id = TASKS[phase]
        baseline = baseline_zero(fn)
        fed_loss = evaluate(fed, fn, task_id)
        red = (baseline - fed_loss) / (baseline + 1e-9)
        print(f"  {phase:>10}  {baseline:>10.3f}  {fed_loss:>10.3f}  {red:>9.1%}")

    print("\nForgetting check (after all training, hit identity again):")
    fn, task_id = TASKS["identity"]
    before = evaluate(fed, fn, task_id)
    # 10 more reinforcement steps on identity — should be quick if remembered.
    for _ in range(10):
        x = rand_x()
        y = fn(x)
        signal = list(task_id)
        fed.step(x, signal, y, lr=0.08)
    after = evaluate(fed, fn, task_id)
    print(f"  identity loss before refresher = {before:.3f}")
    print(f"  identity loss after 10 refresh steps = {after:.3f}")


if __name__ == "__main__":
    main()
