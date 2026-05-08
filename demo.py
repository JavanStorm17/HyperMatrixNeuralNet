"""
Demo: stream a sequence of distinct synthetic 'tasks' at a Federation and
watch the network of networks grow itself.

Each task has a different (input -> target) rule and a different task-id
vector. The federation has no idea what's coming. We expect:

  * On first sight of a task, novelty triggers a new cluster.
  * Repeat exposure reinforces that cluster (loss falls, myelination rises).
  * Topology grows organically — relations form between neighbors.

Run with:  python3 demo.py
"""

import random

from hypermatrix import Federation


DIM = 4


def task_a(x):  # identity
    return list(x)


def task_b(x):  # negate
    return [-v for v in x]


def task_c(x):  # double
    return [2.0 * v for v in x]


def task_d(x):  # cyclic shift
    return x[1:] + x[:1]


TASKS = {
    "identity": (task_a, [1.0, 0.0, 0.0, 0.0]),
    "negate":   (task_b, [0.0, 1.0, 0.0, 0.0]),
    "double":   (task_c, [0.0, 0.0, 1.0, 0.0]),
    "shift":    (task_d, [0.0, 0.0, 0.0, 1.0]),
}


def random_input():
    return [random.uniform(-1.0, 1.0) for _ in range(DIM)]


def main():
    random.seed(7)
    fed = Federation(dim_in=DIM, dim_out=DIM, hidden=12, affinity_threshold=0.55)

    schedule = (
        [("identity", 30)]
        + [("negate",   30)]
        + [("identity", 10)]   # revisit — should reuse, not spawn
        + [("double",   30)]
        + [("shift",    30)]
        + [("negate",   10)]   # revisit
    )

    print(f"{'phase':>10} {'step':>5} {'cluster':>8} {'kind':>10} "
          f"{'loss':>8} {'score':>6} {'#C':>3} {'#R':>3}")

    step = 0
    for phase, n in schedule:
        fn, task_id = TASKS[phase]
        for _ in range(n):
            x = random_input()
            target = fn(x)
            signal = [a + b for a, b in zip(x, task_id)]  # input + task tag
            loss, cluster = fed.step(x, signal, target, lr=0.08)
            step += 1
            if step % 5 == 0:
                print(f"{phase:>10} {step:>5} {cluster.id:>8} "
                      f"{str(cluster.kind):>10} {loss:>8.3f} "
                      f"{fed.history[-1]['match_score']:>6.2f} "
                      f"{len(fed.clusters):>3} {len(fed.relations):>3}")

    print("\nFinal topology:")
    topo = fed.topology()
    for c in topo["clusters"]:
        print(" ", c)
    for r in topo["relations"]:
        print(" ", r)

    print(f"\nGrew {len(fed.clusters)} clusters and "
          f"{len(fed.relations)} relations across {step} steps.")


if __name__ == "__main__":
    main()
