import numpy as np
import matplotlib.pyplot as plt


def task1(lam, n, N):
    counts, distances = [], []
    for _ in range(N):
        p = min(lam / n, 1)
        test = np.random.choice([0, 1], p=[1 - p, p], size=n)
        cnt  = np.count_nonzero(test)
        idx = np.where(test == 1)[0]
        dist = list(np.diff(idx) * (1 / n))
        counts.append(cnt)
        distances += dist
    return counts, distances

def task2(lam, N):
    counts, distances = [], []
    for _ in range(N):
        randomX = np.random.poisson(lam)
        points = [np.random.uniform(0, 1) for _ in range(randomX)]
        dist = list(np.diff(points))
        counts.append(randomX)
        distances += dist
    return counts, distances


c1, d1 = task1(1, 10, 10000)
c2, d2 = task2(1, 10000)

plt.subplot(2, 2, 1)
plt.hist(c1, bins=50, color='g', alpha=0.7, edgecolor='k', linewidth=1)
plt.subplot(2, 2, 2)
plt.hist(d1, bins=50, color='b', alpha=0.7, edgecolor='k', linewidth=1)

plt.subplot(2, 2, 3)
plt.hist(c2, bins=50, color='g', alpha=0.7, edgecolor='k', linewidth=1)
plt.subplot(2, 2, 4)
plt.hist(d2, bins=50, color='b', alpha=0.7, edgecolor='k', linewidth=1)

plt.show()
