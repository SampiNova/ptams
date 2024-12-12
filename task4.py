import numpy as np
import matplotlib.pyplot as plt


def task1(lam, n, N):
    counts, distances = [], []
    for _ in range(N):
        p = min(lam / n, 1)
        test = np.random.choice([0, 1], p=[1 - p, p], size=n)
        cnt  = np.count_nonzero(test)
        idx = np.where(test == 1)[0]
        dist = list(np.diff(idx) / n)
        counts.append(cnt)
        distances += dist
    return counts, distances

def task2(lam, N):
    counts, distances = [], []
    for _ in range(N):
        randomX = np.random.poisson(lam)
        points = sorted([np.random.uniform(0, 1) for _ in range(randomX)])
        dist = list(np.diff(points))
        counts.append(randomX)
        distances += dist
    return counts, distances

def task3(lam, N):
    counts, distances = [], []
    for _ in range(N):
        u = np.random.uniform(0, 1, size=int(lam * 1.5))
        inter_arrival_times = -np.log(u / lam) / lam

        event_times = np.cumsum(inter_arrival_times)
        event_times = event_times[event_times <= 1]

        cnt = len(event_times)
        counts.append(cnt)
        distances += list(np.diff(event_times))

    return counts, distances


L = 10
Num = 100000
c1, d1 = task1(L, 77, Num)
c2, d2 = task2(L, Num)
c3, d3 = task3(L, Num)

plt.subplot(3, 2, 1)
plt.hist(c1, bins=50, color='g', alpha=0.7, edgecolor='k', linewidth=1)
plt.subplot(3, 2, 2)
plt.hist(d1, bins=50, color='b', alpha=0.7, edgecolor='k', linewidth=1)

plt.subplot(3, 2, 3)
plt.hist(c2, bins=50, color='g', alpha=0.7, edgecolor='k', linewidth=1)
plt.subplot(3, 2, 4)
plt.hist(d2, bins=50, color='b', alpha=0.7, edgecolor='k', linewidth=1)

plt.subplot(3, 2, 5)
plt.hist(c3, bins=50, color='g', alpha=0.7, edgecolor='k', linewidth=1)
plt.subplot(3, 2, 6)
plt.hist(d3, bins=50, color='b', alpha=0.7, edgecolor='k', linewidth=1)

plt.show()
