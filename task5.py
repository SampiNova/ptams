import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma

def Bp(p):
    return 1 if np.random.uniform(0, 1) < p else 0

def Bnp(n, p):
    return sum([Bp(p) for _ in range(n)])

def Cp(p):
    f = (lambda x: np.log(1 - x) / np.log(1 - p) - 1)
    return f(np.random.uniform(0, 1))

def Plam(lam):
    f = (lambda x: -(np.log(x / lam) / lam))
    return f(np.random.uniform(0, 1))

def Uab(a, b):
    f = (lambda x: x * (b - a) + a)
    return f(np.random.uniform(0, 1))

def ExpAlpha(alpha):
    f = (lambda x: -(np.log(1 - x) / alpha))
    return f(np.random.uniform(0, 1))

def LAlpha(alpha, mu=0):
    f = (lambda x: -alpha * np.log(2 * x))
    g = (lambda x: alpha * np.log(2 * (1 - x)))
    u = np.random.uniform(0, 1)
    return mu + (f(u) if u < 0.5 else g(u))

def Norm(alpha, sigma):
    u1 = np.random.uniform(0, 1)
    u2 = np.random.uniform(0, 1)
    R = np.sqrt(-2 * np.log(u1))
    teta = 2 * np.pi * u2
    z = R * np.cos(teta)
    return sigma * z + alpha

def Cauchy(x0, gamma):
    f = (lambda x: x0 + gamma * np.tan(np.pi * (x - 0.5)))
    return f(np.random.uniform(0, 1))

def MyGen():
    f = (lambda x: 1 / np.sqrt(1 - x))
    return f(np.random.uniform(0, 1))


def asymmetry(X):
    X = np.asarray(X)
    return np.mean((X - np.mean(X)) ** 3) / (np.var(X) ** (3 / 2))

def excess(X):
    X = np.asarray(X)
    return np.mean((X - np.mean(X)) ** 4) / (np.var(X) ** 2) - 3

objects = [[], [], [], [], [], [], [], [], [], [], [], []]
names = ["Bernoulli",
         "Binomial",
         "Geometric",
         "Poisson",
         "Uniform",
         "Exponential",
         "Laplace",
         "Normal",
         "Cauchy",
         "Kernel f(t) = 1/t^3 I{t > 1}",
         "Gamma",
         "Beta"]

N = 10000
for i in range(N):
    objects[0].append(Bp(0.7))
    objects[1].append(Bnp(1000, 0.7))
    objects[2].append(Cp(0.7))
    objects[3].append(Plam(7))
    objects[4].append(Uab(0, 1))
    objects[5].append(ExpAlpha(0.8))
    objects[6].append(LAlpha(0.8))
    objects[7].append(Norm(5, 2))
    objects[8].append(Cauchy(0.5, 0.2))
    objects[9].append(MyGen())
    objects[10].append(gamma.rvs(a=2, loc=1, scale=0.5))
    objects[11].append(beta.rvs(a=2, b=5))
    print(f"{round((i + 1) / N * 100, 2)}%")

print("Random generation done!")

xi = [[], [], [], [], [], [], [], [], [], [], [], []]
eta = [[], [], [], [], [], [], [], [], [], [], [], []]

N = 1000
n = 100
for i in range(N):
    for _ in range(n):
        xi[0].append(Bp(0.7))
        xi[1].append(Bnp(10, 0.7))
        xi[2].append(Cp(0.7))
        xi[3].append(Plam(7))
        xi[4].append(Uab(0, 1))
        xi[5].append(ExpAlpha(0.8))
        xi[6].append(LAlpha(0.8))
        xi[7].append(Norm(5, 2))
        xi[8].append(Cauchy(0.5, 0.2))
        xi[9].append(MyGen())
        xi[10].append(gamma.rvs(a=2, loc=1, scale=0.5))
        xi[11].append(beta.rvs(a=2, b=5))
    for j in range(12):
        mean_xi = np.mean(xi[j])
        var_xi = np.var(xi[j])
        eta[j].append((np.sum(xi[j]) - n * mean_xi) / np.sqrt(n * var_xi))
    print(f"{round((i + 1) / N * 100, 2)}%")


bns = [2, 50, 50, 50, 50, 50, 50, 50, 20, 50, 50, 50]

for i in range(len(objects)):
    plt.subplot(4, 6, i * 2 + 1)
    plt.title(names[i])
    plt.hist(objects[i], bins=bns[i], color='g', alpha=0.5, edgecolor='k', linewidth=1)
    plt.subplot(4, 6, i * 2 + 2)
    plt.title(names[i])
    plt.hist(eta[i], bins=50, color='b', alpha=0.5, edgecolor='k', linewidth=1)
    print(names[i], asymmetry(objects[i]), excess(objects[i]))

plt.show()
