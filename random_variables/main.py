import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

# -------------------1-------------------

X = np.random.choice([1, 2, 3, 4], 100)

# задача - посчитать вероятности p_1,..., p_4
p1 = (X == 1).sum() / 100
p2 = (X == 2).sum() / 100
p3 = (X == 3).sum() / 100
p4 = (X == 4).sum() / 100

print("p1={}, p2={}, p3={}, p4={}".format(p1, p2, p3, p4))

# -------------------2-------------------

# выбрать распределение https://docs.scipy.org/doc/scipy-0.16.1/reference/stats.html#continuous-distributions
random_var = sts.chi(.5, .5)
sample_size = 100
# выберите подходящий масштаб
x = random_var.rvs(size=sample_size)
cdf = random_var.cdf(x)
plt.plot(x, cdf, label='theoretical CDF')

from statsmodels.distributions.empirical_distribution import ECDF

ecdf = ECDF(x)
plt.step(ecdf.x, ecdf.y, label='ECDF')

plt.ylabel('$f(x)$')
plt.xlabel('$x$')
plt.legend(loc='upper left')

# -------------------3-------------------

mu, sigma = 0.0, 0.1
# зададим нормально распределенную случайную величину
norm_rv = sts.norm(loc=mu, scale=sigma)
# сгенерируем 10 значений
X = norm_rv.rvs(size=10)

# найдите квантиль уровня p = 5% по выборке
F = norm_rv.cdf(X).tolist()
filtered = [i for i in range(len(F)) if F[i] >= .005]
p005_empirical = min(map(lambda i: X[i], filtered))

# найдите теоретическое значение квантили
p005_theoretical = np.quantile(X, .05)

print("p005_empirical={}, p005_theoretical={}".format(p005_empirical, p005_theoretical))

# -------------------4-------------------

gilbrat_rv = sts.gilbrat(0.2, 0.2)
X = gilbrat_rv.rvs(100)

plt.hist(X, density=True)
plt.ylabel('fraction of samples')
plt.xlabel('$x$')
# -------------------5-------------------

results = []
n = 100
exp_rv = sts.expon(1)
for i in range(10000):
    x = exp_rv.rvs(n)
    results.append(x.mean())

plt.hist(results, density=True)
plt.ylabel('fraction of samples')
plt.xlabel('$x$')
plt.show()