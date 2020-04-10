import numpy as np
import math
from matplotlib import pyplot as plt

def acha_u(c_2, n_t, n_x, x_c):
    # Acha o valor da equação de onda u discretizada em t e x
    f = np.zeros((n_t + 1, n_x + 1))
    j_c = int(x_c * n_x)
    for i in range(n_t + 1):
        f[i][j_c] = 1000 * c_2 * (1 - 2 * (10 * i / n_t * math.pi)**2 * np.exp(-(10 * i / n_t * math.pi)**2))
        u = np.zeros((n_t + 1, n_x + 1))
        alfa_2 = c_2 * (n_x / n_t)**2
        for i in range(1, n_t):
            for j in range(1, n_x):
                try:
                    u[i + 1][j] = -u[i - 1][j] + 2 * (1 - alfa_2) * u[i][j] + alfa_2 * 2 * (u[i][j + 1] + u[i][j - 1]) + (1 / n_t)**2 * f[i][j]
                except:
                    u[i + 1][j] = 0.0
    return u

def main():
    c_2 = 10
    n_t = 350
    n_x = 1000
    x_c = 0.7

    u = acha_u(c_2, n_t, n_x, x_c)
    x = []
    for j in range(n_x + 1):
        x.append(j / n_x)
    t0 = 0.1
    i = int(t0 * n_t)
    plt.plot(x, u[i])
    plt.show()

main()
    