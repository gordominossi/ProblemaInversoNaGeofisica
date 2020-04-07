from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import math

def plot(raizes, formatação, nome, coeficientes):

    reais = list(map(lambda raiz : raiz.real, raizes))
    imaginários = list(map(lambda raiz : raiz.imag, raizes))

    plt.plot(reais, imaginários, formatação)

    plt.title(str(np.poly1d(coeficientes)), loc='left', family='monospace')
    plt.xlabel("Eixo real", family='monospace')
    plt.ylabel("Eixo imaginário", family='monospace')
    plt.legend([nome])

    plt.savefig(nome + str(coeficientes) + ".png")
    plt.show()

# Ex1
def Ex1():
    def uj_iplus1(uj_i, uj_iminus1, ujplus1_i, ujminus1_i, alpha2, deltaT2, fj_i):
        try:
            return -uj_iminus1 + 2.0 * (1.0 - alpha2) * uj_i + alpha2 * (ujplus1_i + ujminus1_i) + deltaT2 * fj_i
        except:
            print(fj_i)
    
    def calculaU(T, n_t, n_x, c2, f):
        deltaT = T / n_t
        deltaX = 1.0 / n_x
        alpha2 = c2 / (deltaT / deltaX * deltaT / deltaX)

        U = np.empty((n_t + 1, n_x + 1))

        for j in range(n_x + 1):
            U[0][j] = 0.0
            U[1][j] = 0.0
        
        for i in range(1, n_t):
            t_i = i * deltaT
            U[i + 1][0] = 0.0
            for j in range(1, n_x):
                x_j = j * deltaX
                fj_i = f(t_i, x_j)
                if(fj_i > 0.00001): print(fj_i)
                U[i + 1][j] = uj_iplus1(U[i][j], U[i-1][j], U[i][j+1], U[i][j-1], alpha2, deltaT * deltaT, fj_i)
            U[i + 1][n_x] = 0.0
        
        return U

    T = 1

    c2 = 10.0
    n_x = int(1 / 0.1)

    x_c = 0.7
    beta = 10.0
    def f(t, x):
        if(abs(x - x_c) > 0.0001): return 0.0
        ret = 1000.0 * c2 * (1.0 - 2.0 * beta * beta * t * t * math.pi * math.pi) * math.exp(-1 * beta * beta * t * t * math.pi * math.pi)
        if np.isnan(ret) or np.isinf(abs(ret)): return 0.0
        return ret
    
    n_t = 100
    U = calculaU(T, n_t, n_x, c2, f)

Ex1()
