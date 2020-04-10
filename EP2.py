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
#   function f(t, x)
    def f(t, x):
#       f constants
        x_c = 0.7
        pi = math.pi
        beta2 = 100.0

#       return 0 when x != x_c
        if(abs(x - x_c) > 0.0001): return 0.0

#       calculate f(t, x)
        functionF = 1000.0 * c2 * (1.0 - (2.0 * beta2 * (t**2) * (pi**2))) * math.exp(-1 * beta2 * (t**2) * (pi**2))

#       if f = nan | inf, returns 0.0, other case return calculated value
        if np.isnan(functionF) or np.isinf(abs(functionF)): return 0.0
        return functionF

#   function uj_iplus1
    def uj_iplus1(uj_i, uj_iminus1, ujplus1_i, ujminus1_i, alpha2, deltaT2, fj_i):
#       try calculate uj_iplus1, in case of error, print( Error ${f(fj_i)} )
        print('uj_i', uj_i)
        print('uj_iminus1', uj_iminus1)
        print('ujplus1_i', ujplus1_i)
        print('ujminus1_i', ujminus1_i)
        print('alpha2', alpha2)
        print('deltaT2', deltaT2)
        print('fj_i', fj_i)

        try:
            return -1 * uj_iminus1 + (2.0 * (1.0 - alpha2) * uj_i) + (alpha2 * (ujplus1_i + ujminus1_i)) + (deltaT2 * fj_i)
        except:
            print('Error', fj_i)

    def calculaU(T, n_t, n_x, c2, f):
#       u calculable variables
        deltaT = T / n_t
        deltaT2 = T * T / n_t * n_t
        deltaX = 1.0 / n_x
        alpha2 = c2 / ((deltaT * deltaT) / (deltaX * deltaX))

#       lenght of t and x dimensions
        lengthT = n_t + 1
        lengthX = n_x + 1

#       defines U[i][j]
        U = np.empty((lengthT, lengthX))

#       u[0 | 1][j] = 0.0
        for j in range(1, lengthX - 1):
            U[0][j] = 0.0
            U[1][j] = 0.0

#       runs t by i and calculates t_i
        for i in range(1, lengthT - 1):
            t_i = i * deltaT

#           runs x by j, and calculates x_j
            for j in range(1, lengthX - 1):
                x_j = j * deltaX

#               calculates f(t, x)
                fj_i = f(t_i, x_j)

#               calculates U[i+1][j]
                U[i + 1][j] = uj_iplus1(U[i][j], U[i-1][j], U[i][j+1], U[i][j-1], alpha2, deltaT2, fj_i)
#                 print('U[I+1][J]', U[i + 1][j])

#           fixed 0 when x = 0 or x = n_x
            U[i + 1][0] = 0.0
            U[i + 1][n_x] = 0.0
        
        return U

#   U constants
    T = 1
    n_x = 10 # 1/n_x = 0.01
    n_t = 10
    c2 = 10.0

    U = calculaU(T, n_t, n_x, c2, f)

    deltaT = T / n_t
    deltaX = 0.1 / n_x

    for i in range(1, n_t):
        t_i = i * deltaT

        for j in range(1, n_x):
            x_j = j * deltaX
#             print('t_i', t_i)
#             print('x_j', x_j)
#             print('U[t_i][x_j]', U[i+1][j+1])
    return U




print(Ex1())
