"""
Exemplo simplificado de Euler-Maruyama extraído de Process.ipynb

Este script simula uma EDE genérica:
 dX = a(X,t) * dt + b(X,t) * dW
com o método de Euler-Maruyama.

Salva a primeira trajetória em `euler_path.csv`.
"""
import numpy as np


def euler_maruyama(a, b, x0, T=1.0, N=1000, rng=None):
    """Simula uma trajetória com Euler-Maruyama.

    a, b: funções a(x,t) e b(x,t)
    x0: valor inicial
    T: horizonte de tempo
    N: número de passos
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = T / N
    t = np.linspace(0, T, N + 1)
    X = np.empty(N + 1)
    X[0] = x0

    for i in range(N):
        ti = t[i]
        dW = rng.normal(0.0, np.sqrt(dt))
        X[i + 1] = X[i] + a(X[i], ti) * dt + b(X[i], ti) * dW

    return t, X


def demo_gbm():
    # exemplo: Geometric Brownian Motion em forma EM (log-step simplificado)
    mu = 0.05
    sigma = 0.2

    def a(x, t):
        return mu * x

    def b(x, t):
        return sigma * x

    t, X = euler_maruyama(a, b, x0=100.0, T=1.0, N=1000)
    # salva apenas a trajetória
    np.savetxt("euler_path.csv", X, delimiter=",")
    print("Simulação concluída. Trajetória salva em euler_path.csv")


if __name__ == "__main__":
    demo_gbm()
