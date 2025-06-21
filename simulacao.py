import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def ler_parametros(arquivo):
    params = {}
    with open(arquivo, 'r') as f:
        for linha in f:
            if '=' in linha and not linha.startswith('#'):
                chave, valor = linha.split('=')
                params[chave.strip()] = float(valor.strip())
    parametros_obrigatorios = {'b2', 'b3', 'd1', 'd2', 'd3', 'a1', 'a2'}
    if not parametros_obrigatorios.issubset(params.keys()):
        missing = parametros_obrigatorios - set(params.keys())
        raise ValueError(f"Parâmetros obrigatórios faltando no arquivo: {missing}")
    
    return params

parametros = ler_parametros('parametros.txt') #coloque aqui o arquivo contendo os parametros da população a ser estudada

b2 = parametros['b2']
b3 = parametros['b3']
d1 = parametros['d1']
d2 = parametros['d2']
d3 = parametros['d3']
a1 = parametros['a1']
a2 = parametros['a2']

def build_matrix(b2, b3, d1, d2, d3, a1, a2):
    return np.array([
        [-(d1 + a1), b2, b3],
        [a1, -(d2 + a2), 0],
        [0, a2, -d3]
    ])

A = build_matrix(b2, b3, d1, d2, d3, a1, a2)

def population_system(t, x, A):
    return A @ x

def rk4(f, t_span, y0, dt, A):
    t0, tf = t_span
    n = int((tf - t0) / dt)
    t = np.linspace(t0, tf, n+1)
    y = np.zeros((len(y0), n+1))
    y[:, 0] = y0
    
    for i in range(n):
        k1 = f(t[i], y[:, i], A)
        k2 = f(t[i] + dt/2, y[:, i] + dt/2 * k1, A)
        k3 = f(t[i] + dt/2, y[:, i] + dt/2 * k2, A)
        k4 = f(t[i] + dt, y[:, i] + dt * k3, A)
        
        y[:, i+1] = y[:, i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, y

x0 = np.array([30, 60, 10])
t_span = (2000, 2100)
dt = 1

t, y_rk4 = rk4(population_system, t_span, x0, dt, A)

# visualização grafica
plt.figure(figsize=(12, 6))
plt.plot(t, y_rk4[0], label='Jovens')
plt.plot(t, y_rk4[1], label='Adultos')
plt.plot(t, y_rk4[2], label='Idosos')
plt.title('Evolução Populacional (2000-2100)')
plt.xlabel('Ano')
plt.ylabel('População (milhões)')
plt.legend()
plt.grid(True)
plt.show()

elderly_proportion = y_rk4[2] / np.sum(y_rk4, axis=0)
plt.figure(figsize=(10, 5))
plt.plot(t, elderly_proportion * 100)
plt.title('Proporção de Idosos na População Total')
plt.xlabel('Ano')
plt.ylabel('Percentual (%)')
plt.grid(True)
plt.show()