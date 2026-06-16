# Variáveis do modelo numériro que podem ser modificadas pelo usuário:
#       nx, ny, nt, lx, ly, tf, tol, it_m, 
# Variáveis do modelo físico que podem ser modificadas pelo usuário:
#       ho, cp, k0, k1, T1, qs, ql, qn, T0, To

# Bibliotecas
import functions
import numpy as np
import copy as copy
import matplotlib.pyplot as plt

# Parâmetros do modelo numérico
nx = 60         # PAR: Número de pontos na direção x
ny = 60         # PAR: Número de pontos na direção y
nt = 10         # Número de pontos na direção t
lx = 4.0        # Comprimento do retângulo (domínio)
ly = 4.0        # Altura do retângulo (domínio)
tf = 2.0        # Tempo (final) da simulação
dx = lx/nx      # Tamanho do incremento na direção x
dy = ly/ny      # Tamanho do incremento na direção y
dt = tf/(nt-1)  # Tamanho do incremento na direção t
tol = 10**(-4)  # Tolerância ao erro
it_m = 100      # Número máximo de iterações

# Parametros do Modelo Físico: Condutiviade térmica - K(T) = k0*exp(k1*(T-T1))
rho = 1.0
cp = 3.0
k0 = 0.5
k1 = 0.01
T1 = 100.0
qs = 0.0
ql = 10.0
qn = 10.0

# Coeficientes do sistema algébrico
dxm1 = 1.0/dx
dym1 = 1.0/dy
dxm2 = dxm1/dx
dym2 = dym1/dy
a0 = rho*cp/dt
a2 = k0*dxm2/k1
a3 = k0*dym2/k1
a1 = 2.0*(a2+a3)

# Condições iniciais, condições de contorno, termo fonte e fonte de calor
T0 = 20.0
To = 100.0
Te = To*np.ones((ny,1))
T = T0*np.ones((ny,nx))
F = np.zeros((ny,nx))
Q = np.zeros((ny,nx))

# Estimativa inicial
T_n = T.copy()
T_n[:,0] = Te[:,0]
# Tensor de soluções
T_g = np.zeros((ny,nx,nt))
T_g[:,:,0] = T.copy()
x = np.linspace(0,lx,nx)
y = np.linspace(0,ly,ny)

# Varredura temporal
for k in range(1,nt):
    Q = functions.calcula_Q(Q,k*dt,x,y,nx,ny)
    F = a0*T + Q # Fonte fixa
    T_n = functions.solver(T_n,F,T1,nx,ny,a0,a1,a2,a3,k1,qs,qn,ql,dym1,dxm1,tol,it_m)
    T = T_n.copy()
    T_g[:,:,k] = T

# Saída Gráfica
fig, ax = plt.subplots()
extent = np.min(x), np.max(x), np.min(y), np.max(y)
cax = ax.imshow(T_g[:,:,nt-1], origin='lower', extent=extent,cmap='coolwarm') # 'magma', 'coolwarm' ou 'viridis' são ótimas opções
# 3. Adicione uma barra lateral para indicar a escala de valores
fig.colorbar(cax)

plt.title("Mapa de Calor")
plt.show()
