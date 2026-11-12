# Variáveis do modelo numériro que podem ser modificadas pelo usuário:
#       nx, ny, nt, lx, ly, tf, tol, it_m, 
# Variáveis do modelo físico que podem ser modificadas pelo usuário:
#       ho, cp, k0, k1, T1, qs, ql, qn, T0, To

# Bibliotecas
import functions_wr
import dados_wr
import numpy as np
import copy as copy
import matplotlib.pyplot as plt

[rho,cp,k0,k1,T1,qs,ql,qn,k0k1,T0,To] = dados_wr.parametros_fis()
[syc,solv,nt,nm,lx,ly,tf,dt,tol_mg,tol_lin,it_mg,it_lin,npac,npet,npxy,npe,ni1,ni2] = dados_wr.parametros_num()
[ap,awf,aef,awg,aeg,apf,apg,aNT,aqs,aql,aqn] = dados_wr.coeficientes(lx,ly,npxy,nm-1,dt,rho,cp,k0k1)
[T,Q,Ta,x,y] = dados_wr.condicoes_ini_cont(T0,To,npet,npac,npe,npxy,nt,nm,lx,ly,dt,qs,ql,qn,syc)

if (syc == 1):  # multigrid
    T = functions_wr.multigrid(ni1,ni2,solv,T,Ta,Q,nm,npxy,npac,nt,npe,npet,ap,aef,awf,aeg,awg,k1,T1,aNT,dt,rho,cp,k0k1,lx,ly,tol_mg,tol_lin,it_mg,it_lin)
else:           # singlegrid
    T = functions_wr.single_grid(T,Q,npxy,nt,npe,npac,ap,aef,awf,aeg,awg,k1,T1,aNT,tol_mg,tol_lin,it_mg,it_lin,solv)

# Visualização gráfica
n = 0
if (syc == 1):
    n = nm-1
nx = npxy[n]
ny = nx
npe = npe[n]

Tg = np.zeros((ny,nx))
nptf = (nt-1)*npe+npac[n]
for i in range(ny):
    ip = nptf+i*nx
    for j in range(nx):
        jp = ip+j
        Tg[i][j] = T[jp][0]

print(Tg)
# Saída Gráfica
fig, ax = plt.subplots()
extent = np.min(x), np.max(x), np.min(y), np.max(y)
cax = ax.imshow(Tg, origin='lower', extent=extent,cmap='coolwarm') # 'magma', 'coolwarm' ou 'viridis' são ótimas opções
# 3. Adicione uma barra lateral para indicar a escala de valores
fig.colorbar(cax)

plt.title("Mapa de Calor")
plt.show()

