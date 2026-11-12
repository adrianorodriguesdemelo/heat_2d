# Bibliotecas
import numpy as np
import sys

# Parametros do Modelo Físico: Condutiviade térmica - K(T) = k0*exp(k1*(T-T1))
def parametros_fis():
    rho = 1.0
    cp = 3.0
    k0 = 0.5
    k1 = 0.01
    T1 = 100.0
    T0 = 20.0       # CI: Temperatura inicial
    To = 100.0      # CC Dirichlet: Temperatura no contorno esquerdo
    qs = 0.0        # CC Neumann: fluxo sul
    ql = 10.0       # CC Neumann: Fluxo leste
    qn = 10.0       # CC Neumann: Fluxo norte
    k0k1 = k0/k1
    return [rho,cp,k0,k1,T1,qs,ql,qn,k0k1,T0,To]

# Parâmetros do modelo numérico
def parametros_num():
    syc = 1         # ciclo: 0=SG; 1=MG
    ni1 = 2         # n de pre-suavisacoes
    ni2 = 2         # n de pos-suavisacoes
    solv = 1        # Solver: 0=Gaus-Seidel; 1=Gauss-Seidel-Red-Black
    nm = 5          # Número de malhas (>0)
    lx = 4.0        # Comprimento do retângulo (domínio)
    ly = 4.0        # Altura do retângulo (domínio)
    tf = 2.0        # Tempo (final) da simulação
    nt = 40         # Número de pontos na direção t: use 0 se nt = 2^nm+1
    tol = 10**(-7)  # Tolerância ao erro
    it_m = 1000      # Número máximo de iterações

    if (nt == 0): nt = 2**nm+1
    dt = tf/(nt-1)  # Tamanho do incremento na direção t

    # Multigrid
    npxy = []; npe = []
    for i in range(nm):
        npxy.append(2**(i+1)+1)
        npe.append(npxy[i]**2)
    npac = [0]; npet = [npe[0]*nt]          # numero de pontos no espaço/tempo na malha 1
    for i in range(1,nm):
        nx = 2**(i+1)+1; ny = nx
        npac.append(npet[i-1])              # numero de pontos totais na malha i-1
        npet.append(npac[i]+nx*ny*nt)       # numero de pontos totais na malha i
    if syc==0:      # Single-grid
        nx = 2**nm+1    # ÍMPAR: Número de pontos na direção x
        npac[0] = 0
        npxy[0] = nx
        npe[0] = nx*nx 
        npet[0] = npe[0]*nt

    if solv == 1:
        if ((nx % 2 != 1) or (ny % 2 != 1)):
            print('ATENÇÃO! Numero de pontos nx e ny devem ser impar (GAUSS-SEIDEL RED-BLACK)!')
            sys.exit()
    if (syc !=0 and syc !=1):
        print('ATENÇÃO! Ciclo invalido: use 0=SG ou 1=MG!')
        sys.exit()
    return [syc,solv,nt,nm,lx,ly,tf,dt,tol,it_m,npac,npet,npxy,npe,ni1,ni2]

def coeficientes(lx,ly,npxy,nm,dt,rho,cp,k0k1):
    # Sistema algebrico
    nx = npxy[nm]-1
    dx = lx/nx
    dy = ly/nx
    dxm1 = 1.0/dx
    dym1 = 1.0/dy
    aef = k0k1*dxm1*dxm1
    aeg = k0k1*dym1*dym1
    ap = rho*cp/dt
    apf = 2.0*aef
    apg = 2.0*aeg
    awf = aef
    awg = aeg
    aqs = -2.0*dym1
    aql = -2.0*dxm1
    aqn = aqs
    aNT = apf+apg
    return [ap,awf,aef,awg,aeg,apf,apg,aNT,aqs,aql,aqn]

# Insere as condicoes iniciais, condicoes de contorno e inicializa os vetores
def condicoes_ini_cont(T0,To,npet,npac,npe,npxy,nt,nm,lx,ly,dt,qs,ql,qn,syc):
    nx = 2**nm+1; ny = nx
    dxm1 = (nx-1)/lx; dym1 = (ny-1)/ly 
    aqs = -2.0*dym1
    aql = -2.0*dxm1
    aqn = aqs

    if syc==0:  # Single-Grid
        # Condições iniciais, condições de contorno, termo fonte e fonte de calor
        T = T0*np.ones((npet[0],1))     # Condicao inicial + estimativa inicial do waveform
        for i in range(1,nt):       # Contorno Dirichlet
            it = i*npe[0]
            for j in range(ny):
                k = j*nx+it
                T[k] = To

        # Fonte de calor
        x = np.linspace(0,lx,nx)
        y = np.linspace(0,ly,ny)
        Q = np.zeros((npet[0],1))
        # Inicializa vetor resizuo
        Ta = 0

        for k in range(1,nt):
            temp = k*dt
            it = k*npe[0]
            for i in range(ny):
                iy = it + i*nx
                for j in range(nx):
                    ix = iy + j
                    Q[ix] = 3.0*temp*np.exp(-temp)*(1+5*x[j]+y[i]+10*x[j]*y[i])
        # Ajusta o termo fonte a condicao de Neumann
        for k in range(1,nt):
            inis = npe[0]*k+1; fims = inis+nx-3
            fimn = npe[0]*(k+1)-2; inin = fimn-nx+3
            Q[inis:fims] = Q[inis:fims]+aqs*qs  # sul
            fims = fims+1
            Q[fims] = Q[fims]+aqs*qs+aql*ql         # sudeste
            Q[inin:fimn] = Q[inin:fimn]+aqn*qn  # norte
            Q[fimn+1] = Q[fimn+1]+aqn*qn+aql*ql         # nordeste
            for i in range(1,ny-1):                 # leste
                ip = fims+i*nx
                Q[ip] = Q[ip]+aql*ql
    else:   # Multigrid
        
        npemf = nx*ny
        # Condições iniciais, condições de contorno, termo fonte e fonte de calor
        ini = npac[nm-1]; fim = npet[nm-1]
        T = np.zeros((fim,1))
        T[npet[nm-2]:fim] = T0    # Condicao inicial + estimativa inicial do waveform
        
        for i in range(1,nt):       # Contorno Dirichlet
            it = ini+i*npemf
            for j in range(ny):
                k = j*nx+it
                T[k] = To
        
        # Fonte de calor
        x = np.linspace(0,lx,nx)
        y = np.linspace(0,ly,ny)
        Q = np.zeros((fim,1))
        # Vetor para armazenamento da solucao
        Ta = np.zeros((fim,1))

        for k in range(1,nt):
            temp = k*dt
            it = ini+k*npemf
            for i in range(ny):
                iy = it + i*nx
                for j in range(nx):
                    ix = iy + j
                    Q[ix] = 3.0*temp*np.exp(-temp)*(1+5*x[j]+y[i]+10*x[j]*y[i])
        
        # Ajusta o termo fonte a condicao de Neumann
        nxmg = npxy[nm-1]; nymg = nxmg
        npemg = nxmg*nxmg; img = npac[nm-1]
        for k in range(1,nt):
            inis = img+npemg*k+1; fims = inis+nxmg-2
            fimn = img+npemg*(k+1)-1; inin = fimn-nxmg+2
            Q[inis:fims-1] = Q[inis:fims-1]+aqs*qs  # sul
            Q[fims] = Q[fims]+aqs*qs+aql*ql         # sudeste
            Q[inin:fimn-1] = Q[inin:fimn-1]+aqn*qn  # norte
            Q[fimn] = Q[fimn]+aqn*qn+aql*ql         # nordeste
            for i in range(1,nymg-1):                 # leste
                ip = fims+i*nxmg
                Q[ip] = Q[ip]+aql*ql
    return [T,Q,Ta,x,y]