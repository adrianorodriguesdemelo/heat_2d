import numpy as np
import sys

# Solver de Newton
def resolve_newton(Tnf,T1f,fontef,k1f,a0f,a1f,tolf,it_mf):
    erro = 1; it = 0; Tn0 = Tnf
    while (erro > tolf and it < it_mf):
        G = a0f*Tnf+a1f*np.exp(k1f*(Tnf-T1f))
        div = 1.0/(a0f+k1f*a1f*np.exp(k1f*(Tnf-T1f)))
        inc = div*(fontef-G)
        Tnf = Tnf+inc
        erro = abs(inc)/abs(Tn0)
        it += 1
    return Tnf

def fonte(TpA,TpM1,Tpm1,TyM1,Tym1,Qf,aef,awf,aeg,awg,apff,k1f,T1):
    num = apff*TpA+aef*np.exp(k1f*(TpM1-T1))+awf*np.exp(k1f*(Tpm1-T1))+\
    aeg*np.exp(k1f*(TyM1-T1))+awg*np.exp(k1f*(Tym1-T1))+Qf
    return num

def solver_wr_gs(nuf,nmf,Tf,Qf,npxyf,ntf,npefv,npacf,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf,tolf,it_mf):
    nxyf = npxyf[nmf]
    ini = npacf[nmf]; npef = npefv[nmf]
    for s in range(nuf):
        # Sul
        for j in range(1,nxyf-1):
            jp = ini+j
            for k in range(1,ntf):
                kp = k*npef+jp; kpa = kp-npef
                #print("sul", kp, kpa)
                fon = fonte(Tf[kpa],Tf[kp+1],Tf[kp-1],Tf[kp+nxyf],Tf[kp+nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
                Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)
        # Sudeste
        jp = ini+nxyf-1
        for k in range(1,ntf):
            kp = k*npef+jp; kpa = kp-npef
            #print("sudeste", kp, kpa)
            fon = fonte(Tf[kpa],Tf[kp-1],Tf[kp-1],Tf[kp+nxyf],Tf[kp+nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
            Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)
        # Pontos internos
        for i in range(1,nxyf-1):
            ip = ini+i*nxyf
            for j in range(1,nxyf-1):
                jp = ip+j
                for k in range(1,ntf):
                    kp = k*npef+jp; kpa = kp-npef
                    #print("interno", kp, kpa)
                    fon = fonte(Tf[kpa],Tf[kp+1],Tf[kp-1],Tf[kp+nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
                    Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)
            # Ponto leste
            jp = ip+nxyf-1
            for k in range(1,ntf):
                kp = k*npef+jp; kpa = kp-npef
                #print("leste", kp, kpa)
                fon = fonte(Tf[kpa],Tf[kp-1],Tf[kp-1],Tf[kp+nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
                Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)
        # Norte
        ip = ini+(nxyf-1)*nxyf
        for j in range(1,nxyf-1):
            jp = ip+j
            for k in range(1,ntf):
                kp = k*npef+jp; kpa = kp-npef
                #print('norte',kp,kpa)
                fon = fonte(Tf[kpa],Tf[kp+1],Tf[kp-1],Tf[kp-nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
                Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)
        # Nordeste
        jp = ip+nxyf-1
        for k in range(1,ntf):
            kp = k*npef+jp; kpa = kp-npef
            #print('nordeste',kp,kpa)
            fon = fonte(Tf[kpa],Tf[kp-1],Tf[kp-1],Tf[kp-nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
            Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)
    return Tf

def solver_wr_gsrb(nuf,nmf,Tf,Qf,npxyf,ntf,npefv,npacf,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf,tolf,it_mf):
    nxyf = npxyf[nmf]
    ini = npacf[nmf]; npef = npefv[nmf]
    for s in range(nuf):
        # PONTOS RED
        # Sul
        for j in range(1,nxyf-1,2):
            jp = ini+j
            for k in range(1,ntf):
                kp = k*npef+jp; kpa = kp-npef
                #print("RED 1: sul", kp, kpa)
                fon = fonte(Tf[kpa],Tf[kp+1],Tf[kp-1],Tf[kp+nxyf],Tf[kp+nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
                Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)
        # Pontos internos
        for i in range(2,nxyf-1,2):
            ip = ini+i*nxyf
            for j in range(1,nxyf-1,2):
                jp = ip+j
                for k in range(1,ntf):
                    kp = k*npef+jp; kpa = kp-npef
                    #print("RED 1: interno", kp, kpa)
                    fon = fonte(Tf[kpa],Tf[kp+1],Tf[kp-1],Tf[kp+nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
                    Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)
        # Norte
        ip = ini+(nxyf-1)*nxyf
        for j in range(1,nxyf-1,2):
            jp = ip+j
            for k in range(1,ntf):
                kp = k*npef+jp; kpa = kp-npef
                #print('RED 1: norte',kp,kpa)
                fon = fonte(Tf[kpa],Tf[kp+1],Tf[kp-1],Tf[kp-nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
                Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)

        # Pontos internos
        for i in range(1,nxyf-1,2):
            ip = ini+i*nxyf
            for j in range(2,nxyf-1,2):
                jp = ip+j
                for k in range(1,ntf):
                    kp = k*npef+jp; kpa = kp-npef
                    #print('RED 2: interno',kp,kpa)
                    fon = fonte(Tf[kpa],Tf[kp+1],Tf[kp-1],Tf[kp+nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
                    Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)
            # Ponto leste
            jp = ip+nxyf-1
            for k in range(1,ntf):
                kp = k*npef+jp; kpa = kp-npef
                #print("RED 2: leste", kp, kpa)
                fon = fonte(Tf[kpa],Tf[kp-1],Tf[kp-1],Tf[kp+nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
                Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)
        
        # PONTOS BLACK
        # Sul
        for j in range(2,nxyf-1,2):
            jp = ini+j
            for k in range(1,ntf):
                kp = k*npef+jp; kpa = kp-npef
                #print("BLACK 1: sul", kp, kpa)
                fon = fonte(Tf[kpa],Tf[kp+1],Tf[kp-1],Tf[kp+nxyf],Tf[kp+nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
                Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)
        # Sudeste
        jp = ini+nxyf-1
        for k in range(1,ntf):
            kp = k*npef+jp; kpa = kp-npef
            #print("BLACK 1: sudeste", kp, kpa)
            fon = fonte(Tf[kpa],Tf[kp-1],Tf[kp-1],Tf[kp+nxyf],Tf[kp+nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
            Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)

        # Pontos internos
        for i in range(2,nxyf-1,2):
            ip = ini+i*nxyf
            for j in range(2,nxyf-1,2):
                jp = ip+j
                for k in range(1,ntf):
                    kp = k*npef+jp; kpa = kp-npef
                    #print("BLACK 1: interno", kp, kpa)
                    fon = fonte(Tf[kpa],Tf[kp+1],Tf[kp-1],Tf[kp+nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
                    Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)
            # Ponto leste
            jp = ip+nxyf-1
            for k in range(1,ntf):
                kp = k*npef+jp; kpa = kp-npef
                #print("BLACK 1: leste", kp, kpa)
                fon = fonte(Tf[kpa],Tf[kp-1],Tf[kp-1],Tf[kp+nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
                Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)

        # Norte
        ip = ini+(nxyf-1)*nxyf
        for j in range(2,nxyf-1,2):
            jp = ip+j
            for k in range(1,ntf):
                kp = k*npef+jp; kpa = kp-npef
                #print('BLACK 1: norte',kp,kpa)
                fon = fonte(Tf[kpa],Tf[kp+1],Tf[kp-1],Tf[kp-nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
                Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)
        # Nordeste
        jp = ip+nxyf-1
        for k in range(1,ntf):
            kp = k*npef+jp; kpa = kp-npef
            #print('BLACK 1: nordeste',kp,kpa)
            fon = fonte(Tf[kpa],Tf[kp-1],Tf[kp-1],Tf[kp-nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
            Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)

        # Pontos internos
        for i in range(1,nxyf,2):
            ip = ini+i*nxyf
            for j in range(1,nxyf-1,2):
                jp = ip+j
                for k in range(1,ntf):
                    kp = k*npef+jp; kpa = kp-npef
                    #print('BLACK 2: interno',kp,kpa)
                    fon = fonte(Tf[kpa],Tf[kp+1],Tf[kp-1],Tf[kp+nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
                    Tf[kp] = resolve_newton(Tf[kp],T1f,fon,k1f,apf,aNTf,tolf,it_mf)       
    
    return Tf

def residuo(Tf,Qf,nmf,npxyf,npacf,ntf,npefv,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf):
    
    nxyf = npxyf[nmf]
    rf = np.zeros((nxyf*nxyf*ntf,1))
    ini = npacf[nmf]; npef = npefv[nmf]
    # Sul
    for j in range(1,nxyf-1):
        jp = ini+j
        for k in range(1,ntf):
            kp = k*npef+jp; kpa = kp-npef
            pos = k*npef+j
            fon = fonte(Tf[kpa],Tf[kp+1],Tf[kp-1],Tf[kp+nxyf],Tf[kp+nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
            rf[pos] = fon - apf*Tf[kp]-aNTf*np.exp(k1f*(Tf[kp]-T1f))
    # Sudeste
    j = nxyf-1; jp = ini+j
    for k in range(1,ntf):
        kp = k*npef+jp; kpa = kp-npef
        pos = k*npef+j
        fon = fonte(Tf[kpa],Tf[kp-1],Tf[kp-1],Tf[kp+nxyf],Tf[kp+nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
        rf[pos] = fon - apf*Tf[kp]-aNTf*np.exp(k1f*(Tf[kp]-T1f))
    # Pontos internos
    for i in range(1,nxyf-1):
        ip = i*nxyf; im = ip+ini
        for j in range(1,nxyf-1):
            jp = ip+j; jm = im+j
            for k in range(1,ntf):
                kp = k*npef+jm; kpa = kp-npef
                pos = k*npef+jp
                fon = fonte(Tf[kpa],Tf[kp+1],Tf[kp-1],Tf[kp+nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
                rf[pos] = fon - apf*Tf[kp]-aNTf*np.exp(k1f*(Tf[kp]-T1f))
        # Ponto leste
        jp = ip+nxyf-1; jm = im+nxyf-1
        for k in range(1,ntf):
            kp = k*npef+jm; kpa = kp-npef
            pos = k*npef+jp
            fon = fonte(Tf[kpa],Tf[kp-1],Tf[kp-1],Tf[kp+nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
            rf[pos] = fon - apf*Tf[kp]-aNTf*np.exp(k1f*(Tf[kp]-T1f))
    # Norte
    ip = (nxyf-1)*nxyf; im = ip+ini
    for j in range(1,nxyf-1):
        jp = ip+j; jm = im+j
        for k in range(1,ntf):
            kp = k*npef+jm; kpa = kp-npef
            pos = k*npef+jp
            fon = fonte(Tf[kpa],Tf[kp+1],Tf[kp-1],Tf[kp-nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
            rf[pos] = fon - apf*Tf[kp]-aNTf*np.exp(k1f*(Tf[kp]-T1f))
    # Nordeste
    jp = ip+nxyf-1; jm = im+nxyf-1
    for k in range(1,ntf):
        kp = k*npef+jm; kpa = kp-npef
        pos = k*npef+jp
        fon = fonte(Tf[kpa],Tf[kp-1],Tf[kp-1],Tf[kp-nxyf],Tf[kp-nxyf],Qf[kp],aeff,awff,aegf,awgf,apf,k1f,T1f)
        rf[pos] = fon - apf*Tf[kp]-aNTf*np.exp(k1f*(Tf[kp]-T1f))

    return rf

# Coeficientes do sistema algébrico
def atualiza_coef(lx,ly,npxy,nm,dt,rho,cp,k0k1):
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
    aNT = apf+apg
    return [ap,awf,aef,awg,aeg,aNT]

def restrige_solucao(V,nm,npxy,npac,npt,nt):
    # Restringe um vetor (nele mesmo) da malha nm (FINA) para a malha nm-1 (GROSSA)
    ds = 0.0625; ds0 = 0.25; ds1=0.125
    nxmg = npxy[nm-1]
    nymg = nxmg;       nxmf = npxy[nm]
    npemf = nxmf*nxmf; npemg = nxmg*nxmg
    img = npac[nm-1];     imf = npt[nm-1]
    # Sul
    for l in range(0,nt):
        lmf = npemf*l; lmg = npemg*l
        jmf = imf+lmf; jmg = img+lmg
        V[jmg] = V[jmf] # Contorno de Dirichlet
        for j in range(1,nxmg-1):
            jmg+=1; jmf+=2
            je=jmf+1; jw=jmf-1;jn=jmf+nxmf
            #print('sul',jmg,jmf,jn,jw,je)
            V[jmg] = ds0*(V[jmf]+V[jn])+ds1*(V[jw]+V[je]+V[jn+1]+V[jn-1])
        # Sudeste
        jmg+=1; jmf+=2
        je=jmf+1; jw=jmf-1;jn=jmf+nxmf
        #print('sudeste',jmg,jmf,jn,jw)
        V[jmg] = ds0*(V[jmf]+V[jn]+V[jw]+V[jn-1])
        # Pontos internos
        for i in range(1,nymg-1):
            jmg+=1  # pois pula o contorno de Dirichlet
            jmf = imf+2*i*nxmf+lmf
            V[jmg] = V[jmf] # Contorno de Dirichlet
            for j in range(1,nxmg-1):
                jmg+=1; jmf+=2
                je=jmf+1; jw=jmf-1;jn=jmf+nxmf;js=jmf-nxmf
                #print('interno',jmg,jmf,jn,jw,je,js)
                V[jmg] = ds0*V[jmf]+ds1*(V[je]+V[jw]+V[jn]+V[js])+ds*(V[jn+1]+V[jn-1]+V[js+1]+V[js-1])
            # Leste
            jmg+=1; jmf+=2
            jw=jmf-1;jn=jmf+nxmf;js=jmf-nxmf
            #print('leste',jmg,jmf,jn,jw,js)
            V[jmg] = ds0*(V[jmf]+V[jw])+ds1*(V[jn]+V[js]+V[jn-1]+V[js-1])
       # Norte
        jmg+=1  # pois pula o contorno de Dirichlet
        jmf = imf+2*(nymg-1)*nxmf+lmf
        V[jmg] = V[jmf] # Contorno de Dirichlet
        for j in range(1,nxmg-1):
            jmg+=1; jmf+=2
            je=jmf+1; jw=jmf-1;js=jmf-nxmf
            #print('norte',jmg,jmf,js,je,jw)
            V[jmg] = ds0*(V[jmf]+V[js])+ds1*(V[je]+V[jw]+V[js+1]+V[js-1])
        # Nordeste
        jmg+=1; jmf+=2
        jw=jmf-1;js=jmf-nxmf
        #print('nordeste',jmg,jmf,js,jw)
        V[jmg] = ds0*(V[jmf]+V[js]+V[jw]+V[js-1])
    return V

def restrige_residuo(V,Q,nm,npxy,npac,npe,nt):
    # Restringe um vetor V da malha nm (FINA) para a malha nm-1 (GROSSA) e aloca no vetor Q
    ds = 0.0625; ds0 = 0.25; ds1=0.125
    nxmg = npxy[nm-1]; nymg = nxmg; nxmf = npxy[nm]
    npemf = npe[nm]; npemg = npe[nm-1]
    img = npac[nm-1]
    # Sul
    for l in range(1,nt):
        lmf = npemf*l; lmg = npemg*l
        jmg = img+lmg; jp = lmf
        for j in range(1,nxmg-1):
            jmg+=1; jp+=2; jn = jp+nxmf
            #print('sul',jmg, jp, jn)
            Q[jmg] = ds0*(V[jp]+V[jn])+ds1*(V[jp-1]+V[jp+1]+V[jn-1]+V[jn+1])
        # Sudeste
        jmg+=1; jp+=2; jn = jp+nxmf
        #print('sudeste',jmg, jp, jn)
        Q[jmg] = ds0*(V[jp]+V[jn]+V[jp-1]+V[jn-1])
        # Pontos internos
        for i in range(1,nymg-1):
            jmg+=1          # pois pula o contorno de Dirichlet
            jp+= nxmf+1     # pula o contorno + uma linha
            for j in range(1,nxmg-1):
                jmg+=1; jp+=2
                jn = jp+nxmf; js = jp-nxmf
                #print('interno',jmg, jp, jn,js)
                Q[jmg] = ds0*V[jp]+ds1*(V[jp+1]+V[jp-1]+V[jn]+V[js])+ds*(V[jn+1]+V[jn-1]+V[js+1]+V[js-1]) 
            # Leste
            jmg+=1; jp+=2
            jn = jp+nxmf; js = jp-nxmf
            #print('leste',jmg, jp, jn,js)
            Q[jmg] = ds0*(V[jp]+V[jp-1])+ds1*(V[jn]+V[js]+V[jn-1]+V[js-1])
            
       # Norte
        jmg+=1  # pois pula o contorno de Dirichlet
        jp+= nxmf+1     # pula o contorno + uma linha
        for j in range(1,nxmg-1):
            jmg+=1; jp+=2; js = jp-nxmf
            #print('norte',jmg, jp,js)
            Q[jmg] = ds0*(V[jp]+V[js])+ds1*(V[jp+1]+V[jp-1]+V[js+1]+V[js-1])

        # Nordeste
        jmg+=1; jp+=2; js=jp-nxmf
        #print('nordeste',jmg, jp, js)
        Q[jmg] = ds0*(V[jp]+V[js]+V[jp-1]+V[js-1])

    return Q

def prolonga(V,Va,nm,npxy,npac,npt,nt):
    # Prolonga um vetor (nele mesmo) da malha nm (GROSSA) para a malha nm+1 (FINA)
    nxmg = npxy[nm]
    nymg = nxmg;       nxmf = npxy[nm+1]
    npemf = nxmf*nxmf; npemg = nxmg*nxmg
    img = npac[nm];     imf = npt[nm]
    for l in range(1,nt):
        lmf = npemf*l; lmg = npemg*l
        jmf = imf+lmf; jmg = img+lmg
        # Sul
        for j in range(1,nxmg):
            jmg+=1; jmf+=2; jw=jmf-1
            V[jmf] = Va[jmf] + V[jmg]
            V[jw]  = Va[jw] + 0.5*(V[jmg-1]+V[jmg])
            #print('sul: nó coincidente',jmf,jmg)
            #print('sul: nó central',jw,jmg-1,jmg)
        # Pontos internos
        for i in range(1,nymg):
            jmg+=1  # pois pula o contorno de Dirichlet
            jmf = imf+2*i*nxmf+lmf
            for j in range(1,nxmg):
                jmg+=1; jmf+=2
                jw=jmf-1;js=jmf-nxmf;jsmg=jmg-nxmg
                V[jmf] = Va[jmf] + V[jmg]
                #print('inerno: nó coincidente',jmf,jmg)
                V[jw]  = Va[jw] + 0.5*(V[jmg]+V[jmg-1])
                #print('inerno: linha norte',jw,jmg,jmg-1)
                V[js]  = Va[js] + 0.5*(V[jmg]+V[jsmg])
                #print('inerno: linha sul',js,jmg,jsmg)
                V[js-1]= Va[js-1] + 0.25*(V[jmg]+V[jmg-1]+V[jsmg]+V[jsmg-1])
                #print('inerno: nó central',js-1,jmg,jmg-1,jsmg,jsmg-1)
                
    return V

def ajusta_fonte(Qf,Tf,apf,aNTf,aeff,awff,aegf,awgf,k1f,T1f,npacf,npef,npxyf,nmf,ntf):
    # Ajusta o termo fonte na malha nmf

    npe = npef[nmf]; i0 = npacf[nmf]+npe; nx = npxyf[nmf]; ny = nx
    for k in range(1,ntf):
        # Sul
        for j in range(1,nx-1):
            i0+=1
            # Qf[i0] = Qf[i0]+apf*TpM+apff*f_p1(TpM,k0f,k1f,T1f)+apgf*g_p1(TpM,k0f,k1f,T1f)-apf*Tf[i0-np]-\
            #     aeff*f_p1(Tf[i0+1],k0f,k1f,T1f)-awff*f_p1(Tf[i0-1],k0f,k1f,T1f)-apgf*g_p1(Tf[i0+nx],k0f,k1f,T1f)
            fon = fonte(Tf[i0-npe],Tf[i0+1],Tf[i0-1],Tf[i0+nx],Tf[i0+nx],0.0,aeff,awff,aegf,awgf,apf,k1f,T1f)
            Qf[i0] = Qf[i0]+apf*Tf[i0]+aNTf*np.exp(k1f*(Tf[i0]-T1f))-fon
            #print('sul','malha= ', nmf,i0,i0-npe,i0+nx)
        # Sudeste
        i0+=1
        # Qf[i0] = Qf[i0]+apf*TpM+apff*f_p1(TpM,k0f,k1f,T1f)+apgf*g_p1(TpM,k0f,k1f,T1f)-apf*Tf[i0-np]-\
        #     -apff*f_p1(Tf[i0-1],k0f,k1f,T1f)-apgf*g_p1(Tf[i0+nx],k0f,k1f,T1f)
        fon = fonte(Tf[i0-npe],Tf[i0-1],Tf[i0-1],Tf[i0+nx],Tf[i0+nx],0.0,aeff,awff,aegf,awgf,apf,k1f,T1f)
        Qf[i0] = Qf[i0]+apf*Tf[i0]+aNTf*np.exp(k1f*(Tf[i0]-T1f))-fon
        #print('sudeste',i0,i0-npe,i0+nx)
        # Interior
        for i in range(1,ny-1):
            i0+=1   # pula contorno de Dirichlet
            for j in range(1,nx-1):
                i0+=1
                # Qf[i0] = Qf[i0]+apf*TpM+apff*f_p1(TpM,k0f,k1f,T1f)+apgf*g_p1(TpM,k0f,k1f,T1f)-apf*Tf[i0-np]-\
                #     aeff*f_p1(Tf[i0+1],k0f,k1f,T1f)-awff*f_p1(Tf[i0-1],k0f,k1f,T1f)-\
                #     aegf*g_p1(Tf[i0+nx],k0f,k1f,T1f)-awgf*g_p1(Tf[i0-nx],k0f,k1f,T1f)
                fon = fonte(Tf[i0-npe],Tf[i0+1],Tf[i0-1],Tf[i0+nx],Tf[i0-nx],0.0,aeff,awff,aegf,awgf,apf,k1f,T1f)
                Qf[i0] = Qf[i0]+apf*Tf[i0]+aNTf*np.exp(k1f*(Tf[i0]-T1f))-fon
                #print('interno',i0,i0-npe,i0-nx,i0+nx)
            # Leste
            i0+=1
            # Qf[i0] = Qf[i0]+apf*TpM+apff*f_p1(TpM,k0f,k1f,T1f)+apgf*g_p1(TpM,k0f,k1f,T1f)-apf*Tf[i0-np]-\
            #     -apff*f_p1(Tf[i0-1],k0f,k1f,T1f)-aegf*g_p1(Tf[i0+nx],k0f,k1f,T1f)-awgf*g_p1(Tf[i0-nx],k0f,k1f,T1f)
            fon = fonte(Tf[i0-npe],Tf[i0-1],Tf[i0-1],Tf[i0+nx],Tf[i0-nx],0.0,aeff,awff,aegf,awgf,apf,k1f,T1f)
            Qf[i0] = Qf[i0]+apf*Tf[i0]+aNTf*np.exp(k1f*(Tf[i0]-T1f))-fon
            #print('leste',i0,i0-npe,i0-nx,i0+nx)
        # Norte
        i0+=1 # Pula o contorno de Dirichlet
        for j in range(1,nx-1):
            i0+=1
            # Qf[i0] = Qf[i0]+apf*TpM+apff*f_p1(TpM,k0f,k1f,T1f)+apgf*g_p1(TpM,k0f,k1f,T1f)-apf*Tf[i0-np]-\
            #     aeff*f_p1(Tf[i0+1],k0f,k1f,T1f)-awff*f_p1(Tf[i0-1],k0f,k1f,T1f)-apgf*g_p1(Tf[i0-nx],k0f,k1f,T1f)
            fon = fonte(Tf[i0-npe],Tf[i0+1],Tf[i0-1],Tf[i0-nx],Tf[i0-nx],0.0,aeff,awff,aegf,awgf,apf,k1f,T1f)
            Qf[i0] = Qf[i0]+apf*Tf[i0]+aNTf*np.exp(k1f*(Tf[i0]-T1f))-fon
            #print('norte',i0,i0-npe,i0-nx)
        # Nordeste
        i0+=1
        # Qf[i0] = Qf[i0]+apf*TpM+apff*f_p1(TpM,k0f,k1f,T1f)+apgf*g_p1(TpM,k0f,k1f,T1f)-apf*Tf[i0-np]-\
        #     -apff*f_p1(Tf[i0-1],k0f,k1f,T1f)-apgf*g_p1(Tf[i0-nx],k0f,k1f,T1f)
        fon = fonte(Tf[i0-npe],Tf[i0-1],Tf[i0-1],Tf[i0-nx],Tf[i0-nx],0.0,aeff,awff,aegf,awgf,apf,k1f,T1f)
        Qf[i0] = Qf[i0]+apf*Tf[i0]+aNTf*np.exp(k1f*(Tf[i0]-T1f))-fon
        #print('nordeste',i0,i0-npe,i0-nx)
        
        i0+=1   # Pula o contorno de Dirichlet do próximo nível de tempo
    return Qf

def solver_mg(solvf,ni1f,grid,Tf,Qf,npxyf,ntf,npef,npacf,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf,tolf,it_mf):
    if (solvf == 0):
        Tf = solver_wr_gs(ni1f,grid,Tf,Qf,npxyf,ntf,npef,npacf,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf,tolf,it_mf)
    else:
        Tf = solver_wr_gsrb(ni1f,grid,Tf,Qf,npxyf,ntf,npef,npacf,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf,tolf,it_mf)
    return Tf

def multigrid(ni1f,ni2f,solvf,Tf,Taf,Qf,nmf,npxyf,npacf,ntf,npef,npetf,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf,dtf,rhof,cpf,k0k1f,lxf,lyf,tolf,it_mf):
    R = residuo(Tf,Qf,nmf-1,npxyf,npacf,ntf,npef,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf)
    res0 = np.max(abs(R))
    it = 0; resk = res0
    while (resk>tolf*res0 and it<it_mf):
        for grid in range(nmf-1,0,-1):
            Tf = solver_mg(solvf,ni1f,grid,Tf,Qf,npxyf,ntf,npef,npacf,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf,tolf,it_mf)
            R = residuo(Tf,Qf,grid,npxyf,npacf,ntf,npef,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf)
            Qf = restrige_residuo(R,Qf,grid,npxyf,npacf,npef,ntf)
            Tf = restrige_solucao(Tf,grid,npxyf,npacf,npetf,ntf)
            Taf[npacf[grid-1]:npacf[grid]] = Tf[npacf[grid-1]:npacf[grid]]
            [apf,awff,aeff,awgf,aegf,aNTf] = atualiza_coef(lxf,lyf,npxyf,grid-1,dtf,rhof,cpf,k0k1f)
            Qf = ajusta_fonte(Qf,Tf,apf,aNTf,aeff,awff,aegf,awgf,k1f,T1f,npacf,npef,npxyf,grid-1,ntf)
        # Malha grossa
        Tf = solver_mg(solvf,10,0,Tf,Qf,npxyf,ntf,npef,npacf,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf,tolf,it_mf)
        Tf[0:npacf[1]] = Tf[0:npacf[1]]-Taf[0:npacf[1]]
        Tf = prolonga(Tf,Tf,0,npxyf,npacf,npetf,ntf)
        for grid in range(1,nmf-1):
            [apf,awff,aeff,awgf,aegf,aNTf] = atualiza_coef(lxf,lyf,npxyf,grid,dtf,rhof,cpf,k0k1f)
            Tf = solver_mg(solvf,ni2f,grid,Tf,Qf,npxyf,ntf,npef,npacf,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf,tolf,it_mf)
            Tf[npacf[grid]:npacf[grid+1]] = Tf[npacf[grid]:npacf[grid+1]]-Taf[npacf[grid]:npacf[grid+1]]
            Tf = prolonga(Tf,Tf,grid,npxyf,npacf,npetf,ntf)
        grid = nmf-1
        [apf,awff,aeff,awgf,aegf,aNTf] = atualiza_coef(lxf,lyf,npxyf,grid,dtf,rhof,cpf,k0k1f)
        Tf = solver_mg(solvf,ni2f,grid,Tf,Qf,npxyf,ntf,npef,npacf,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf,tolf,it_mf)
        R = residuo(Tf,Qf,grid,npxyf,npacf,ntf,npef,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf)

        resk = np.max(abs(R))
        it +=1
        print(it, resk, res0, resk/res0)
    return Tf

def single_grid(Tf,Qf,nxyf,ntf,npef,npacf,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf,it_mf,tolf,met):
    
    Rf = residuo(Tf,Qf,0,nxyf,npacf,ntf,npef,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf)
    res0 = np.max(abs(Rf))
    it = 0; resk = res0
    if met == 0:    # Gauss-Seidel
        while (resk>tolf*res0 and it<it_mf):
            Tf = solver_wr_gs(1,0,Tf,Qf,nxyf,ntf,npef,npacf,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf,tolf,it_mf)
            Rf = residuo(Tf,Qf,0,nxyf,npacf,ntf,npef,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf)
            resk = np.max(abs(Rf))
            print('gs',it, resk)
            it +=1
        return Tf
    elif met == 1:   # Gauss-Seidel red-blacl
        while (resk>tolf*res0 and it<it_mf):
            Tf = solver_wr_gsrb(1,0,Tf,Qf,nxyf,ntf,npef,npacf,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf,tolf,it_mf)
            Rf = residuo(Tf,Qf,0,nxyf,npacf,ntf,npef,apf,aeff,awff,aegf,awgf,k1f,T1f,aNTf)
            resk = np.max(abs(Rf))
            print('gsrb',it, resk)
            it +=1
        return Tf
    else:
        print('ATENÇÃO! Solver inválido: use 0 = GS ou 1 = GS-RB')
        sys.exit()        