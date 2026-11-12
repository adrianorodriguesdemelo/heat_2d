import numpy as np

# Fonte de calor
def calcula_Q(Qf,temp,xf,yf,nxf,nyf):
    for j in range(nxf):
        for i in range(nyf):
            Qf[i,j] = 5*xf[j]+yf[i]+10*xf[j]*yf[i]
    Qf = 3.0*temp*np.exp(-temp)*(1.0+Qf)
    return Qf

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

# Solver do sistema algébrico
def solver(Tnf,Ff,T1f,nxf,nyf,a0f,a1f,a2f,a3f,k1f,qsf,qnf,qlf,dym1f,dxm1f,tolf,it_mf):
    erro_g = 1; it_g = 0; Tf = Tnf.copy()
    while (erro_g>tolf and it_g<it_mf):
        # Sul
        for j in range(1,nxf-1):
            fonte = Ff[0,j]+a2f*(np.exp(k1f*(Tnf[0,j+1]-T1f))+np.exp(k1f*(Tnf[0,j-1]-T1f)))+2.0*a3f*np.exp(k1f*(Tnf[1,j]-T1f))-2.0*qsf*dym1f
            Tnf[0,j] = resolve_newton(Tnf[0,j],T1f,fonte,k1f,a0f,a1f,tolf,it_mf)
        # Sudeste
        j = nxf-1
        fonte = Ff[0,j]+2.0*a2f*np.exp(k1f*(Tnf[0,j-1]-T1f))+2.0*a3f*np.exp(k1f*(Tnf[1,j]-T1f))-2.0*(qsf*dym1f+qlf*dxm1f)
        Tnf[0,j] = resolve_newton(Tnf[0,j],T1f,fonte,k1f,a0f,a1f,tolf,it_mf)
        # Pontos Internos
        for i in range(1,nyf-1):
            for j in range(1,nxf-1):
                fonte = Ff[i,j]+a2f*(np.exp(k1f*(Tnf[i,j+1]-T1f))+np.exp(k1f*(Tnf[i,j-1]-T1f)))+a3f*(np.exp(k1f*(Tnf[i+1,j]-T1f))+np.exp(k1f*(Tnf[i-1,j]-T1f)))
                Tnf[i,j] = resolve_newton(Tnf[i,j],T1f,fonte,k1f,a0f,a1f,tolf,it_mf)
            # Leste
            j = nxf-1
            fonte = Ff[i,j]+2.0*a2f*np.exp(k1f*(Tnf[i,j-1]-T1f))+a3f*(np.exp(k1f*(Tnf[i+1,j]-T1f))+np.exp(k1f*(Tnf[i-1,j]-T1f)))-2.0*qlf*dxm1f
            Tnf[i,j] = resolve_newton(Tnf[i,j],T1f,fonte,k1f,a0f,a1f,tolf,it_mf)           

        # Norte
        i = nyf-1
        for j in range(1,nxf-1):
            fonte = Ff[i,j]+a2f*(np.exp(k1f*(Tnf[i,j+1]-T1f))+np.exp(k1f*(Tnf[i,j-1]-T1f)))+2.0*a3f*np.exp(k1f*(Tnf[i-1,j]-T1f))-2.0*qnf*dym1f
            Tnf[i,j] = resolve_newton(Tnf[i,j],T1f,fonte,k1f,a0f,a1f,tolf,it_mf)     
        # Nordeste
        j = nxf-1
        fonte = Ff[i,j]+2.0*a2f*np.exp(k1f*(Tnf[i,j-1]-T1f))+2.0*a3f*np.exp(k1f*(Tnf[i-1,j]-T1f))-2.0*(qlf*dxm1f+qnf*dym1f)
        Tnf[i,j] = resolve_newton(Tnf[i,j],T1f,fonte,k1f,a0f,a1f,tolf,it_mf)

        it_g +=1
        erro_g = np.max(abs(Tf-Tnf))/np.max(abs(Tf))
        Tf = Tnf.copy()
    return Tnf