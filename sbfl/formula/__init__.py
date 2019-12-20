def ochiai(ep, np, ef, nf):
    from numpy import sqrt
    return ef/sqrt((ef+nf)*(ef+ep))

def op2(ep, np, ef, nf):
    return ef-ep/(ep+np+1)

def dstar(ep, np, ef, nf, star=2):
    from numpy import power
    return power(ef,star)/(ep+nf)

def tarantula(ep, np, ef, nf):
    return (ef/(ef+nf))/((ef/(ef+nf))+(ep/(ep+np)))
