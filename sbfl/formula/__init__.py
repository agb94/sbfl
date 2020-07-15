import numpy as np

def ochiai(e_p, n_p, e_f, n_f):
    return e_f/np.sqrt((e_f+n_f)*(e_f+e_p))

def op2(e_p, n_p, e_f, n_f):
    return e_f-e_p/(e_p+n_p+1)

def dstar(e_p, n_p, e_f, n_f, star=2):
    return np.power(e_f,star)/(e_p+n_f)

def tarantula(e_p, n_p, e_f, n_f):
    return (e_f/(e_f+n_f))/((e_f/(e_f+n_f))+(e_p/(e_p+n_p)))

def jaccard(e_p, n_p, e_f, n_f):
    return e_f/(e_f+n_f+e_p)

def gp13(e_p, n_p, e_f, n_f):
    return e_f*(1 + 1/(2*e_p+e_f))
