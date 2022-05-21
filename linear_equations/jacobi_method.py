import numpy as np


def fx(x,y,z):
    return (12+y-2*z)/5

def fy(x,y,z):
    return (-25+3*x+2*z)/8

def fz(x,y,z):
    return (6-x-y)/4


def jacobi_method():
    max_iter = 100
    guess = [0,0,0]
    tmp = [0,0,0]
    for i in range(max_iter):
        tmp[0] = fx(guess[0],guess[1],guess[2])
        tmp[1] = fy(guess[0], guess[1], guess[2])
        tmp[2] = fz(guess[0], guess[1], guess[2])
        guess = tmp.copy()
        print(guess)
    return guess


print(jacobi_method())





