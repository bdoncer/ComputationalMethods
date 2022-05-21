def calculate_derivative(x):
    h = 1e-5
    return (function(x + h) - function(x - h)) / (2 * h)

def function(x):
    n = 14
    m = 13
    return x**n + x**m


def newton1(x0):
    epsilon = 0.0001
    ctr = 0
    x1 = x0 - function(x0) / calculate_derivative(x0)
    while ctr < 1000 and abs(x1 - x0) < epsilon:
        x1 = x0 - function(x0) / calculate_derivative(x0)
        x0 = x1
        ctr += 1
    return x1,ctr+1


def newton2(x0):
    epsilon = 0.0001
    ctr = 0
    while ctr < 1000 and abs(function(x0)) > epsilon:
        x1 = x0 - function(x0) / calculate_derivative(x0)
        x0 = x1
        ctr += 1
    return x1,ctr+1


def secant1(x0, x1):
    epsilon = 0.0001
    x2 = x1 - function(x1) * (x1 - x0) / (function(x1) - function(x0))
    ctr = 0
    while ctr < 1000 and abs(x1 - x0) > epsilon:
        x0 = x1
        x1 = x2
        x2 = x1 - function(x1) * (x1 - x0) / (function(x1) - function(x0))
        ctr += 1
    return x1,ctr+1

def secant2(x0, x1):
    epsilon = 0.0001
    x2 = x1 - function(x1) * (x1 - x0) / (function(x1) - function(x0))
    ctr = 0
    while ctr < 1000 and abs(function(x1)) > epsilon:
        x0 = x1
        x1 = x2
        x2 = x1 - function(x1) * (x1 - x0) / (function(x1) - function(x0))
        ctr += 1
    return x1,ctr



