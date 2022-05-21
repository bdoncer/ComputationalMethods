import math
import numpy as np
from matplotlib import pyplot as plt


def function(x):
    return 20+(x*x)/2-20*math.cos(2*x)

#obliczenie n rownoodleglych wezlow
def normal_points(n,points,values):
    a = -3 * np.pi
    b = 3 * np.pi
    diff = (b-a)/(n-1)
    for i in range(n):
        points = np.append(points,np.array([a+i*diff]))
        values = np.append(values, np.array([function(a + i * diff)]))
    return points,values

#obliczenie n wezlow Czebyszewa
def czebyszew_points(n,points,values):
    a = -3 * np.pi
    b = 3 * np.pi
    points = np.append(points, np.array([a]))
    values = np.append(values, np.array([function(a)]))
    for i in range(1,n-1):
        x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * (i + 1) - 1) / (2 * n) * np.pi)
        points = np.append(points,np.array([x]))
        values = np.append(values, np.array([function(x)]))
    points = np.append(points, np.array([b]))
    values = np.append(values, np.array([function(b)]))
    return points,values

#obliczenie macierzy - metoda Newtona
def count_matrix(matrix,points,values):
    n = len(points)
    for i in range(n):
        for j in range(i+1):
            if j == 0:
                matrix[i][j] = values[i]
            else:
                matrix[i][j] = (matrix[i][j-1] - matrix[i-1][j-1]) / (points[i] - points[i-j])

#obliczanie wartosci interpolowanego wielomianu dla konkretnego x - metoda Newtona
def count_polynomial_newton(matrix,x,points):
    n = len(points)
    res = 0
    for i in range(n):
        product = 1
        for j in range(i):
            product *= (x - points[j])
        res += product * matrix[i][i]
    return res

#obliczanie czesci wyniku - metoda Lagrangea
def count_l(k,x,points):
    numerator = 1
    denominator = 1
    n = len(points)
    for i in range(n):
        if i != k:
            numerator *= (x - points[i])
            denominator *= (points[k] - points[i])
    return numerator/denominator

#obliczanie wartosci interpolowanego wielomianu dla konkretnego x - metoda Lagrangea
def count_polynomial_lagrange(x,points,values):
    n = len(values)
    res = 0
    for i in range(n):
        res += values[i] * count_l(i,x,points)
    return res

#tworzenie wykresu
def plot(matrix,points,values):
    #obliczanie wyniku metoda Newtona
    x_plot_newton = [i for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi/998.5)]
    x_plot_newton.append(3 * np.pi)
    y_plot_newton = [count_polynomial_newton(matrix, i,points) for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi/998.5)]
    y_plot_newton.append(count_polynomial_newton(matrix,3 * np.pi,points))
    plt.plot(x_plot_newton, y_plot_newton, color="green", label = "newton")

    #obliczanie wyniku metoda Lagrangea
    x_plot_lagrange = [i for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi/998.5)]
    x_plot_lagrange.append(3 * np.pi)
    y_plot_lagrange = [count_polynomial_lagrange(i,points,values) for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi/998.5)]
    y_plot_lagrange.append(count_polynomial_lagrange(3 * np.pi,points,values))
    plt.plot(x_plot_lagrange, y_plot_lagrange,  color="yellow", label = "lagrange")

    #rysowanie interpolowanej funkcji
    x_plot = [i for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi/998.5)]
    x_plot.append(3 * np.pi)
    y_plot = [function(i) for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi/998.5)]
    y_plot.append(function(3 * np.pi))
    plt.plot(x_plot, y_plot, color="blue",label = "function")
    plt.legend(loc="upper left")

    #dodawanie wezlow
    plt.scatter(points, values, marker='o', s=20, color="black")

    #liczenie bledu kwadratowego
    count_accuracy(x_plot, y_plot, y_plot_lagrange, x_plot_newton)

    #liczenie maksymalnej roznicy
    count_accuracy2(x_plot, y_plot, y_plot_lagrange,x_plot_newton)

    #pokazywanie wykresu
    plt.show()


#liczenie bledu kwadratowego
def count_accuracy(x,y,y_lagrange,y_newton):
    print("blad kwadratowy")
    res_lagrange = 0
    res_newton = 0
    for i in range(len(x)):
        res_lagrange += (y_lagrange[i] - y[i])*(y_lagrange[i] - y[i])
        res_newton += (y_newton[i] - y[i]) * (y_newton[i] - y[i])
    print("lagrange")
    print(res_lagrange)
    print("newton")
    print(res_newton)

#liczenie maksymalnej roznicy
def count_accuracy2(x,y,y_lagrange,y_newton):
    print("maksymalna roznica")
    res_lagrange = 0
    res_newton = 0
    for i in range(len(x)):
        res_lagrange = max(abs(y_lagrange[i] - y[i]),res_lagrange)
        res_newton = max(abs(y_newton[i] - y[i]),res_newton)
    print("lagrange")
    print(res_lagrange)
    print("newton")
    print(res_newton)


#wywolanie programu: n - liczba wezlow, type - rodzaj wezlow (0 = rownoodlegle, 1 = Czebyszewa)
def main(n,type):
    points = np.array([])
    values = np.array([])

    if (type == 0):
        points,values = normal_points(n,points,values)
    else:
        points, values = czebyszew_points(n, points, values)

    matrix = [[0 for i in range(n)] for j in range(n)]
    count_matrix(matrix,points,values)

    plot(matrix,points,values)


