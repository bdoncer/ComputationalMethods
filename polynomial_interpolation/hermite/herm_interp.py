import math
import numpy as np
from matplotlib import pyplot as plt


def function(x):
    return 20 + (x * x) / 2 - 20 * math.cos(2 * x)

def calculate_derivative(x):
    h = 1e-10
    return (function(x + h) - function(x - h)) / (2 * h)

#obliczenie n rownoodleglych wezlow
def normal_points(n,points,values):
    a = -3 * np.pi
    b = 3 * np.pi
    diff = (b-a)/(n-1)
    for i in range(n):
        points.append(a+i*diff)
        values.append([function(a + i * diff),calculate_derivative(a + i * diff)])
    return points,values

#obliczenie n wezlow Czebyszewa
def czebyszew_points(n,points,values):
    a = -3 * np.pi
    b = 3 * np.pi
    points.append(a)
    values.append([function(a),calculate_derivative(a)])
    for i in range(1,n-1):
        x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * (i + 1) - 1) / (2 * n) * np.pi)
        points.append(x)
        values.append([function(x),calculate_derivative(x)])
    points.append(b)
    values.append([function(b), calculate_derivative(b)])
    return points,values

def count_matrix(matrix,points,values):
    n = len(points)
    for i in range(n):
        for j in range(i+1):
            if matrix[i][j] != None:
                continue
            if j == 0:
                matrix[i][j] = values[i]
            else:
                matrix[i][j] = (matrix[i][j-1] - matrix[i-1][j-1]) / (points[i] - points[i-j])


def count_polynomial(matrix, x, points):
    n = len(points)
    res = 0
    for i in range(n):
        product = 1
        for j in range(i):
            product *= (x - points[j])
        res += product * matrix[i][i]
    return res


def plot(matrix,points,values):
    x_plot_hermite = [i for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi/999)]
    x_plot_hermite.append(3 * np.pi)
    y_plot_hermite = [count_polynomial(matrix, i, points) for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi / 999)]
    y_plot_hermite.append(count_polynomial(matrix, 3 * np.pi, points))
    plt.plot(x_plot_hermite, y_plot_hermite, color="green", label = "hermite")
    plt.title(len(points)/2)


    x_plot = [i for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi/999)]
    x_plot.append(3 * np.pi)
    y_plot = [function(i) for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi/999)]
    y_plot.append(function(3 * np.pi))
    plt.plot(x_plot, y_plot, color="blue",label = "function")
    new_values = []
    new_points = []
    for i in range(len(values)):
        if i%2 == 0:
            new_values.append(values[i])
            new_points.append(points[i])
    plt.scatter(new_points, new_values, marker='o', s=20, color="black")
    plt.legend(loc="upper left")

    plt.show()

    return count_accuracy1(x_plot, y_plot, y_plot_hermite)
    # return count_accuracy2(x_plot, y_plot, y_plot_hermite)


def how_many(new_points,i):
    ctr = 0
    j = i
    while (new_points[j] == new_points[i]):
        ctr += 1
        j -= 1
    return ctr

#blad kwadratowy
def count_accuracy1(x,y,y_hermite):
    res_hermite = 0
    for i in range(len(x)):
        res_hermite += (y_hermite[i] - y[i]) * (y_hermite[i] - y[i])
    return res_hermite

#maksymalna roznica
def count_accuracy2(x,y,y_hermite):
    res_hermite = 0
    for i in range(len(x)):
        res_hermite = max(abs(y_hermite[i] - y[i]), res_hermite)
    return res_hermite




def main(n,type):
    points = []
    values = []
    if type == 0:
        points,values = normal_points(n,points,values)
    else:
        points, values = czebyszew_points(n, points, values)
    n = len(points)
    ctr = 0
    for i in range(n):
        for j in range(len(values[i])):
            ctr += 1

    new_points = [0]*ctr
    i = 0
    j = 0

    while (j < ctr):
        x = len(values[i])
        while (x > 0):
            new_points[j] = points[i]
            j += 1
            x -= 1
        i += 1

    i = 0
    j = 0
    new_values = []
    for row in values:
        for elem in row:
            new_values.append(elem)




    matrix = [[None for i in range(ctr)] for j in range(ctr)]

    row = 0
    for i in range(len(points)):
        point_values = values[i]
        for j in range(len(point_values)):
            for k in range(j+1):
                matrix[row][k] = point_values[k] / math.factorial(k)
            row += 1

    count_matrix(matrix,new_points,new_values)

    return plot(matrix,new_points,new_values)


#main(ilosc wezlow, typ (0 - rownoodlegle, 1 - czebyszewa)


