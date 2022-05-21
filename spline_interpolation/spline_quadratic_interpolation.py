import math
from sympy import *
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

#type = 0 - warunek brzegowy a1 = 0
#type = 1 - warunek brzegowy s1(x0)' = 0
def make_main_matrix(matrix,n,points,type):
    i = 1
    j = 0
    x = 1
    matrix = matrix.astype(np.float64)
    matrix[0][0] = points[0] * points[0]
    matrix[0][1] = points[0]
    matrix[0][2] = 1
    while i < 2*n-1:
        matrix[i][j] = points[x] * points[x]
        matrix[i][j+1] = points[x]
        matrix[i][j+2] = 1
        matrix[i+1][j+3] = points[x] * points[x]
        matrix[i+1][j + 4] = points[x]
        matrix[i+1][j + 5] = 1
        i += 2
        j += 3
        x += 1
    matrix[i][j] = points[n] * points[n]
    matrix[i][j+1] = points[n]
    matrix[i][j+2] = 1
    i += 1
    j = 0
    x = 1
    while i < 3*n-1:
        matrix[i][j] = 2 * points[x]
        matrix[i][j + 1] = 1
        matrix[i][j + 3] = -2 * points[x]
        matrix[i][j + 4] = -1
        j += 3
        i += 1
        x += 1
    if (type == 0):
        matrix[3*n-1][0] = 1
    if (type == 1):
        matrix[3 * n - 1][0] = 2 * points[0]
        matrix[3 * n - 1][0] = 1
    return matrix

def calculate_derivative(x):
    #jesli zalozylibysmy, ze nie znamy pochodnych na krancach, to mozemy przyblizyc pochodna
    #ze wzoru (f[i+1]-f[i])/hi
    h = 1e-10
    return (function(x + h) - function(x - h)) / (2 * h)

#wypelnianie 3 macierzy (z wynikami funkcji dla danych x)
def make_y_matrix(y,n,values,points,type):
    i = 1
    y = np.append(y,np.array([values[0]]))
    while i < n:
        y = np.append(y,np.array([values[i]]))
        y = np.append(y,np.array([values[i]]))
        i += 1
    y = np.append(y,np.array([values[i]]))
    i = 2 * i
    while i < 3 * n - 1:
        y = np.append(y,np.array([0]))
        i += 1
    if (type == 0):
        y = np.append(y, np.array([0]))
    if (type == 1):
        y = np.append(y, np.array([calculate_derivative(points[0])]))
    return y

    
#zwracanie wartosci splajnu
def return_value(x,res,points):
    for i in range(1,len(points)):
        if x <= points[i]:
            return x*x*res[3*i-3] + x*res[3*i-2] + res[3*i-1]

#rysowanie funkcji
def plot_polynomial(n,res1,res2,points,values):
    #rysowanie splajna z pierwszym warunkiem brzegowym
    x_plot_spline1 = [i for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi / 998.5)]
    x_plot_spline1.append(3 * np.pi)
    y_plot_spline1 = [return_value(i,res1,points) for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi / 998.5)]
    y_plot_spline1.append(return_value(3 * np.pi,res1,points))
    plt.title("Quadratic spline")
    plt.plot(x_plot_spline1, y_plot_spline1, color="green",label="spline - warunek 1")

    # rysowanie splajna z drugim warunkiem brzegowym
    x_plot_spline2 = [i for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi / 998.5)]
    x_plot_spline2.append(3 * np.pi)
    y_plot_spline2 = [return_value(i, res2, points) for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi / 998.5)]
    y_plot_spline2.append(return_value(3 * np.pi, res2, points))
    plt.title("Quadratic spline")
    plt.plot(x_plot_spline2, y_plot_spline2, color="yellow", label="spline - warunek 2")

    #dodawanie wezlow
    plt.scatter(points, values, marker='o', s=20, color="black")

    # rysowanie interpolowanej funkcji
    x_plot = [i for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi / 998.5)]
    x_plot.append(3 * np.pi)
    y_plot = [function(i) for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi / 998.5)]
    y_plot.append(function(3 * np.pi))
    plt.plot(x_plot, y_plot, color="blue", label="function")
    plt.legend(loc="upper left")

    plt.show()


#liczenie bledu kwadratowego
def count_accuracy1(x, y, y_quadratic):
    res_quadratic = 0
    for i in range(len(x)):
        res_quadratic += (y_quadratic[i] - y[i]) * (y_quadratic[i] - y[i])
    return res_quadratic

#liczenie maksymalnej roznicy
def count_accuracy2(x, y, y_quadratic):
    res_quadratic = 0
    for i in range(len(x)):
        res_quadratic = max(abs(y_quadratic[i] - y[i]), res_quadratic)
    return res_quadratic

def main(n):

    points = np.array([])
    values = np.array([])
    #obliczanie punktow i wartosci dla n punktow
    points,values = normal_points(n,points,values)
    n -= 1

    #tworzenie 1 macierzy (ze liczbami przy a,b,c)
    matrix1 = np.array([[0 for i in range(3 * n)] for i in range(3 * n)])
    matrix2 = np.array([[0 for i in range(3 * n)] for i in range(3 * n)])
    matrix1 = make_main_matrix(matrix1,n,points,0)
    matrix2 = make_main_matrix(matrix2, n, points, 0)

    #tworzenie 3 macierzy (z wynikami funkcji dla danych x)
    y1 = np.array([])
    y2 = np.array([])
    y1 = make_y_matrix(y1,n,values,points,0)
    y2 = make_y_matrix(y2, n, values,points,1)

    #obliczanie 2 macierzy (ze wspolczynnikami a,b,c)
    res1 = np.linalg.solve(matrix1, y1)
    res2 = np.linalg.solve(matrix2, y2)

    #rysowanie wykresu
    plot_polynomial(n,res1,res2,points,values)





