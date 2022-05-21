from matplotlib import pyplot as plt
import math
import numpy as np


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

def count_matrix(m,points,values,matrix):
    results = [0 for i in range(2*m+1)]
    n = len(points)
    for i in range(2*m+1):
        for j in range(n):
            results[i] += points[j]**i

    for i in range(m+1):
        for j in range(m+1):
            matrix[i][j] = results[i+j]

    return matrix

def count_y(y,points,values,m):
    for i in range(m+1):
        for j in range(len(points)):
            y[i] += values[j]*(points[j])**i
    return y

def calculate_f(x,a):
    res = 0
    for i in range(len(a)):
        res += a[i] * x**i
    return res

#tworzenie wykresu
def plot(points,values,a,m,n):
    #obliczanie wyniku metoda Newtona
    x_plot_aprox = [i for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi/998.5)]
    x_plot_aprox.append(3 * np.pi)
    y_plot_aprox = [calculate_f(i,a) for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi/998.5)]
    y_plot_aprox.append(calculate_f(3 * np.pi,a))
    plt.plot(x_plot_aprox, y_plot_aprox, color="green", label = "aproximation")


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
    count_accuracy(x_plot, y_plot, y_plot_aprox,m,n)

    #liczenie maksymalnej roznicy
    count_accuracy2(x_plot, y_plot, y_plot_aprox,m,n)

    #pokazywanie wykresu
    plt.title(f"n = {n}, m = {m}")
    plt.show()

#liczenie bledu kwadratowego
def count_accuracy(x,y,y_aprox,m,n):
    print("błąd kwadratowy")
    res_aprox = 0
    for i in range(len(x)):
        res_aprox += (y_aprox[i] - y[i])*(y_aprox[i] - y[i])
    print(res_aprox)


#liczenie maksymalnej roznicy
def count_accuracy2(x,y,y_aprox,m,n):
    print("maksymalna różnica")
    res_aprox = 0
    for i in range(len(x)):
        res_aprox = max(abs(y_aprox[i] - y[i]),res_aprox)
    print(res_aprox)


#m - stopien wielomianu
#n - ilosc wezlow
def main(m,n):
    points = np.array([])
    values = np.array([])

    points, values = normal_points(n, points, values)

    matrix = np.zeros((m+1, m+1))
    matrix = count_matrix(m,points,values,matrix)

    y = np.zeros((m+1))
    y = count_y(y,points,values,m)

    a = np.linalg.solve(matrix, y)
    plot(points,values,a,m,n)



main(3,10)