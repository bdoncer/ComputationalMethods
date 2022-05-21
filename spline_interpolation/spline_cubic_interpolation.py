import numpy as np
import matplotlib.pyplot as plt
import math

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

def calculate_h(i,points):
    return points[i+1] - points[i]

def calculate_delta(i,values,points):
    return (values[i+1] - values[i])/calculate_h(i,points)

def make_main_matrix(matrix,n,points):
    matrix = matrix.astype(np.float64)
    for i in range(1,n):
        j = i - 1
        matrix[i][j] = calculate_h(i-1,points)
        matrix[i][j + 1] = 2 * (calculate_h(i-1,points) + calculate_h(i,points))
        matrix[i][j + 2] = calculate_h(i,points)

    return matrix

def make_y_matrix(y,n,values,points):
    y = np.append(y,np.array([0]))
    for i in range(n-1):
        y = np.append(y,np.array([calculate_delta(i+1,values,points)-calculate_delta(i,values,points)]))
    y = np.append(y, np.array([0]))
    return y


#obliczanie wartosci splajnu w punkcie x
def calculate_s(x,res,points,values):
    ind = 0
    while ind <= len(points) and (points[ind] > x or x > points[ind + 1]):
        ind += 1
    return (res[ind]*(points[ind+1]-x)**3)/calculate_h(ind,points)+ \
           (res[ind+1]*(x-points[ind])**3)/calculate_h(ind,points)+\
           (values[ind+1]/calculate_h(ind,points)-res[ind+1]*calculate_h(ind,points))*(x-points[ind])+\
           (values[ind]/calculate_h(ind,points)-res[ind]*calculate_h(ind,points))*(points[ind+1]-x)

#rysowanie funkcji
def plot_polynomial(n,res1,res2,points,values):
    # rysowanie splajna z pierwszym warunkiem brzegowym
    x_plot_spline1 = [i for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi / 998.5)]
    x_plot_spline1.append(3 * np.pi)
    y_plot_spline1 = [calculate_s(i,res1,points,values) for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi / 998.5)]
    y_plot_spline1.append(calculate_s(3 * np.pi,res1,points,values))
    plt.title("Cubic spline")
    plt.plot(x_plot_spline1, y_plot_spline1, color="green",label="spline - warunek 1")

    # rysowanie splajna z drugim warunkiem brzegowym
    x_plot_spline2 = [i for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi / 998.5)]
    x_plot_spline2.append(3 * np.pi)
    y_plot_spline2 = [calculate_s(i, res2, points, values) for i in np.arange(-3 * np.pi, 3 * np.pi, 6 * np.pi / 998.5)]
    y_plot_spline2.append(calculate_s(3 * np.pi, res2, points, values))
    plt.plot(x_plot_spline2, y_plot_spline2, color="yellow", label="spline - warunek 2")

    # dodawanie wezlow
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
def count_accuracy1(x,y,y_cubic):
    res_cubic = 0
    for i in range(len(x)):
        res_cubic += (y_cubic[i] - y[i])*(y_cubic[i] - y[i])
    return res_cubic

#liczenie maksymalnej roznicy
def count_accuracy2(x,y,y_cubic):
    res_cubic = 0
    for i in range(len(x)):
        res_cubic = max(abs(y_cubic[i] - y[i]),res_cubic)
    return res_cubic

#obliczenia potrzebne do 2 warunku brzegowego
def calculate_delta_derivative(ind,points,values):
    tmp = np.array([[0,0,0],[0,0,0],[0,0,0]])
    for i in range(3):
        tmp[i][0] = values[ind + i]
    for i in range(3):
        for j in range(i):
            tmp[i][j+1] = (tmp[i][j]-tmp[i - 1][j]) / (points[i]-points[i-j-1])
    return tmp[2][2]

def main(n):
    points = np.array([])
    values = np.array([])
    # obliczanie punktow i wartosci dla n punktow
    points,values = normal_points(n,points,values)
    n -= 1

    # tworzenie 1 macierzy
    matrix1 = np.array([[0 for i in range(n+1)] for i in range(n+1)])
    matrix2 = np.array([[0 for i in range(n + 1)] for i in range(n + 1)])
    matrix1 = make_main_matrix(matrix1, n,points)
    matrix2 = make_main_matrix(matrix1, n, points)

    # tworzenie 3 macierzy
    y1 = np.array([])
    y2 = np.array([])
    y1 = make_y_matrix(y1, n,values,points)
    y2 = make_y_matrix(y2, n, values, points)

    #drugie pochodne na krancach = 0
    matrix1[0][0] = 6
    matrix1[n][n] = 6
    #cztery punkty
    y2[0] = calculate_h(0,points)**2 * calculate_delta_derivative(0,points,values)
    y2[len(y2)-1] = (-1) * calculate_h(len(y2)-2,points) ** 2 * calculate_delta_derivative(n - 4,points,values)
    matrix2[0][0] = -calculate_h(0,points)
    matrix2[0][1] = calculate_h(0,points)
    matrix2[n][n-1] = calculate_h(n-1, points)
    matrix2[n][n] = -calculate_h(n-1, points)

    #obliczanie macierzy wynikowej
    res1 = np.linalg.solve(matrix1, y1)
    res2 = np.linalg.solve(matrix2, y2)

    #rysowanie wykresu
    plot_polynomial(n, res1,res2, points, values)





