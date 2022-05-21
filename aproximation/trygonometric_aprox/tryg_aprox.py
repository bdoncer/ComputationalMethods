from matplotlib import pyplot as plt
import math
import numpy as np

range_start = -3*np.pi
range_end = 3*np.pi

def function(x):
    return 20+(x*x)/2-20*math.cos(2*x)

#obliczenie n rownoodleglych wezlow
def normal_points(n,points,values):
    a = range_start
    b = range_end
    diff = (b-a)/(n-1)
    for i in range(n):
        points = np.append(points, np.array([a+i*diff]))
        values = np.append(values, np.array([function(a + i * diff)]))
    return points,values

#liczenie wspolczynnikow a
def caluclate_a(a_values,points,values):
    n = len(points)
    for j in range(n):
        res = 0
        for i in range(n-1):
            res += values[i] * math.cos(j*points[i])
        a_values = np.append(a_values,np.array([(2*res)/n]))
    return a_values


#liczenie wspolczynnikow b
def caluclate_b(b_values, points, values):
    n = len(points)
    for j in range(n):
        res = 0
        for i in range(n-1):
            res += values[i] * math.sin(j * points[i])
        b_values = np.append(b_values, np.array([(2 * res) / n]))
    return b_values

def calculate_f(a_values,b_values,m,x):
    x = normalize(x)
    res = 0.5 * a_values[0]
    for j in range(1,m+1):
        res += a_values[j]*math.cos(j*x) + b_values[j]*math.sin(j*x)
    return res


#tworzenie wykresu
def plot(a_values,b_values,points,values,n,m):
    #obliczanie wyniku
    x_plot_aprox = [i for i in np.arange(range_start, range_end, (range_end- range_start)/998.5)]
    x_plot_aprox.append(range_end)
    y_plot_aprox = [calculate_f(a_values,b_values,m,i) for i in np.arange(range_start, range_end, (range_end - range_start)/998.5)]
    y_plot_aprox.append(calculate_f(a_values,b_values,m,range_end))
    plt.plot(x_plot_aprox, y_plot_aprox, color="green", label = "aproximation")


    #rysowanie interpolowanej funkcji
    x_plot = [i for i in np.arange(range_start, range_end, (range_end- range_start)/998.5)]
    x_plot.append(range_end)
    y_plot = [function(i) for i in np.arange(range_start, range_end, (range_end- range_start)/998.5)]
    y_plot.append(function(range_end))
    plt.plot(x_plot, y_plot, color="blue",label = "function")
    plt.legend(loc="upper left")

    #dodawanie wezlow
    plt.scatter(points, values, marker='o', s=20, color="black")

    #liczenie bledu kwadratowego
    count_accuracy(x_plot, y_plot, y_plot_aprox)

    #liczenie maksymalnej roznicy
    count_accuracy2(x_plot, y_plot,y_plot_aprox)

    #pokazywanie wykresu
    plt.title(f"n = {n}, m = {m}")
    plt.show()

#liczenie bledu kwadratowego
def count_accuracy(x,y,y_aprox):
    print("Błąd kwadratowy")
    res_aprox = 0
    for i in range(len(x)):
        res_aprox += (y_aprox[i] - y[i])*(y_aprox[i] - y[i])
    print(res_aprox)


#liczenie maksymalnej roznicy
def count_accuracy2(x,y,y_aprox):
    print("Maksymalna różnica")
    res_aprox = 0
    for i in range(len(x)):
        res_aprox = max(abs(y_aprox[i] - y[i]),res_aprox)
    print(res_aprox)

def normalize(x):
    return x / (range_end - range_start) * 2 * math.pi - math.pi - (range_start / (range_end - range_start) * 2 * math.pi)

def denormalize(x):
    return ((x - (-math.pi - (range_start / (range_end - range_start) * 2 * math.pi))) / (2 * math.pi)) * (range_end - range_start)

#m - stopien wielomianu
#n - ilosc wezlow
def main(m,n):

    points = np.array([])
    values = np.array([])
    a_values = np.array([])
    b_values = np.array([])

    points, values = normal_points(n, points, values)

    for i in range(len(points)):
        points[i] = normalize(points[i])

    a_values = caluclate_a(a_values,points,values)
    b_values = caluclate_b(b_values, points, values)

    for i in range(len(points)):
        points[i] = denormalize(points[i])

    plot(a_values, b_values, points, values, n,m)



main(20,100)
