def gauss_elimination(matrixB,matrixL) :
    n = len(matrixL)
    matrix = [[0 for i in range(n+1)] for i in range(n)]
    res = [0 for i in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = matrixB[i][j]
        matrix[i][n] = matrixL[i]

    for i in range(n-1):
        for j in range(i+1,n):
            c= matrix[j][i]/matrix[i][i]
            for k in range(n+1):
                matrix[j][k] -= c*matrix[i][k]


    res[n-1] = matrix[n-1][n]/matrix[n-1][n-1]
    for i in range(n-2,-1,-1):
        sum=0
        for j in range(i+1,n):
            sum+=matrix[i][j]*res[j]
        res[i]=(matrix[i][n]-sum)/matrix[i][i]
    return res


