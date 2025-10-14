import math
import pathlib

# exercitiul 1

def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    A = []
    B = []

    with open(path, 'r') as f:
        for line in f:
            sistem = line.split()

            B.append(float(sistem[-1]))

            coef_x = 0.0
            coef_y = 0.0
            coef_z = 0.0

            semn_poz = 1
            for token in sistem:
                if token == '+':
                    semn_poz = 1
                elif token == '-':
                    semn_poz = -1
                elif token == '=':
                    break
                elif 'x' in token:
                    num = token.replace('x', '')
                    if num == '' or num == '+':
                        coef_x = 1.0 * semn_poz
                    elif num == '-':
                        coef_x = -1.0 * semn_poz
                    else:
                        coef_x = float(num) * semn_poz
                elif 'y' in token:
                    num = token.replace('y', '')
                    if num == '' or num == '+':
                        coef_y = 1.0 * semn_poz
                    elif num == '-':
                        coef_y = -1.0 * semn_poz
                    else:
                        coef_y = float(num) * semn_poz
                elif 'z' in token:
                    num = token.replace('z', '')
                    if num == '' or num == '+':
                        coef_z = 1.0 * semn_poz
                    elif num == '-':
                        coef_z = -1.0 * semn_poz
                    else:
                        coef_z = float(num) * semn_poz

            A.append([coef_x, coef_y, coef_z])

    return A, B


A, B = load_system(pathlib.Path("system.txt"))
print(f"Matricea A = {A}")
print(f"Vectorul B = {B}")


# exercitiul 2

def determinant(matrix: list[list[float]]) -> float:
    det = 0
    a11 = matrix[0][0]
    a12 = matrix[0][1]
    a13 = matrix[0][2]
    a21 = matrix[1][0]
    a22 = matrix[1][1]
    a23 = matrix[1][2]
    a31 = matrix[2][0]
    a32 = matrix[2][1]
    a33 = matrix[2][2]
    det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)

    return det

def trace(matrix: list[list[float]]) -> float:
    suma = 0
    for i in range(len(matrix)):
        suma += matrix[i][i]
    return suma

def norm(vector: list[float]) -> float:
    suma = 0
    for element in vector:
        suma += element ** 2
    radical = math.sqrt(suma)
    return radical


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    randuri = len(matrix)
    coloane = len(matrix[0])

    result = []
    for j in range(coloane):
        rand = []
        for i in range(randuri):
            rand.append(matrix[i][j])
        result.append(rand)

    return result


def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    result = []
    for i in range(len(matrix)):
        suma = 0
        for j in range(len(vector)):
            suma += matrix[i][j] * vector[j]
        result.append(suma)

    return result
#
# A = [[1, 5, 6], [6, 3, 6], [7, 10, 9]]
# B = [5, 6, 7]

# print(f"Matricea A:")
# for row in A:
#     print(row)
# print(f"\nVectorul B: {B}\n")

print(f"determinant(A) = {determinant(A)}")
print(f"trace(A) = {trace(A)}")
print(f"norm(B) = {norm(B)}")
print(f"transpose(A) = {transpose(A)}")
print(f"multiply(A, B) = {multiply(A, B)}")



# exercitiul 3


def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det_A = determinant(matrix)
    result = []

    for col in range(3):
        matrix_aux = []
        for i in range(3):
            row = []
            for j in range(3):
                row.append(matrix[i][j])
            matrix_aux.append(row)

        for i in range(3):
            matrix_aux[i][col] = vector[i]

        det_temp = determinant(matrix_aux)

        result.append(det_temp / det_A)

    return result

print(f"solve_cramer(A, B) = {solve_cramer(A, B)}")


# exercitiul 4

def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    rezultat = []
    for idx_rand in range(len(matrix)):
        if idx_rand == i:
            continue
        rand = []
        for idx_col in range(len(matrix[0])):
            if idx_col == j:
                continue
            rand.append(matrix[idx_rand][idx_col])
        rezultat.append(rand)
    return rezultat


def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    rezultat = []
    for i in range(len(matrix)):
        rand = []
        for j in range(len(matrix[0])):
            matrice_minor = minor(matrix, i, j)

            det_minor = matrice_minor[0][0] * matrice_minor[1][1] - matrice_minor[0][1] * matrice_minor[1][0]

            semn = (-1) ** (i + j)
            valoare_cofactor = semn * det_minor

            rand.append(valoare_cofactor)
        rezultat.append(rand)
    return rezultat


def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    matrice_cofactor = cofactor(matrix)

    randuri = len(matrice_cofactor)
    coloane = len(matrice_cofactor[0])

    transpusa = []
    for j in range(coloane):
        rand = []
        for i in range(randuri):
            rand.append(matrice_cofactor[i][j])
        transpusa.append(rand)

    return transpusa


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    a11 = matrix[0][0]
    a12 = matrix[0][1]
    a13 = matrix[0][2]
    a21 = matrix[1][0]
    a22 = matrix[1][1]
    a23 = matrix[1][2]
    a31 = matrix[2][0]
    a32 = matrix[2][1]
    a33 = matrix[2][2]

    det_A = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
    adj_A = adjoint(matrix)

    inversa = []
    for i in range(len(adj_A)):
        rand = []
        for j in range(len(adj_A[0])):
            rand.append(adj_A[i][j] / det_A)
        inversa.append(rand)

    solutie = []
    for i in range(len(inversa)):
        suma = 0
        for j in range(len(vector)):
            suma += inversa[i][j] * vector[j]
        solutie.append(suma)
    return solutie

# print(f"minor(A, 0, 0) = {minor(A, 0, 0)}")
# print(f"cofactor(A) = {cofactor(A)}")
# print(f"adjoint(A) = {adjoint(A)}")
print(f"solve(A, B) = {solve(A, B)}")
