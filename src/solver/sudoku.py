import numpy as np



def print_solution(grid):
    for i in range(len(grid)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")
        for j in range(len(grid[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(grid[i][j])
            else:
                print(str(grid[i][j]) + " ", end="")




def solve(grid):

    solutions = []
    find_solution(grid, solutions)

    if len(solutions) == 0:
        print("El sudoku no tiene solución")

    elif len(solutions) > 1:
        print("El sudoku tiene multiples soluciones")

        for sol in solutions:
            print_solution(sol)

    else:
        print("Solución Encontrada")
        print_solution(solutions[0])


def posible(y, x, n, grid):

    # chequeo los elementos de la misma fila
    if n in grid[y, :]:
        return False

    # chequeo los elementos de la misma columna
    if n in grid[:, x]:
        return False

    # Controlar que el elemento no esté en el sub cuadrado
    inicio_x = (x//3)*3
    inicio_y = (y//3)*3

    for i in range(3):
        for j in range(3):
            if grid[inicio_y + i, inicio_x + j] == n:
                return False
    return True


def find_solution(grid, solutions):

    for y in range(9):
        for x in range(9):
            if grid[y, x] == 0:
                for n in range(1, 10):
                    if posible(y, x, n, grid):
                        grid[y, x] = n
                        find_solution(grid, solutions)
                        grid[y, x] = 0
                # Si no fue posible poner ningún numero, volve atras
                return

    solutions.append(np.copy(grid))


sudoku = [
        [6, 1, 0, 0, 0, 0, 0, 5, 4],
        [3, 0, 8, 0, 0, 4, 0, 0, 7],
        [0, 0, 0, 0, 2, 0, 0, 1, 0],

        [0, 8, 0, 0, 0, 0, 0, 0, 3],
        [5, 0, 1, 2, 7, 0, 0, 0, 0],
        [0, 0, 0, 8, 0, 1, 0, 7, 0],

        [0, 0, 0, 0, 5, 0, 6, 3, 0],
        [9, 5, 3, 1, 0, 6, 7, 4, 0],
        [1, 7, 0, 4, 3, 2, 8, 9, 0]]
sudoku = np.matrix(sudoku)


solve(sudoku)
