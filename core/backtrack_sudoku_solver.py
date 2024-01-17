#backtracking algorithm

def is_valid(grid, r, c, k):
    not_in_row = k not in grid[r]
    not_in_column = k not in [grid[i][c] for i in range(9)]
    not_in_box = k not in [grid[i][j] for i in range(r//3*3, r//3*3+3) for j in range(c//3*3, c//3*3+3)]
    return not_in_row and not_in_column and not_in_box


def solve(grid, r=0, c=0):
    if r == 9:
        return True
    elif c == 9:
        return solve(grid, r+1, 0)
    elif grid[r][c] != 0:
        return solve(grid, r, c+1)
    else:
        for k in range(1, 10):
            if is_valid(grid, r, c, k):
                grid[r][c] = k
                if solve(grid, r, c+1):
                    return True
                grid[r][c] = 0
        return False
    
    

#create 2d sudoku array shape
def dimension_array(sudoku_holder_model):
    sudoku_board_vir = [[]]

    row = 0 #start at first row
    for counter, num in enumerate(sudoku_holder_model): #loop by column
        if counter % 9 == 0 and counter != 0:
            sudoku_board_vir.append([])
            row += 1
            
        sudoku_board_vir[row].append(num)

    return sudoku_board_vir


def main_backtrack_solve(sudoku):
    grid = sudoku.copy()
    board = dimension_array(grid)
    solve(board)

    sudoku_solved = []
    for i,num in enumerate(board):
        for j, inner_num in enumerate(num):
            sudoku_solved.append(inner_num)
    
    
    return sudoku_solved