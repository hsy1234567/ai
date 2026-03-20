def solve_n_queens(n):
    def is_safe(board, row, col):
        for i in range(row):
            # 使用括号换行，避免使用反斜杠
            if (board[i] == col or
                board[i] - i == col - row or
                board[i] + i == col + row):
                return False
        return True

    def backtrack(row, board):
        if row == n:
            # 构建棋盘表示：每行一个字符串，'Q' 表示皇后，'.' 表示空
            solution = []
            for i in range(n):
                line = '.' * board[i] + 'Q' + '.' * (n - board[i] - 1)
                solution.append(line)
            solutions.append(solution)
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                backtrack(row + 1, board)

    solutions = []
    board = [-1] * n
    backtrack(0, board)
    return solutions