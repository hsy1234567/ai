import unittest
from hw01.src.queens import solve_n_queens  # 确认函数名一致

class TestNQueens(unittest.TestCase):
    def test_n4(self):
        solutions = solve_n_queens(4)
        self.assertEqual(len(solutions), 2)

    def test_n8(self):
        solutions = solve_n_queens(8)
        self.assertEqual(len(solutions), 92)  # 修正：比较值，不是取模

    def test_n1(self):
        solutions = solve_n_queens(1)
        self.assertEqual(len(solutions), 1)

    def test_n2(self):
        solutions = solve_n_queens(2)
        self.assertEqual(len(solutions), 0)

if __name__ == '__main__':
    unittest.main()