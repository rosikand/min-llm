import unittest

# math_utils.py
def add(a, b):
    return a + b


class TestAddFunction(unittest.TestCase):
    def test_add_positive_numbers(self):
        result = add(2, 3)
        self.assertEqual(result, 5)

    def test_add_negative_numbers(self):
        result = add(-2, -3)
        self.assertEqual(result, -5)

    def test_add_mixed_numbers(self):
        result = add(-2, 3)
        self.assertEqual(result, 12)

    def test_add_zero(self):
        result = add(0, 0)
        self.assertEqual(result, 0)

if __name__ == "__main__":
    unittest.main()