
import unittest
import pysal


class Test_Components(unittest.TestCase):
    def setUp(self):
        self.w = pysal.lat2W(5, 5)

    def test_check_contiguity(self):
        result = pysal.region.check_contiguity(self.w, [0, 1, 2, 3, 4], 4)
        self.assertEqual(result, True)
        result = pysal.region.check_contiguity(self.w, [0, 1, 2, 3, 4], 3)
        self.assertEqual(result, False)
        result = pysal.region.check_contiguity(self.w, [0, 1, 2, 3, 4], 0)
        self.assertEqual(result, True)
        result = pysal.region.check_contiguity(self.w, [0, 1, 2, 3, 4], 1)
        self.assertEqual(result, False)


suite = unittest.TestLoader().loadTestsFromTestCase(Test_Components)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
