import unittest
from pyquantifier.data import Dataset

param_list = [([55806, 37635, 11456, 8512, 6406, 4923, 4508, 4792, 6588, 5217], 400,
               {0: 101, 1: 112, 2: 41, 3: 34, 4: 27, 5: 20, 6: 18, 7: 17, 8: 20, 9: 9}), 
              ([116971, 16457, 4609, 2046, 1247, 1069, 1156, 1456, 1536, 15], 400,
               {0: 274, 1: 63, 2: 21, 3: 10, 4: 7, 5: 6, 6: 6, 7: 7, 8: 6, 9: 0}), 
              ([16305, 8266, 8390, 9212, 16248, 12285, 15839, 17256, 17064, 25697], 400,
               {0: 26, 1: 21, 2: 26, 3: 32, 4: 58, 5: 44, 6: 55, 7: 54, 8: 44, 9: 40})]

class TestGetNeymanAllocation(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_neyman_allocation(self):
        for case_num, param in enumerate(param_list):
            result = Dataset._get_neyman_allocation(param[0], param[1])
            with self.subTest(msg=f'Checking case number {case_num+1}'):
                self.assertDictEqual(result, param[2], 'Test failed: output does not match expected result')
        
if __name__ == "__main__":
    unittest.main()
