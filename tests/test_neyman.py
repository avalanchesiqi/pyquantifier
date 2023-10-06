import unittest
from pyquantifier.data import Dataset

param_list = [({0: 55806, 1: 37635, 2: 11456, 3: 8512, 4: 6406, 5: 4923, 6: 4508, 7: 4792, 8: 6588, 9: 5217}, 400,
               {0: 101, 1: 112, 2: 41, 3: 34, 4: 27, 5: 20, 6: 18, 7: 17, 8: 20, 9: 9}), 
              ({0: 116971, 1: 16457, 2: 4609, 3: 2046, 4: 1247, 5: 1069, 6: 1156, 7: 1456, 8: 1536, 9: 15}, 400,
               {0: 274, 1: 63, 2: 21, 3: 10, 4: 7, 5: 6, 6: 6, 7: 7, 8: 6, 9: 0}), 
              ({0: 16305, 1: 8266, 2: 8390, 3: 9212, 4: 16248, 5: 12285, 6: 15839, 7: 17256, 8: 17064, 9: 25697}, 400,
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
