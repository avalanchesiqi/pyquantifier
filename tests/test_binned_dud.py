import unittest
from pyquantifier.distributions import BinnedDUD

test_case_list = [
    (['apple', 'apple', 'banana', 'pear', 'pear', 'pear'] * 100,
     {'apple': 0.3333, 'banana': 0.1667, 'pear': 0.5})
    ]


class TestBinnedDudDensity(unittest.TestCase):
    def setUp(self):
        pass

    def test_binned_dud_density(self):
        for case_num, test_case in enumerate(test_case_list):
            data, expected_result = test_case
            dud = BinnedDUD(data)
            with self.subTest(msg=f'Checking case number {case_num+1}'):
                for label, density in expected_result.items():
                    self.assertAlmostEqual(dud.get_density(label), 
                                           density, 
                                           places=3, 
                                           msg=f'Test failed: density of label "{label}" does not match expected result'
                                           )


if __name__ == "__main__":
    unittest.main()
