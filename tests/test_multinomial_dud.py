import unittest
from pyquantifier.distributions import MultinomialDUD

test_case_list = [
    (['apple', 'banana', 'pear'],
     [0.7, 0.1, 0.2],
     {'apple': 0.7, 'banana': 0.1, 'pear': 0.2})
    ]


class TestMultinomialDUDDensity(unittest.TestCase):
    def setUp(self):
        pass

    def test_multinomial_dud_density(self):
        for case_num, test_case in enumerate(test_case_list):
            labels, probs, expected_result = test_case
            dud = MultinomialDUD(labels, probs)
            with self.subTest(msg=f'Checking case number {case_num+1}'):
                for label, density in expected_result.items():
                    self.assertEqual(dud.get_density(label), 
                                     density, 
                                     msg=f'Test failed: density of label "{label}" does not match expected result')


if __name__ == "__main__":
    unittest.main()