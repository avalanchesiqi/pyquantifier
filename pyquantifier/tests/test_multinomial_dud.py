from pyquantifier.distributions import MultinomialDUD


# generate a unit test for the MultinomialDUD class
def test_multinomial_dud():
    labels = ['apple', 'banana', 'pear']
    probs = [0.7, 0.1, 0.2]
    dud = MultinomialDUD(labels, probs)
    dud.plot()
    print(dud.generate_data(10))
    print(dud.get_density('apple'))
    print(dud.get_density('banana'))
    print(dud.get_density('pear'))


test_multinomial_dud()
