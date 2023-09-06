from pyquantifier.distributions import BinnedDUD


# generate a unit test for the BinnedDUD class
def test_binned_dud():
    data = ['apple', 'apple', 'banana', 'pear', 'pear', 'pear'] * 100
    dud = BinnedDUD(data)
    dud.plot(ci=True)
    print(dud.sample(3))
    print(dud.get_density('apple'))
    print(dud.get_density('banana'))
    print(dud.get_density('pear'))


test_binned_dud()
