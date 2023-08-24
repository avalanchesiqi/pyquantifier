from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import beta, uniform


class UnivariateDistribution(ABC):
    @abstractmethod
    def get_density(self, val):
        pass

    @abstractmethod
    def sample(self, n):
        pass


class DiscreteUnivariateDistribution(UnivariateDistribution, ABC):
    def get_density(self, label: str):
        pass


class Bernoulli(DiscreteUnivariateDistribution):
    def __init__(self, a, b):
        self.dist = beta(a, b)

    def get_density(self, label):
        pass

    def sample(self, n):
        pass


class ContinuousUnivariateDistribution(UnivariateDistribution, ABC):
    def get_density(self, score: float):
        pass


class BetaDistribution(ContinuousUnivariateDistribution):
    def __init__(self, a, b):
        self.dist = beta(a, b)

    def get_density(self, score):
        return self.dist.pdf(score)

    def sample(self, n):
        return self.dist.rvs(n)


class MixtureDistribution(ContinuousUnivariateDistribution):
    def __init__(self, components, weights):
        self.components = components
        self.num_component = len(components)
        weight_sum = sum(weights)
        self.weights = [weight / weight_sum for weight in weights]

    def get_density(self, score):
        return sum([weight * component.pdf(score)
                    for weight, component in
                    zip(self.weights, self.components)])

    def sample(self, n):
        component_choices = np.random.choice(range(self.num_component),
                                             size=n,
                                             p=self.weights)
        component_samples = [component.rvs(size=n)
                             for component in self.components]
        data_sample = np.choose(component_choices, component_samples)
        return data_sample
