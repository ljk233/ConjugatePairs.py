
'''
'''

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats as st
from scipy import optimize
from numpy.typing import NDArray, ArrayLike
import seaborn as sns


@dataclass
class Prior:
    """A stub superclass that does nothing by itself.
    It is used to represent all the Prior classes defined.
    """

    @property
    def model(self):
        """Return the model as a scipy rv object.
        """
        raise NotImplementedError('Use a concrete Prior')

    @classmethod
    def from_belief(cls, *args, **kwargs):
        """Return an initialised prior based on some belief about the
        location of theta.

        Preconditions:
        - Arguments are valid values for the modelled parameter.
        """
        raise NotImplementedError('Use a concrete Prior')

    @classmethod
    def from_obs(cls, *args, **kwargs):
        raise NotImplementedError('Use a concrete Prior')

    def fit(self):
        """Return the fitted theta as a scipy rv object.
        """
        raise NotImplementedError('Use a concrete Prior')

    def is_uniform(self) -> bool:
        """Return true if self is an improper prior.
        Otherwise false.
        """
        raise NotImplementedError('Use a concrete Prior')

    def to_posterior(self, *args, **kwargs):
        """Return a new instance of the prior, updated to match the given
        observations.
        """
        raise NotImplementedError('Use a concrete Prior')


@dataclass
class Beta(Prior):
    """A class to model the use of a beta conjugate prior distribution.

    The beta distribution is used to model the likelihood of p in a
    binominal or Bernoulli distribition.
    """
    a: float = 1.0
    b: float = 1.0

    @property
    def model(self) -> st.rv_continuous:
        # return the distribution for theta
        return st.beta(self.a, self.b)

    @classmethod
    def from_belief(
            cls,
            mode: float,
            low: float,
            high: float,
            spread: float = 0.5
    ) -> Prior:
        def get_a(beta):
            anum = (2 * mode) - (beta * mode) - 1
            aden = mode - 1
            return anum / aden

        def minimise_iqr(b):
            uq = st.beta.cdf(high, get_a(b), b)
            lq = st.beta.cdf(low, get_a(b), b)
            return abs(uq - lq - spread)

        b = (
            optimize.minimize_scalar(
                    minimise_iqr,
                    method='brent',
                    options={'xtol': 1e-8}
                )
        )
        return cls(get_a(b.x), b.x)

    @classmethod
    def from_obs(cls, obs: NDArray) -> Prior:
        return cls().to_posterior(obs.size, obs.sum())

    def fit(self, n: int) -> st.rv_discrete:
        """Return bernoulli(self.theta)
        """
        return st.binom(n, self.model.median())

    def is_uniform(self) -> bool:
        return False

    def to_posterior(self, arr: ArrayLike) -> Prior:
        n, x = len(arr), sum(arr)
        return type(self)(self.a + x, self.b + n-x)

    def __str__(self) -> str:
        return f'Beta({self.a:.2f}, {self.b:.2f})'


@dataclass
class Gamma(Prior):
    """A class to model the use of a gamma conjugate prior likelihood.
    """
    a: float = 0.0
    b: float = 0.0

    @property
    def model(self) -> st.rv_continuous:
        return st.gamma(self.a, scale=1/self.b)

    @classmethod
    def from_belief(
            cls,
            loc: float,
            low: float,
            high: float,
            spread: float = 0.5
    ) -> Prior:
        def get_a(b):
            return (loc * b) + 1

        def minimise_spread(b):
            return abs(
                st.gamma.cdf(high, get_a(b), scale=1/b)
                - st.gamma.cdf(low, get_a(b), scale=1/b)
                - spread
            )

        b = optimize.minimize_scalar(
                            minimise_spread,
                            bounds=(0.000000001, 2*high),
                            method='bounded'
        )
        return cls(get_a(b.x), b.x)

    @classmethod
    def from_obs(cls, obs: NDArray) -> Prior:
        return cls().to_posterior(obs)

    def fit(self) -> st.rv_continuous:
        return st.poisson(self.model.median())

    def is_uniform(self) -> bool:
        return self.a == 0.0 or self.b == 0.0

    def to_posterior(self, obs: NDArray) -> Prior:
        return type(self)(self.a + sum(obs), self.b + len(obs))

    def __str__(self) -> str:
        return f'Gamma({self.a:.2f}, {self.b:.2f})'


@dataclass
class Normal(Prior):
    """A class to model the use of a normal conjugate prior likelihood.
    """
    known_var: float
    a: float = 0.0
    b: float = 0.0

    @property
    def model(self) -> st.rv_continuous:
        return st.norm(self.a, np.sqrt(self.b))

    @classmethod
    def from_belief(
            cls,
            mode: float,
            known_var: float,
            low: float,
            high: float,
    ) -> Beta:
        return cls(
            known_var,
            a=mode,
            b=round(np.square((high-low) / (2 * 0.6745)), 2)
        )

    @classmethod
    def from_obs(cls, obs: NDArray) -> Beta:
        return cls().to_posterior(obs)

    def fit(self) -> st.rv_continuous:
        return st.norm(self.model.median(), np.sqrt(self.known_var))

    def is_uniform(self) -> bool:
        return self.b == 0.0

    def to_posterior(self, obs: NDArray) -> Beta:
        xbar, n = np.mean(obs), len(obs)
        if self.is_uniform():
            return type(self)(self.known_var, xbar, n)
        anum = (self.known_var * self.a) + (n * self.b * xbar)
        bnum = self.known_var * self.b
        denom = self.known_var + (n * self.b)
        return type(self)(self.known_var, anum/denom, bnum/denom)

    def __str__(self) -> str:
        return f'Normal({self.a:.2f}, {self.b:.2f})'

# ======================================================================
#  FUNCTIONS
# ======================================================================

def credible_interval(prior: Prior, alpha: float = 0.05) -> NDArray:
    """Return the 100(1-alpha)% credible interval of theta.

    Preconditions:
    - 0 < alpha < 1
    """
    return np.array([prior.model.ppf(alpha/2),
                     prior.model.ppf(1 - alpha/2)])


def describe_model(prior: Prior) -> pd.Series:
    """Return descriptive statistics of prior.

    Preconditions:
    - not prior.is_improper()
    """
    return pd.Series(
        [prior.model.median(),
         prior.model.var(),
         *credible_interval(prior)],
        index=['median', 'var', 'lcb', 'ucb'],
    )

def plot_models(*priors: Prior) -> sns.FacetGrid:
    # return the pdf of rv as a DataFrame
    def to_frame(rv, label):
        # get the end points
        a, b = rv.a, rv.b
        # are either of the end-points undefined?
        if a == -np.inf:
            a = 0.001
        if b == np.inf:
            b = 0.999
        # return the DataFrame
        xs = np.linspace(rv.ppf(a), rv.ppf(b), 100)
        return (
            pd.DataFrame()
            .assign(
                model=[label] * len(xs),
                theta=xs,
                lik=rv.pdf(xs)
            )
        )

    gs = pd.concat([to_frame(prior.model, str(prior)) for prior in priors])
    return sns.relplot(data=gs, x='theta', y='lik', hue='model', kind='line')

def sample_fit(prior: Prior, *args: int) -> NDArray:
    ...
