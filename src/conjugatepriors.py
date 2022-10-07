
'''
'''

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats as st
from scipy import optimize
from numpy.typing import NDArray


@dataclass
class _Prior:
    a: float = 1.0
    b: float = 1.0

    @property
    def params(self) -> NDArray:
        # return self.a, self.b as an NDArray
        return np.array([self.a, self.b])

    @property
    def rv_theta(self):
        """
        """
        raise NotImplementedError('Use a concrete Prior')

    @property
    def theta(self):
        """Return the likely location of theta.

        Preconditions:
        - not self.is_improper()
        """
        return self.rv_theta.median()

    @classmethod
    def from_belief(cls):
        """Return an initialised prior based on some belief about the
        location of theta.

        Preconditions:
        - Arguments are valid values for the modelled parameter.
        """
        raise NotImplementedError('Use a concrete Prior')

    @classmethod
    def from_obs(cls):
        raise NotImplementedError('Use a concrete Prior')

    def credible_interval(self, alpha: float = 0.5) -> NDArray:
        """Return the 100(1-alpha)% credible interval of theta.

        Preconditions:
        - 0 < alpha < 1
        """
        return np.array([self.rv_theta.ppf(alpha/2), self.rv_theta.ppf(1 - alpha/2)])

    def describe(self) -> NDArray:
        """Return descriptive statistics of prior.
        """
        return pd.Series(
            [self.theta, self.rv_theta.var(), *self.credible_interval()],
            index=['median', 'var', 'lcb', 'ucb'],
            name=str(self)
        )

    def is_improper(self) -> bool:
        """Return true if self is an improper prior.
        """
        raise NotImplementedError('Use a concrete Prior')

    def stats(self):
        """Return descriptive statistics of prior.
        """

    def update(self):
        """Return a new updated instance of the prior.
        """
        raise NotImplementedError('Use a concrete Prior')


@dataclass(frozen=True)
class Beta(_Prior):
    """A class to model the use of a beta conjugate prior distribution.

    The beta distribution is used to model the likelihood of p in a
    binominal or Bernoulli distribition.
    """
    a: float = 1.0
    b: float = 1.0

    @property
    def rv_theta(self) -> st.rv_continuous:
        # return the distribution for theta
        return st.beta(self.a, self.b)

    @classmethod
    def from_belief(
            cls,
            loc: float,
            low: float,
            high: float,
            spread: float = 0.5
    ) -> Beta:
        def get_a(beta):
            anum = (2 * loc) - (beta * loc) - 1
            aden = loc - 1
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
    def from_obs(cls, obs: NDArray) -> Beta:
        return cls().update(obs.size, obs.sum())

    def rv_bernoulli(self) -> st.rv_discrete:
        """Return binom(n, self.theta)
        """
        return st.bernoulli(self.theta)

    def rv_binom(self, n: int) -> st.rv_discrete:
        """Return bernoulli(self.theta)
        """
        return st.binom(n, self.theta)

    def is_improper(self) -> bool:
        return False

    def update(self, nobs: int, succs: int) -> Beta:
        return type(self)(self.a + succs, self.b + nobs - succs)

    def __str__(self) -> str:
        return f'Beta({self.a:.2f}, {self.b:.2f})'


@dataclass(frozen=True)
class _Gamma(_Prior):
    """A class to model the use of a gamma conjugate prior likelihood.
    """
    a: float = 1.0
    b: float = 1.0

    @property
    def rv_theta(self) -> st.rv_continuous:
        # return the distribution for theta
        return st.gamma(self.a, self.b)

    @classmethod
    def from_belief(
            cls,
            loc: float,
            low: float,
            high: float,
            spread: float = 0.5
    ) -> Beta:
        def get_a(beta):
            anum = (2 * loc) - (beta * loc) - 1
            aden = loc - 1
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
    def from_obs(cls, obs: NDArray) -> Beta:
        return cls().update(obs.size, obs.sum())

    def rv_bernoulli(self) -> st.rv_discrete:
        """Return binom(n, self.theta)
        """
        return st.bernoulli(self.theta)

    def rv_binom(self, n: int) -> st.rv_discrete:
        """Return bernoulli(self.theta)
        """
        return st.binom(n, self.theta)

    def is_improper(self) -> bool:
        return False

    def update(self, nobs: int, succs: int) -> Beta:
        return type(self)(self.a + succs, self.b + nobs - succs)

    def __str__(self) -> str:
        a = round(self.a, 2)
        b = round(self.b, 2)
        return f'Beta({a}, {b})'

