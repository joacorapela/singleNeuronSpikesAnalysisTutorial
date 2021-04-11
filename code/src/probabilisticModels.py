
import pdb
import abc
import numpy as np

class ProbabilisticModel(abc.ABC):
    @abc.abstractmethod
    def train(self, x):
        pass

    @abc.abstractmethod
    def logLikelihood(self, x):
        pass

class Exponential(ProbabilisticModel):

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value

    def train(self, x):
        self._lambda = 1/x.mean()

    def density(self, x):
        if x>=0:
            answer = self._lamba*np.exp(-self._lambda*x)
        else:
            answer = 0
        return answer

    def logLikelihood(self, x):
        N = len(x)
        answer = N*np.log(self._lambda)*(1-x.mean())
        return answer

class InverseGaussian(ProbabilisticModel):
    def train(self, x):
        self._mu = x.mean()
        self._lambda = 1/(1/x-1/self._mu).mean()

    @property
    def lbda(self):
        return self._lambda

    @lbda.setter
    def lbda(self, value):
        self._lambda = value

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value

    def density(self, x):
        answer = np.sqrt(self._lambda/(2*np.pi*x**3))*np.exp(-self._lambda*(x-self._mu)**2/(2*self._mu**2*x))
        return answer

    def logLikelihood(self, x):
        N = len(x)
        answer = N/2*np.log(self._lambda/(2*np.pi))
        answer -= 3.0/2.0*np.log(x).sum()
        answer -= self._lambda*np.divide((x-self._mu)**2, 2*x*self._mu**2).sum()
        return answer
