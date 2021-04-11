
import sys
import pdb
import numpy as np
sys.path.append("../src")
import probabilisticModels

class NaiveBayes:
    def train(self, x, y, model_class=probabilisticModels.Exponential):
        self._models = {}
        for i, x_item in enumerate(x):
            self._models[y[i]] = model_class()
            self._models[y[i]].train(x=x_item)

    def classify(self, x):
        maxLogLikeValue = -np.Inf
        maxLogLikeClass = None
        for class_label, model in self._models.items():
            logLike = model.logLikelihood(x=x)
            if logLike>maxLogLikeValue:
                maxLogLikeValue = logLike
                maxLogLikeClass = class_label
        return maxLogLikeClass
