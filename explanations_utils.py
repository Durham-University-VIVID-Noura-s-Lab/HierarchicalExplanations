import pandas as pd
import numpy as np
def softmax(y_linear):
    exp = np.exp(y_linear-np.max(y_linear, axis=1).reshape((-1,1)))
    norms = np.sum(exp, axis=1).reshape((-1,1))
    return exp / norms
class Predictor():
    def __init__(self,model,features,preprocessor=None,other_processor=None):
        self.model = model
        self.preprocessor = preprocessor
        self.features = features
        self.other_processor = other_processor
    def predict_proba(self,X):
        x_test = pd.DataFrame(X, columns=self.features).astype('int32')
        if self.other_processor:
            x_test = self.other_processor(x_test)
        if self.preprocessor:
            x_test = self.preprocessor.transform(x_test)
        return self.model.predict_proba(x_test)
    def predict(self,X):
        return self.predict_proba(X).argmax(-1)
    def __call__(self,X):
        return self.predict_proba(X)