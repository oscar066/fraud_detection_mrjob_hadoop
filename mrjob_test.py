
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONValueProtocol, RawValueProtocol
import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import sys
sys.argv=['0']

class MRDecisionTree(MRJob):
    OUTPUT_PROTOCOL = JSONValueProtocol

    def steps(self):
        return [
            MRStep(mapper=self.mapper,
                   reducer=self.reducer),
            MRStep(reducer=self.reducer_final)
        ]

    def mapper(self, _, line):
        yield 'key', line

    def reducer(self, key, values):
        df = pd.read_csv(values)
        X = df.drop('is_fraud',axis=1)
        y = df['is_fraud']
        X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        pipe = Pipeline([('scaler', StandardScaler()), ('decision_tree', DecisionTreeClassifier())])
        param_grid = {'decision_tree__max_depth': [2, 4, 6, 8, 10, None]}
        grid = GridSearchCV(pipe, param_grid, cv=5)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        yield 'accuracy', accuracy_score(y_test, y_pred)
        yield 'precision', precision_score(y_test, y_pred)
        yield 'recall', recall_score(y_test, y_pred)
        yield 'f1', f1_score(y_test, y_pred)
        yield 'roc_auc', roc_auc_score(y_test, y_pred)
        yield 'confusion_matrix', confusion_matrix(y_test, y_pred)

    def reducer_final(self, key, values):
        yield key, np.mean(list(values))


if __name__ == '__main__':
    MRDecisionTree.run()