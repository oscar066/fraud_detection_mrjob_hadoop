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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE



class FraudDetection(MRJob):
    OUTPUT_PROTOCOL = JSONValueProtocol

    def steps(self):
        return [
            MRStep(mapper=self.mapper,
                   reducer=self.reducer),
            MRStep(reducer=self.reducer_final)
        ]
    
    def mapper(self, _, line):
        yield 'data', line.split(',')
        
    def reducer(self, key, values):
        df = pd.DataFrame(values)
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        df = df.astype({'is_fraud': 'int64'})
        df
        X = df.drop('is_fraud',axis=1)
        y = df['is_fraud']
        X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        sm = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
        # Logistic Regression
        lr = LogisticRegression()
        lr.fit(X_train_resampled, y_train_resampled)
        y_pred_lr = lr.predict(X_test)
        # Decision Tree
        dt = DecisionTreeClassifier()
        dt.fit(X_train_resampled, y_train_resampled)
        y_pred_dt = dt.predict(X_test)
        # Random Forest
        rf = RandomForestClassifier()
        rf.fit(X_train_resampled, y_train_resampled)
        y_pred_rf = rf.predict(X_test)
        # XGBoost
        xgb = XGBClassifier()
        xgb.fit(X_train_resampled, y_train_resampled)
        y_pred_xgb = xgb.predict(X_test)
        # Accuracy
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        accuracy_dt = accuracy_score(y_test, y_pred_dt)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        # Precision
        precision_lr = precision_score(y_test, y_pred_lr)
        precision_dt = precision_score(y_test, y_pred_dt)
        precision_rf = precision_score(y_test, y_pred_rf)
        precision_xgb = precision_score(y_test, y_pred_xgb)
        # Recall
        recall_lr = recall_score(y_test, y_pred_lr)
        recall_dt = recall_score(y_test, y_pred_dt)
        recall_rf = recall_score(y_test, y_pred_rf)
        recall_xgb = recall_score(y_test, y_pred_xgb)
        # F1 Score
        f1_lr = f1_score(y_test, y_pred_lr)
        f1_dt = f1_score(y_test, y_pred_dt)
        f1_rf = f1_score(y_test, y_pred_rf)
        f1_xgb = f1_score(y_test, y_pred_xgb)
        # ROC AUC Score
        roc_auc_lr = roc_auc_score(y_test, y_pred_lr)
        roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
        roc_auc_rf = roc_auc_score(y_test, y_pred_rf)
        roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb)
        # Confusion Matrix
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        cm_dt = confusion_matrix(y_test, y_pred_dt)
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        cm_xgb = confusion_matrix(y_test, y_pred_xgb)
        # Results
        results = {
            'Logistic Regression': {
                'Accuracy': accuracy_lr,
                'Precision': precision_lr,
                'Recall': recall_lr,
                'F1 Score': f1_lr,
                'ROC AUC Score': roc_auc_lr,
                'Confusion Matrix': cm_lr.tolist()
            },
            'Decision Tree': {
                'Accuracy': accuracy_dt,
                'Precision': precision_dt,
                'Recall': recall_dt,
                'F1 Score': f1_dt,
                'ROC AUC Score': roc_auc_dt,
                'Confusion Matrix': cm_dt.tolist()
            },
            'Random Forest': {
                'Accuracy': accuracy_rf,
                'Precision': precision_rf,
                'Recall': recall_rf,
                'F1 Score': f1_rf,
                'ROC AUC Score': roc_auc_rf,
                'Confusion Matrix': cm_rf.tolist()
            },
            'XGBoost': {
                'Accuracy': accuracy_xgb,
                'Precision': precision_xgb,
                'Recall': recall_xgb,
                'F1 Score': f1_xgb,
                'ROC AUC Score': roc_auc_xgb,
                'Confusion Matrix': cm_xgb.tolist()
            }
        }
        yield 'results', results

    def reducer_final(self, key, values):
        yield key, list(values)
        

if __name__ == '__main__':
    FraudDetection.run()