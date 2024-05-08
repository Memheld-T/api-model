import pandas as pd
from sklearn.model_selection import train_test_split
import xlrd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import ClassifierMixin
import joblib


def ingest_data(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df=df[['survived','pclass','sex','age']]
    df.dropna(axis=0,inplace=True)
    df['sex'].replace(to_replace=['male','female'],value=[0,1],inplace=True)
    
    return df

def train_model(df: pd.DataFrame) -> ClassifierMixin:
    model=KNeighborsClassifier(4)

    y = df['survived'] # colonne target
    X = df[['pclass', 'sex', 'age']] # colonnes features
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)


    model.fit(X_train, y_train) # entrainement du modele
    model.score(X_test, y_test) # évaluation du model

    score=model.score(X_test,y_test)
    print(f"model score: {score}")
    return model

if __name__=="__main__":
    df = ingest_data("train/titanic.xls")
    df=clean_data(df)
    model=train_model(df)  
    joblib.dump(model,"model_titanic.joblib")

