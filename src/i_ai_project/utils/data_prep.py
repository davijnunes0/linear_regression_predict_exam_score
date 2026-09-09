import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from i_ai_project.utils.paths import read_dataset, path_resolve


CSV_PATH = "Student Performance Factors.csv"
TARGET = "Exam_Score"

ORDINAL_ORDERS = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
}

def prepare_data():
    #1. Carrega e separa features do alvo
    df = read_dataset(path_resolve(2, CSV_PATH))
    X = df.drop(columns=[TARGET])
    y = df[TARGET]


    # Detecta tipos automaticamente
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # ordinais = categóricas que tem ordem definida acima; o resto é nominal
    ordinal_cols = [c for c in categorical_cols if c in ORDINAL_ORDERS]
    nominal_cols = [c for c in categorical_cols if c not in ORDINAL_ORDERS]


    # 3. split antes tudo - teste fica invisível desce o início (sem leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.20, random_state=42
    )

    # imputação - estatisticas calculadas SÒ no treino, aplicadas nos dois
    for col in numeric_cols:
        mediana = X_train[col].median() # Mediana do treino (ignora NAN)
        X_train[col] = X_train[col].fillna(mediana)
        X_test[col] = X_test[col].fillna(mediana)
    

        # 5. encoding ordinal na mão — Low/Medium/High viram 0/1/2
    for col, order in ORDINAL_ORDERS.items():
        mapping = {categoria: i for i, categoria in enumerate(order)}
        X_train[col] = X_train[col].map(mapping)
        X_test[col] = X_test[col].map(mapping)
        # categoria desconhecida vira NaN no map -> vira -1
        X_train[col] = X_train[col].fillna(-1)
        X_test[col] = X_test[col].fillna(-1)

    # 6. one-hot nas nominais
    X_train = pd.get_dummies(X_train, columns=nominal_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=nominal_cols, drop_first=True)

    # 7. alinha as colunas — o teste pode ter categoria que o treino não tinha
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # garante tudo float (dummies saem como bool)
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    # 8. z-score com média/desvio do TREINO — mesmo motivo do passo 4
    mean = X_train.mean()
    std = X_train.std().replace(0, 1)  # std 0 daria divisão por zero
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test