"""Agrupación de casas de California (K-Means) + clasificador supervisado sobre las etiquetas."""
import os
import pickle

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_URL = "https://breathecode.herokuapp.com/asset/internal-link?id=809&path=housing.csv"
LOCAL_PATH = os.path.join(BASE_DIR, "housing.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURES = ["Latitude", "Longitude", "MedInc"]
RANDOM_STATE = 42

# Paso 1: carga y división
data = pd.read_csv(LOCAL_PATH if os.path.exists(LOCAL_PATH) else DATA_URL)[FEATURES]
X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)

# Paso 2: K-Means con 6 clusters
kmeans = KMeans(n_clusters=6, n_init=10, random_state=RANDOM_STATE).fit(X_train)
X_train = X_train.assign(cluster=pd.Categorical(kmeans.labels_))

# Paso 3: predicción de clusters para test
X_test = X_test.assign(cluster=pd.Categorical(kmeans.predict(X_test[FEATURES])))

# Paso 4: clasificador supervisado (las fronteras de K-Means son lineales -> regresión logística)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=100))
clf.fit(X_train[FEATURES], X_train["cluster"])
y_pred = clf.predict(X_test[FEATURES])
print(f"Accuracy test: {accuracy_score(X_test['cluster'], y_pred):.4f}")
print(classification_report(X_test["cluster"], y_pred, digits=4))

# Paso 5: guardado de modelos
os.makedirs(MODELS_DIR, exist_ok=True)
with open(os.path.join(MODELS_DIR, "kmeans_6_clusters_42.sav"), "wb") as f:
    pickle.dump(kmeans, f)
with open(os.path.join(MODELS_DIR, "logisticregression_classifier_42.sav"), "wb") as f:
    pickle.dump(clf, f)
