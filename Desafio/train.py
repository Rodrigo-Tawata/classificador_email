import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

# === Carregar dataset ===
df = pd.read_csv(
    "data/emails.csv", 
    sep=",", 
    quotechar='"', 
    names=["email", "categoria"], 
    header=None
)

# Remover linhas inválidas
df = df.dropna(subset=["email", "categoria"])

print("Formato do dataset:", df.shape)
print(df.head())

# === Separar features e labels ===
X = df["email"]
y = df["categoria"]

# === Split treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Pipeline com TF-IDF + RandomForest ===
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

# === Treinar modelo ===
pipeline.fit(X_train, y_train)

# Avaliação rápida
score = pipeline.score(X_test, y_test)
print(f"Acurácia no conjunto de teste: {score:.2f}")

# === Salvar modelo treinado ===
joblib.dump(pipeline, "models/pipeline.pkl")
print("Modelo salvo em models/pipeline.pkl")
