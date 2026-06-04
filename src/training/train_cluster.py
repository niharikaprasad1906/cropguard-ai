
from src.data.loader import load_csv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = load_csv("sensor/data.csv")
X = StandardScaler().fit_transform(df)

model = KMeans(n_clusters=3)
df["cluster"] = model.fit_predict(X)
df.to_csv("data/processed/clusters.csv", index=False)
