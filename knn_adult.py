import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

columns = [
    "age","workclass","fnlwgt","education","education-num",
    "marital-status","occupation","relationship","race","sex",
    "capital-gain","capital-loss","hours-per-week",
    "native-country","income"
]

data = pd.read_csv("adult.data", names=columns, skipinitialspace=True)

data.replace("?", pd.NA, inplace=True)
data.dropna(inplace=True)

le = LabelEncoder()
categorical_cols = data.select_dtypes(include=["object", "string"]).columns

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

X = data.drop("income", axis=1)
y = data["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("KNN Accuracy:", round(accuracy * 100, 2), "%")