import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

file_path = 'bank_marketing_sample_dataset.xlsx'
df = pd.read_excel(file_path)

df['Job'] = df['Job'].astype('category').cat.codes
df['Marital'] = df['Marital'].astype('category').cat.codes
df['Education'] = df['Education'].astype('category').cat.codes
df['Default'] = df['Default'].map({'no': 0, 'yes': 1})
df['Housing'] = df['Housing'].map({'no': 0, 'yes': 1})
df['Loan'] = df['Loan'].map({'no': 0, 'yes': 1})
df['Contact'] = df['Contact'].astype('category').cat.codes
df['POutcome'] = df['POutcome'].astype('category').cat.codes

X = df.drop(columns=['Purchase'])
y = df['Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
