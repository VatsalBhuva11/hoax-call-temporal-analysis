import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Load data
df = pd.read_csv("../data/6_sorted_quoted.csv")

# Load sentence embeddings (if not already saved in DataFrame)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_message(msg):
    return [word.strip().lower() for word in str(msg).split(',') if word.strip()]

df['tokens'] = df['message'].apply(preprocess_message)
df['clean_sentence'] = df['tokens'].apply(lambda words: ' '.join(words))
df['embedding'] = df['clean_sentence'].apply(lambda x: model.encode(x))

# Convert embeddings and labels
X = np.vstack(df['embedding'].to_numpy())
y = df['label'].apply(lambda x: 1 if x.strip().lower() == 'fraud' else 0).to_numpy()

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Train classifier
clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Step 4: Evaluate classifier
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['normal', 'fraud'])

print("📊 Classification Report:\n")
print(report)

# Step 5: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['normal', 'fraud'])

# Create output directory
os.makedirs("../result", exist_ok=True)

# Save confusion matrix
plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Logistic Regression)")
plt.savefig("../result/confusion_matrix.png", bbox_inches='tight')
plt.close()

# Step 6: Save classification report as text
with open("../result/classification_report.txt", "w") as f:
    f.write(report)

# Optional: Save prediction vs actual as CSV
df_test = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred
})
df_test.to_csv("../result/test_predictions.csv", index=False)

print("✅ Model trained and results saved to ../result")
