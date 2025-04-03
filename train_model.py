import pandas as pd  
import numpy as np  
import joblib  
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, classification_report  

# Load the cleaned dataset  
df = pd.read_csv("cleaned_fake_job_postings.csv")  

# Replace NaN values with empty strings  
df.fillna("", inplace=True)  

# Combine text-based features into one  
df["combined_text"] = (
    df["title"] + " " + df["location"] + " " + df["department"] + " "
    + df["company_profile"] + " " + df["description"] + " " + df["requirements"]
)  

# Define features (X) and labels (y)  
X = df["combined_text"]  
y = df["fraudulent"]  

# Convert text data into numerical form using TF-IDF  
vectorizer = TfidfVectorizer(max_features=5000)  
X_tfidf = vectorizer.fit_transform(X)  

# Split dataset into training and testing sets (80% train, 20% test)  
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)  

# Train the model  
model = LogisticRegression()  
model.fit(X_train, y_train)  

# Make predictions  
y_pred = model.predict(X_test)  

# Evaluate model  
accuracy = accuracy_score(y_test, y_pred)  
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")  
print(classification_report(y_test, y_pred))  

# Save the model and vectorizer  
joblib.dump(model, "job_fake_detection_model.pkl")  
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")  
print("✅ Model and vectorizer saved successfully!")  
