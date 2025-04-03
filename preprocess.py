import pandas as pd  

# Load the dataset  
df = pd.read_csv("fake_job_postings.csv")  

# Display the first few rows  
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop columns with too many missing values (if any)
df = df.dropna(axis=1, how='all')

# Fill missing values in text-based columns with an empty string
df.fillna("", inplace=True)

# Selecting relevant columns
df = df[['title', 'location', 'department', 'company_profile', 'description', 'requirements', 'fraudulent']]

# Convert fraudulent column to categorical (0 = Real, 1 = Fake)
df['fraudulent'] = df['fraudulent'].astype(int)

# Save the cleaned dataset
df.to_csv("cleaned_fake_job_postings.csv", index=False)
print("✅ Dataset cleaned and saved as 'cleaned_fake_job_postings.csv'!")
