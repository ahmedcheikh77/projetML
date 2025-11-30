# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the data
df = None
try:
    # Try loading from local file
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("✅ Dataset loaded successfully from local file!")
except FileNotFoundError:
    print("❌ Local file not found. Checking other locations...")
    try:
        # Try different possible file locations
        df = pd.read_csv('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        print("✅ Dataset loaded from ./data/ folder!")
    except:
        try:
            df = pd.read_csv('../WA_Fn-UseC_-Telco-Customer-Churn.csv')
            print("✅ Dataset loaded from parent directory!")
        except:
            print("❌ Could not find the dataset locally.")
            print("Please download it from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
            print("And make sure it's in the same folder as your Python script.")

# Only proceed if df was successfully loaded
if df is not None:
    # Initial data exploration
    print("=" * 50)
    print("DATASET SHAPE:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n" + "=" * 50)
    print("COLUMN NAMES AND DATA TYPES:")
    print(df.info())

    print("\n" + "=" * 50)
    print("FIRST 5 ROWS:")
    print(df.head())

    print("\n" + "=" * 50)
    print("BASIC STATISTICS:")
    print(df.describe())

    print("\n" + "=" * 50)
    print("MISSING VALUES:")
    print(df.isnull().sum())

    print("\n" + "=" * 50)
    print("TARGET VARIABLE DISTRIBUTION (Churn):")
    print(df['Churn'].value_counts())
    print("\nChurn Rate:")
    print(df['Churn'].value_counts(normalize=True) * 100)

    # Let's also check the current working directory
    print("\n" + "=" * 50)
    print("CURRENT WORKING DIRECTORY:")
    print(os.getcwd())
    print("\nFILES IN CURRENT DIRECTORY:")
    print([f for f in os.listdir('.') if f.endswith('.csv')])
else:
    print("\n❌ Please download the dataset first from Kaggle:")
    print("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    print("Save it as 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in your project folder")

# Step 2: Data Cleaning & Preprocessing
print("=" * 60)
print("STEP 2: DATA CLEANING & PREPROCESSING")
print("=" * 60)

# 2.1 Check for duplicate rows
print("2.1 DUPLICATE ROWS:")
print(f"Number of duplicate rows: {df.duplicated().sum()}")

# 2.2 Check for unique values in each column
print("\n" + "=" * 50)
print("2.2 UNIQUE VALUES IN EACH COLUMN:")
for column in df.columns:
    unique_count = df[column].nunique()
    print(f"{column}: {unique_count} unique values")
    if unique_count <= 10:  # Show values for columns with few unique values
        print(f"   Values: {df[column].unique()}")

# 2.3 Examine the TotalCharges column issue we noticed earlier
print("\n" + "=" * 50)
print("2.3 TOTALCHARGES COLUMN EXAMINATION:")
print("Data type:", df['TotalCharges'].dtype)
print("Sample values:")
print(df['TotalCharges'].head(10))

# Check for empty strings or special characters in TotalCharges
total_charges_issues = df[df['TotalCharges'].str.strip() == '']
print(f"Number of empty values in TotalCharges: {len(total_charges_issues)}")

# 2.4 Check the customers with empty TotalCharges
if len(total_charges_issues) > 0:
    print("\nCustomers with empty TotalCharges:")
    print(total_charges_issues[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']])


# 2.5 DATA CLEANING OPERATIONS

# Create a copy of the dataframe for cleaning
df_clean = df.copy()

print("\n" + "=" * 50)
print("2.5 DATA CLEANING OPERATIONS")
print("=" * 50)

# Fix TotalCharges column - convert to numeric and handle empty strings
print("Fixing TotalCharges column...")
# Replace empty strings with NaN
df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
# Convert to numeric
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

# Check how many missing values we have now
print(f"Missing values after cleaning: {df_clean['TotalCharges'].isnull().sum()}")

# Handle missing values in TotalCharges
# For new customers (tenure=0), TotalCharges should equal MonthlyCharges
df_clean.loc[df_clean['tenure'] == 0, 'TotalCharges'] = df_clean.loc[df_clean['tenure'] == 0, 'MonthlyCharges']

# For any remaining missing values, fill with 0
df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)

print("✅ TotalCharges column fixed!")

# 2.6 Check other potential data quality issues
print("\n" + "=" * 50)
print("2.6 DATA QUALITY CHECKS")

# Check for impossible values
print("Tenure range:", df_clean['tenure'].min(), "to", df_clean['tenure'].max())
print("MonthlyCharges range: ${:.2f} to ${:.2f}".format(
    df_clean['MonthlyCharges'].min(), df_clean['MonthlyCharges'].max()))
print("TotalCharges range: ${:.2f} to ${:.2f}".format(
    df_clean['TotalCharges'].min(), df_clean['TotalCharges'].max()))

# Check for customers with tenure=0 but Churn=Yes (shouldn't happen)
tenure_zero_churn = df_clean[(df_clean['tenure'] == 0) & (df_clean['Churn'] == 'Yes')]
print(f"Customers with tenure=0 who churned: {len(tenure_zero_churn)}")

# 2.7 Final data overview after cleaning
print("\n" + "=" * 50)
print("2.7 CLEANED DATA OVERVIEW")
print("Missing values in cleaned dataset:")
print(df_clean.isnull().sum())

print("\nData types after cleaning:")
print(df_clean.dtypes)

print(f"\nFinal dataset shape: {df_clean.shape}")

# 2.8 Save the cleaned dataset for future use
df_clean.to_csv('telco_churn_cleaned.csv', index=False)
print("\n✅ Cleaned dataset saved as 'telco_churn_cleaned.csv'")

# 2.9 Basic Feature Analysis
print("\n" + "=" * 50)
print("2.9 BASIC FEATURE ANALYSIS")
print("=" * 50)

# Analyze numerical features correlation with churn
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Convert Churn to numerical for correlation analysis
churn_numeric = df_clean['Churn'].map({'Yes': 1, 'No': 0})

print("Correlation with Churn (numerical features):")
for feature in numerical_features:
    correlation = df_clean[feature].corr(churn_numeric)
    print(f"{feature}: {correlation:.3f}")

# Check basic statistics by churn status
print("\nAverage values by Churn status:")
churn_summary = df_clean.groupby('Churn')[numerical_features].mean()
print(churn_summary)

# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
print("=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 3.1 Churn Distribution (Target Variable)
print("3.1 CHURN DISTRIBUTION")
plt.figure(figsize=(15, 12))

# Plot 1: Churn distribution
plt.subplot(2, 3, 1)
churn_counts = df_clean['Churn'].value_counts()
plt.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Churn Distribution')

# 3.2 Numerical Features vs Churn
print("3.2 NUMERICAL FEATURES ANALYSIS")

# Plot 2: Tenure vs Churn
plt.subplot(2, 3, 2)
sns.boxplot(x='Churn', y='tenure', data=df_clean)
plt.title('Tenure vs Churn')
plt.ylabel('Tenure (months)')

# Plot 3: MonthlyCharges vs Churn
plt.subplot(2, 3, 3)
sns.boxplot(x='Churn', y='MonthlyCharges', data=df_clean)
plt.title('Monthly Charges vs Churn')
plt.ylabel('Monthly Charges ($)')

# Plot 4: TotalCharges vs Churn
plt.subplot(2, 3, 4)
sns.boxplot(x='Churn', y='TotalCharges', data=df_clean)
plt.title('Total Charges vs Churn')
plt.ylabel('Total Charges ($)')

# 3.3 Correlation Heatmap
plt.subplot(2, 3, 5)
# Convert Churn to numerical for correlation
df_numeric = df_clean.copy()
df_numeric['Churn'] = df_numeric['Churn'].map({'Yes': 1, 'No': 0})

# Select only numerical columns for correlation
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
correlation_matrix = df_numeric[numerical_cols].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.show()

# 3.4 Categorical Variables Analysis
print("\n3.4 CATEGORICAL VARIABLES ANALYSIS")
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'Contract', 'PaperlessBilling',
                       'PaymentMethod']

plt.figure(figsize=(20, 15))
for i, col in enumerate(categorical_columns[:6], 1):  # First 6 columns
    plt.subplot(2, 3, i)

    # Create percentage stacked bar chart
    cross_tab = pd.crosstab(df_clean[col], df_clean['Churn'], normalize='index') * 100
    cross_tab.plot(kind='bar', stacked=True, ax=plt.gca())

    plt.title(f'Churn Rate by {col}')
    plt.xlabel(col)
    plt.ylabel('Percentage (%)')
    plt.legend(title='Churn')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 3.5 Key Insights - Contract Type (Most Important!)
print("\n3.5 CONTRACT TYPE ANALYSIS - KEY INSIGHT")
plt.figure(figsize=(10, 6))

contract_churn = pd.crosstab(df_clean['Contract'], df_clean['Churn'], normalize='index') * 100
contract_churn = contract_churn.sort_values('Yes', ascending=False)

plt.subplot(1, 2, 1)
contract_churn.plot(kind='bar', stacked=True)
plt.title('Churn Rate by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)

# Raw numbers
plt.subplot(1, 2, 2)
contract_counts = pd.crosstab(df_clean['Contract'], df_clean['Churn'])
contract_counts.plot(kind='bar', stacked=True)
plt.title('Churn Count by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Print the exact percentages
print("\nCHURN RATES BY CONTRACT TYPE:")
for contract_type in contract_churn.index:
    churn_rate = contract_churn.loc[contract_type, 'Yes']
    print(f"- {contract_type}: {churn_rate:.1f}% churn rate")

# 3.6 Payment Method Analysis
print("\n3.6 PAYMENT METHOD ANALYSIS")
payment_churn = pd.crosstab(df_clean['PaymentMethod'], df_clean['Churn'], normalize='index') * 100
payment_churn = payment_churn.sort_values('Yes', ascending=False)

plt.figure(figsize=(10, 6))
payment_churn.plot(kind='bar', stacked=True)
plt.title('Churn Rate by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.show()

print("\nCHURN RATES BY PAYMENT METHOD:")
for payment_method in payment_churn.index:
    churn_rate = payment_churn.loc[payment_method, 'Yes']
    print(f"- {payment_method}: {churn_rate:.1f}% churn rate")

# STEP 4: FEATURE ENGINEERING
print("=" * 60)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 60)

# Create a copy of our cleaned data for feature engineering
df_final = df_clean.copy()

print("Original dataset shape:", df_final.shape)
print("Columns before feature engineering:", list(df_final.columns))

# 4.1 Handle Categorical Variables - Encoding
print("\n" + "=" * 50)
print("4.1 CATEGORICAL VARIABLE ENCODING")
print("=" * 50)

# 4.1.1 Label Encoding for Binary Variables
binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

binary_mapping = {'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1}

for col in binary_columns:
    if col in df_final.columns:
        df_final[col] = df_final[col].map(binary_mapping)
        print(f"✅ {col} encoded: {binary_mapping}")

# 4.1.2 One-Hot Encoding for Multi-Category Variables
multi_category_columns = ['InternetService', 'Contract', 'PaymentMethod', 'MultipleLines']

# Services columns that have 'Yes', 'No', 'No service'
service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies']

# Combine all categorical columns
all_categorical = multi_category_columns + service_columns

print(f"\nOne-Hot Encoding for {len(all_categorical)} columns...")

# Perform one-hot encoding
df_encoded = pd.get_dummies(df_final, columns=all_categorical, prefix=all_categorical, drop_first=True)

print(f"Dataset shape after encoding: {df_encoded.shape}")

# 4.2 Create New Features
print("\n" + "=" * 50)
print("4.2 CREATING NEW FEATURES")
print("=" * 50)

# 4.2.1 Tenure Groups
def tenure_group(tenure):
    if tenure <= 12:
        return '0-1_year'
    elif tenure <= 24:
        return '1-2_years'
    elif tenure <= 36:
        return '2-3_years'
    elif tenure <= 48:
        return '3-4_years'
    else:
        return '4+_years'

df_encoded['tenure_group'] = df_encoded['tenure'].apply(tenure_group)
print("✅ Created tenure_group feature")

# 4.2.2 Charge per Month Ratio
df_encoded['charge_per_month'] = df_encoded['TotalCharges'] / (df_encoded['tenure'] + 1)  # +1 to avoid division by zero
print("✅ Created charge_per_month feature")

# 4.2.3 Service Count
service_features = [col for col in df_encoded.columns if any(service in col for service in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'])]
df_encoded['total_services'] = df_encoded[service_features].sum(axis=1)
print("✅ Created total_services feature")

# 4.2.4 High Spender Flag
df_encoded['high_spender'] = (df_encoded['MonthlyCharges'] > df_encoded['MonthlyCharges'].quantile(0.75)).astype(int)
print("✅ Created high_spender feature")

# 4.2.5 New Customer Flag
df_encoded['new_customer'] = (df_encoded['tenure'] <= 3).astype(int)
print("✅ Created new_customer feature")

# 4.2.6 Contract Simplicity
contract_features = [col for col in df_encoded.columns if 'Contract' in col]
df_encoded['simple_contract'] = (~df_encoded[contract_features].any(axis=1)).astype(int)
print("✅ Created simple_contract feature")

# 4.3 Encode the new categorical features
print("\n" + "=" * 50)
print("4.3 FINAL ENCODING")
print("=" * 50)

# One-hot encode the new tenure_group feature
df_final_encoded = pd.get_dummies(df_encoded, columns=['tenure_group'], prefix=['tenure_group'], drop_first=True)

# 4.4 Drop unnecessary columns
columns_to_drop = ['customerID']  # ID column not useful for prediction
df_final_encoded = df_final_encoded.drop(columns=columns_to_drop)

print(f"Final dataset shape: {df_final_encoded.shape}")
print(f"Final number of features: {len(df_final_encoded.columns)}")

# 4.5 Display the new features
print("\n" + "=" * 50)
print("4.5 NEW FEATURES CREATED")
print("=" * 50)

new_features = ['charge_per_month', 'total_services', 'high_spender', 'new_customer', 'simple_contract']
print("New numerical features:")
for feature in new_features:
    if feature in df_final_encoded.columns:
        print(f"- {feature}: min={df_final_encoded[feature].min():.2f}, max={df_final_encoded[feature].max():.2f}, mean={df_final_encoded[feature].mean():.2f}")

# Show correlation of new features with Churn
print("\nCorrelation of new features with Churn:")
for feature in new_features:
    if feature in df_final_encoded.columns:
        correlation = df_final_encoded[feature].corr(df_final_encoded['Churn'])
        print(f"- {feature}: {correlation:.3f}")

# 4.6 Feature Importance Preview
print("\n" + "=" * 50)
print("4.6 FEATURE IMPORTANCE PREVIEW")
print("=" * 50)

# Calculate correlation with target for all features
correlations = df_final_encoded.corr()['Churn'].abs().sort_values(ascending=False)

print("Top 15 features most correlated with Churn:")
for feature, corr in correlations.head(15).items():
    print(f"- {feature}: {corr:.3f}")

# 4.7 Save the final engineered dataset
df_final_encoded.to_csv('telco_churn_engineered.csv', index=False)
print("\n✅ Final engineered dataset saved as 'telco_churn_engineered.csv'")

# Display final dataset info
print("\n" + "=" * 50)
print("FINAL DATASET OVERVIEW")
print("=" * 50)
print(f"Shape: {df_final_encoded.shape}")
print(f"Target variable distribution:")
print(df_final_encoded['Churn'].value_counts())
print(f"\nData types:")
print(df_final_encoded.dtypes.value_counts())

# STEP 5: MODEL BUILDING
print("=" * 60)
print("STEP 5: MODEL BUILDING")
print("=" * 60)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# 5.1 Prepare the Data for Modeling
print("5.1 PREPARING DATA FOR MODELING")
print("=" * 50)

# Separate features (X) and target (y)
X = df_final_encoded.drop('Churn', axis=1)
y = df_final_encoded['Churn']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Churn distribution: {y.value_counts()}")
print(f"Churn rate: {(y.sum() / len(y)) * 100:.2f}%")

# 5.2 Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Important for imbalanced data
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")
print(f"Training churn rate: {(y_train.sum() / len(y_train)) * 100:.2f}%")
print(f"Testing churn rate: {(y_test.sum() / len(y_test)) * 100:.2f}%")

# 5.3 Feature Scaling
print("\n5.2 FEATURE SCALING")
print("=" * 50)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled using StandardScaler")

# 5.4 Initialize Multiple Models
print("\n5.3 INITIALIZING MODELS")
print("=" * 50)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42, class_weight='balanced'),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# 5.5 Train and Evaluate Models
print("\n5.4 TRAINING AND EVALUATING MODELS")
print("=" * 50)

results = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")

    # Train the model
    if name in ['Support Vector Machine', 'K-Nearest Neighbors', 'Logistic Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

    print(f"✅ {name} trained")
    print(f"   Accuracy: {accuracy:.4f}")

# 5.6 Compare Model Performance
print("\n" + "=" * 50)
print("5.5 MODEL PERFORMANCE COMPARISON")
print("=" * 50)

# Create comparison table
performance_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[name]['accuracy'] for name in results.keys()]
}).sort_values('Accuracy', ascending=False)

print(performance_df)

# 5.7 Detailed Evaluation of Best Model
print("\n" + "=" * 50)
print("5.6 DETAILED EVALUATION OF BEST MODEL")
print("=" * 50)

# Get the best model based on accuracy
best_model_name = performance_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"🏆 BEST MODEL: {best_model_name}")
print(f"📊 Accuracy: {performance_df.iloc[0]['Accuracy']:.4f}")

# Detailed classification report
print("\n📋 DETAILED CLASSIFICATION REPORT:")
print(classification_report(y_test, best_predictions))

# Confusion Matrix
print("🎯 CONFUSION MATRIX:")
cm = confusion_matrix(y_test, best_predictions)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 5.8 Feature Importance (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\n" + "=" * 50)
    print("5.7 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))

    # Plot feature importance
    plt.subplot(1, 2, 2)
    top_features = feature_importance.head(10)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'Top 10 Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.show()

else:
    plt.tight_layout()
    plt.show()

# 5.9 Cross-Validation for More Robust Evaluation
print("\n" + "=" * 50)
print("5.8 CROSS-VALIDATION RESULTS")
print("=" * 50)

cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Scores (5-fold): {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 5.10 Save the Best Model
print("\n" + "=" * 50)
print("5.9 SAVING THE BEST MODEL")
print("=" * 50)

import joblib

# Save the best model and scaler
joblib.dump(best_model, 'best_churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Best model saved as 'best_churn_model.pkl'")
print("✅ Scaler saved as 'scaler.pkl'")

# 5.11 Final Summary
print("\n" + "=" * 50)
print("5.10 FINAL SUMMARY")
print("=" * 50)

print(f"🎯 BUSINESS IMPACT ANALYSIS:")
print(f"• We can predict churn with {performance_df.iloc[0]['Accuracy']:.1%} accuracy")
print(
    f"• This means we can identify {int(len(y_test) * performance_df.iloc[0]['Accuracy'])} out of {len(y_test)} customers correctly")
print(f"• We can proactively target at-risk customers before they leave")

print(f"\n📈 NEXT STEPS:")
print(f"• STEP 6: Model Interpretation & Business Insights")
print(f"• Deploy model to production")
print(f"• Create retention campaigns based on predictions")

# STEP 6: MODEL INTERPRETATION & BUSINESS INSIGHTS
print("=" * 60)
print("STEP 6: MODEL INTERPRETATION & BUSINESS INSIGHTS")
print("=" * 60)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 6.1 Load the best model and analyze feature importance
print("6.1 FEATURE IMPORTANCE ANALYSIS")
print("=" * 50)

# Load the best model
best_model = joblib.load('best_churn_model.pkl')

# Get feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 15 Most Important Features for Churn Prediction:")
    for i, row in feature_importance.head(15).iterrows():
        print(f"{i + 1:2d}. {row['feature']:30} : {row['importance']:.4f}")

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_10 = feature_importance.head(10)
    sns.barplot(x='importance', y='feature', data=top_10, palette='viridis')
    plt.title('Top 10 Features Driving Customer Churn', fontsize=16, fontweight='bold')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()

# 6.2 Analyze Customer Segments
print("\n6.2 CUSTOMER SEGMENT ANALYSIS")
print("=" * 50)

# Create customer segments based on predictions
df_test = X_test.copy()
df_test['Churn_Actual'] = y_test.values
df_test['Churn_Predicted'] = best_predictions
df_test['Churn_Probability'] = best_model.predict_proba(X_test)[:, 1]

# Segment customers by risk level
df_test['Risk_Level'] = pd.cut(df_test['Churn_Probability'],
                               bins=[0, 0.3, 0.7, 1.0],
                               labels=['Low Risk', 'Medium Risk', 'High Risk'])

print("Customer Risk Segmentation:")
print(df_test['Risk_Level'].value_counts())
print("\nRisk Level Distribution:")
print(df_test['Risk_Level'].value_counts(normalize=True) * 100)

# 6.3 High-Risk Customer Profile
print("\n6.3 HIGH-RISK CUSTOMER PROFILE")
print("=" * 50)

high_risk_customers = df_test[df_test['Risk_Level'] == 'High Risk']

if not high_risk_customers.empty:
    print("📊 Profile of High-Risk Customers (Top 10%):")

    # Analyze key characteristics
    high_risk_profile = {
        'Avg_Tenure': high_risk_customers['tenure'].mean(),
        'Avg_MonthlyCharges': high_risk_customers['MonthlyCharges'].mean(),
        'Avg_TotalCharges': high_risk_customers['TotalCharges'].mean(),
        'Count': len(high_risk_customers)
    }

    print(f"• Average Tenure: {high_risk_profile['Avg_Tenure']:.1f} months")
    print(f"• Average Monthly Charges: ${high_risk_profile['Avg_MonthlyCharges']:.2f}")
    print(f"• Average Total Charges: ${high_risk_profile['Avg_TotalCharges']:.2f}")
    print(f"• Number of High-Risk Customers: {high_risk_profile['Count']}")

# 6.4 Business Impact Calculation
print("\n6.4 BUSINESS IMPACT ANALYSIS")
print("=" * 50)

# Calculate potential revenue impact
avg_monthly_revenue = df_test['MonthlyCharges'].mean()
high_risk_count = len(high_risk_customers)

# Conservative estimate: 50% of high-risk customers will actually churn
predicted_churners = high_risk_count * 0.5
monthly_revenue_at_risk = predicted_churners * avg_monthly_revenue
annual_revenue_at_risk = monthly_revenue_at_risk * 12

print("💰 REVENUE AT RISK ANALYSIS:")
print(f"• Average Monthly Revenue per Customer: ${avg_monthly_revenue:.2f}")
print(f"• High-Risk Customers Identified: {high_risk_count}")
print(f"• Predicted Actual Churners: {predicted_churners:.0f}")
print(f"• Monthly Revenue at Risk: ${monthly_revenue_at_risk:,.2f}")
print(f"• Annual Revenue at Risk: ${annual_revenue_at_risk:,.2f}")

# 6.5 Retention Strategy Recommendations
print("\n6.5 RETENTION STRATEGY RECOMMENDATIONS")
print("=" * 50)

print("🎯 PRIORITY ACTIONS BASED ON MODEL INSIGHTS:")

# Analyze contract types in high-risk segment
contract_columns = [col for col in high_risk_customers.columns if 'Contract' in col]
if contract_columns:
    contract_analysis = high_risk_customers[contract_columns].sum().sort_values(ascending=False)
    print(f"\n1. CONTRACT STRATEGY:")
    print(f"   • Focus on converting {contract_analysis.index[0]} contracts to longer terms")
    print(f"   • {contract_analysis.iloc[0]:.0f} high-risk customers have this contract type")

# Analyze payment methods
payment_columns = [col for col in high_risk_customers.columns if 'PaymentMethod' in col]
if payment_columns:
    payment_analysis = high_risk_customers[payment_columns].sum().sort_values(ascending=False)
    print(f"\n2. PAYMENT STRATEGY:")
    print(f"   • Target customers using {payment_analysis.index[0]} payment method")
    print(f"   • Offer incentives for automatic payment methods")

# Tenure-based strategy
print(f"\n3. TENURE-BASED STRATEGY:")
print(f"   • High-risk customers average {high_risk_profile['Avg_Tenure']:.1f} months tenure")
print(f"   • Implement special retention offers for customers under 12 months")

# 6.6 Cost-Benefit Analysis
print("\n6.6 COST-BENEFIT ANALYSIS")
print("=" * 50)

# Assume retention campaign costs
campaign_cost_per_customer = 50  # $50 per customer for retention offers
total_campaign_cost = high_risk_count * campaign_cost_per_customer

# Assume 30% success rate in retention
retention_success_rate = 0.3
customers_saved = high_risk_count * retention_success_rate
annual_revenue_saved = customers_saved * avg_monthly_revenue * 12

roi = (annual_revenue_saved - total_campaign_cost) / total_campaign_cost * 100

print("📈 RETENTION CAMPAIGN ROI ANALYSIS:")
print(f"• Campaign Cost: ${total_campaign_cost:,.2f}")
print(f"• Customers Targeted: {high_risk_count}")
print(f"• Expected Customers Saved: {customers_saved:.0f}")
print(f"• Annual Revenue Saved: ${annual_revenue_saved:,.2f}")
print(f"• ROI: {roi:.1f}%")

# 6.7 Actionable Dashboard Metrics
print("\n6.7 EXECUTIVE DASHBOARD METRICS")
print("=" * 50)

print("📊 KEY PERFORMANCE INDICATORS:")
print(f"🎯 Model Accuracy: {performance_df.iloc[0]['Accuracy']:.1%}")
print(f"🚨 High-Risk Customers Identified: {high_risk_count}")
print(f"💰 Annual Revenue at Risk: ${annual_revenue_at_risk:,.0f}")
print(f"💡 Potential Revenue Saved: ${annual_revenue_saved:,.0f}")
print(f"📈 Campaign ROI: {roi:.1f}%")
print(f"🎪 Success Rate Needed: {retention_success_rate:.0%}")

# 6.8 Save Final Business Report
print("\n6.8 SAVING BUSINESS INSIGHTS REPORT")
print("=" * 50)

# Create a comprehensive business insights dataframe
business_insights = pd.DataFrame({
    'Metric': [
        'Model Accuracy',
        'High Risk Customers Identified',
        'Annual Revenue at Risk',
        'Retention Campaign Cost',
        'Potential Annual Revenue Saved',
        'Expected ROI'
    ],
    'Value': [
        f"{performance_df.iloc[0]['Accuracy']:.1%}",
        f"{high_risk_count}",
        f"${annual_revenue_at_risk:,.0f}",
        f"${total_campaign_cost:,.0f}",
        f"${annual_revenue_saved:,.0f}",
        f"{roi:.1f}%"
    ]
})

business_insights.to_csv('churn_business_insights.csv', index=False)
print("✅ Business insights report saved as 'churn_business_insights.csv'")

print("\n" + "=" * 60)
print("🎉 PROJECT COMPLETION SUMMARY")
print("=" * 60)
print("✅ Data Acquired & Cleaned")
print("✅ Exploratory Data Analysis Completed")
print("✅ Feature Engineering Implemented")
print("✅ Machine Learning Model Built & Evaluated")
print("✅ Business Insights Generated")
print("✅ Actionable Recommendations Provided")
print("\n🚀 NEXT: Deploy model & implement retention strategies!")