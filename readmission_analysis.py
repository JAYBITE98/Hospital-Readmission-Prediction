import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD DATA
# ============================================
print("="*60)
print("🏥 HOSPITAL READMISSION PREDICTION")
print("="*60)

df = pd.read_csv('diabetic_data.csv')
print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")

# ============================================
# 2. CLEAN THE TARGET VARIABLE
# ============================================
# Create a binary target: 1 = readmitted within 30 days, 0 = not readmitted
df['readmit_binary'] = df['readmitted'].apply(
    lambda x: 1 if x == '<30' else 0
)

print("\n🎯 Readmission Distribution (Binary):")
print(df['readmit_binary'].value_counts())
print(f"   Readmission Rate: {df['readmit_binary'].mean()*100:.2f}%")

# ============================================
# 3. REMOVE UNUSEFUL COLUMNS
# ============================================
# Columns to drop (patient identifiers and columns with too many missing values)
cols_to_drop = [
    'encounter_id', 'patient_nbr', 'payer_code', 'medical_specialty',
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'max_glu_serum', 'A1Cresult', 'weight'  # Too many missing values
]
df_clean = df.drop(columns=cols_to_drop, errors='ignore')
print(f"\n✅ Dropped {len(cols_to_drop)} columns. Remaining: {df_clean.shape[1]}")

# ============================================
# 4. HANDLE MISSING VALUES
# ============================================
# Replace '?' with NaN
df_clean = df_clean.replace('?', np.nan)

# Drop rows with missing values in key columns
key_columns = ['race', 'gender', 'age', 'diag_1', 'diag_2', 'diag_3']
df_clean = df_clean.dropna(subset=key_columns)
print(f"✅ After dropping missing values: {df_clean.shape[0]} rows")

# ============================================
# 5. CREATE FEATURE SET
# ============================================
# Select features for modeling
features = [
    'race', 'gender', 'age', 'time_in_hospital',
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient',
    'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
    'change', 'diabetesMed'
]

# Check if features exist
available_features = [col for col in features if col in df_clean.columns]
print(f"✅ Using {len(available_features)} features: {available_features}")

# ============================================
# 6. EXPLORE FEATURE DISTRIBUTIONS
# ============================================
print("\n📊 Feature Distributions:")
for feature in available_features[:5]:
    print(f"\n{feature}:")
    print(df_clean[feature].value_counts().head(3))

# ============================================
# 7. SAVE CLEANED DATA
# ============================================
df_clean.to_csv('cleaned_readmission_data.csv', index=False)
print("\n✅ Cleaned data saved as 'cleaned_readmission_data.csv'")
print(f"   Final dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

# ============================================
# 8. PREPARE DATA FOR MACHINE LEARNING
# ============================================
print("\n" + "="*60)
print("🤖 MACHINE LEARNING MODEL")
print("="*60)

# Create a copy with only our features and target
ml_data = df_clean[available_features + ['readmit_binary']].copy()

# Handle categorical variables with Label Encoding
label_encoders = {}
categorical_cols = ['race', 'gender', 'age', 'diag_1', 'diag_2', 'diag_3', 'change', 'diabetesMed']

for col in categorical_cols:
    if col in ml_data.columns:
        le = LabelEncoder()
        ml_data[col] = ml_data[col].astype(str)
        ml_data[col] = le.fit_transform(ml_data[col])
        label_encoders[col] = le
        print(f"✅ Encoded {col}")

# Separate features (X) and target (y)
X = ml_data.drop('readmit_binary', axis=1)
y = ml_data['readmit_binary']

print(f"\n📊 Feature matrix shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Train set: {X_train.shape[0]} rows")
print(f"✅ Test set: {X_test.shape[0]} rows")

# ============================================
# 9. TRAIN A RANDOM FOREST MODEL
# ============================================
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
print("\n🔄 Training Random Forest model...")
model.fit(X_train, y_train)

# ============================================
# 10. EVALUATE THE MODEL
# ============================================
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n📊 MODEL EVALUATION:")
print("-"*40)
print(classification_report(y_test, y_pred))

# Calculate AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(f"   True Negatives: {cm[0][0]}")
print(f"   False Positives: {cm[0][1]}")
print(f"   False Negatives: {cm[1][0]}")
print(f"   True Positives: {cm[1][1]}")

# ============================================
# 11. FEATURE IMPORTANCE
# ============================================
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 Top 10 Most Important Features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance.head(10)['feature'],
         feature_importance.head(10)['importance'])
plt.xlabel('Importance Score')
plt.title('Top 10 Features for Predicting Readmission')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

print("\n✅ Feature importance chart saved as 'feature_importance.png'")

# ============================================
# 12. SAVE PREDICTIONS FOR POWER BI
# ============================================
# Create a copy of the test data with predictions
results_df = X_test.copy()
results_df['actual_readmit'] = y_test
results_df['predicted_readmit'] = y_pred
results_df['risk_score'] = y_pred_proba

# Save to CSV
results_df.to_csv('readmission_predictions.csv', index=False)
print("\n✅ Predictions saved as 'readmission_predictions.csv'")

print("\n" + "="*60)
print("🎉 PROJECT COMPLETE!")
print("   You now have:")
print("   1. Cleaned dataset (cleaned_readmission_data.csv)")
print("   2. ML model with predictions (readmission_predictions.csv)")
print("   3. Feature importance chart (feature_importance.png)")
print("="*60)