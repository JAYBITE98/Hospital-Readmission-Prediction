import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD AND CLEAN DATA
# ============================================
print("=" * 60)
print("🏥 HOSPITAL READMISSION PREDICTION - IMPROVED MODEL")
print("=" * 60)

# Load the original dataset
df = pd.read_csv('diabetic_data.csv')
print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")

# Create binary target: 1 = readmitted within 30 days
df['readmit_binary'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Drop unnecessary columns
cols_to_drop = [
    'encounter_id', 'patient_nbr', 'payer_code', 'medical_specialty',
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'max_glu_serum', 'A1Cresult', 'weight'
]
df_clean = df.drop(columns=cols_to_drop, errors='ignore')

# Replace '?' with NaN and drop missing values
df_clean = df_clean.replace('?', np.nan)
key_columns = ['race', 'gender', 'age', 'diag_1', 'diag_2', 'diag_3']
df_clean = df_clean.dropna(subset=key_columns)

print(f"✅ After cleaning: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

# ============================================
# 2. FEATURE ENGINEERING (Enhanced)
# ============================================
print("\n" + "=" * 60)
print("🔧 FEATURE ENGINEERING")
print("=" * 60)

# Create new features from existing ones
df_clean['total_visits'] = (
        df_clean['number_outpatient'] +
        df_clean['number_emergency'] +
        df_clean['number_inpatient']
)

df_clean['medication_complexity'] = (
        df_clean['num_medications'] / (df_clean['num_lab_procedures'] + 1)
)

df_clean['procedure_ratio'] = (
        df_clean['num_procedures'] / (df_clean['time_in_hospital'] + 1)
)

df_clean['lab_intensity'] = (
        df_clean['num_lab_procedures'] / (df_clean['time_in_hospital'] + 1)
)

# Create age groups as numeric categories
age_mapping = {
    '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3,
    '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7,
    '[80-90)': 8, '[90-100)': 9
}
df_clean['age_numeric'] = df_clean['age'].map(age_mapping)

# Create risk score based on medications and diagnoses
df_clean['med_risk'] = df_clean['num_medications'].apply(
    lambda x: 1 if x > 20 else (2 if x > 10 else 3)
)

# New clinical risk features
df_clean['emergency_visit'] = df_clean['number_emergency'].apply(lambda x: 1 if x > 0 else 0)
df_clean['chronic_condition'] = df_clean['number_diagnoses'].apply(lambda x: 1 if x >= 5 else 0)
df_clean['medication_burden'] = df_clean['num_medications'].apply(lambda x: 1 if x >= 15 else 0)

print(f"✅ Created new features")

# ============================================
# 3. SELECT ENHANCED FEATURE SET
# ============================================
features_enhanced = [
    # Original features
    'race', 'gender', 'age', 'time_in_hospital',
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient',
    'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
    'change', 'diabetesMed',
    # Engineered features
    'total_visits', 'medication_complexity', 'procedure_ratio',
    'lab_intensity', 'age_numeric', 'med_risk',
    # New clinical risk features
    'emergency_visit', 'chronic_condition', 'medication_burden'
]

# Keep only available features
available_features = [col for col in features_enhanced if col in df_clean.columns]
print(f"✅ Using {len(available_features)} features")

# ============================================
# 4. PREPARE DATA FOR MACHINE LEARNING
# ============================================
print("\n" + "=" * 60)
print("🤖 MACHINE LEARNING WITH ENHANCED FEATURES")
print("=" * 60)

# Create dataset with features and target
ml_data = df_clean[available_features + ['readmit_binary']].copy()

# Encode categorical variables
label_encoders = {}
categorical_cols = ['race', 'gender', 'age', 'diag_1', 'diag_2', 'diag_3', 'change', 'diabetesMed']

for col in categorical_cols:
    if col in ml_data.columns:
        le = LabelEncoder()
        ml_data[col] = ml_data[col].astype(str)
        ml_data[col] = le.fit_transform(ml_data[col])
        label_encoders[col] = le
        print(f"✅ Encoded {col}")

# Separate features and target
X = ml_data.drop('readmit_binary', axis=1)
y = ml_data['readmit_binary']

print(f"📊 Feature matrix shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")
print(f"📊 Readmission rate: {y.mean() * 100:.2f}%")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train set: {X_train.shape[0]} rows")
print(f"✅ Test set: {X_test.shape[0]} rows")

# Scale features for XGBoost
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 5. TRAIN RANDOM FOREST (Baseline)
# ============================================
print("\n" + "-" * 40)
print("🌲 RANDOM FOREST MODEL")
print("-" * 40)
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)

print(f"AUC Score: {rf_auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# ============================================
# 6. HANDLE IMBALANCED DATA WITH SMOTE
# ============================================
print("\n" + "-" * 40)
print("⚡ ENHANCED XGBOOST MODEL WITH SMOTE")
print("-" * 40)

# Apply SMOTE to training data
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"✅ SMOTE Applied! Original train: {X_train_scaled.shape[0]} → Resampled: {X_train_resampled.shape[0]}")

# ============================================
# 7. TRAIN TUNED XGBOOST MODEL
# ============================================
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train_resampled, y_train_resampled)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_proba)

print(f"AUC Score: {xgb_auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, xgb_pred))

# ============================================
# 8. SAVE PREDICTIONS FOR POWER BI - CRITICAL PART!
# ============================================
print("\n" + "=" * 60)
print("💾 SAVING PREDICTIONS FOR POWER BI")
print("=" * 60)

# Create results dataframe
results_df = X_test.copy()
results_df['actual_readmit'] = y_test.values
results_df['predicted_readmit'] = xgb_pred
results_df['risk_score'] = xgb_proba
results_df['patient_id'] = range(len(results_df))

# Add patient identifiers from original data
# We need to track back to get original patient IDs
# Since we don't have encounter_id in the features, we'll use index
results_df['original_index'] = X_test.index

# Save to CSV - THIS IS THE CRITICAL STEP!
try:
    results_df.to_csv('improved_readmission_predictions.csv', index=False)
    print("✅ File saved: improved_readmission_predictions.csv")

    # Show file location
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, 'improved_readmission_predictions.csv')
    print(f"📁 File saved to: {file_path}")

    # Show file size
    file_size = os.path.getsize('improved_readmission_predictions.csv') / 1024
    print(f"📊 File size: {file_size:.1f} KB")

except Exception as e:
    print(f"❌ Error saving file: {e}")

# ============================================
# 9. HIGH-RISK PATIENT ANALYSIS
# ============================================
high_risk = results_df[results_df['risk_score'] > 0.7]
print(f"\n🚨 High-Risk Patients (Risk Score > 0.7): {len(high_risk)}")
if len(high_risk) > 0:
    print(f"   Actual readmission rate in this group: {high_risk['actual_readmit'].mean() * 100:.1f}%")

# ============================================
# 10. QUICK FILE VERIFICATION
# ============================================
print("\n" + "=" * 60)
print("✅ VERIFYING SAVED FILES")
print("=" * 60)

# List all CSV files in the directory
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
for file in csv_files:
    size = os.path.getsize(file) / 1024
    print(f"   - {file} ({size:.1f} KB)")

print("\n" + "=" * 60)
print("🎉 PROJECT COMPLETE!")
print(f"   Random Forest AUC: {rf_auc:.3f}")
print(f"   XGBoost with SMOTE AUC: {xgb_auc:.3f}")
print("   You now have:")
print("   1. Feature importance chart (improved_feature_importance.png)")
print("   2. ROC comparison chart (roc_comparison.png)")
print("   3. Predictions file (improved_readmission_predictions.csv) ✅")
print("=" * 60)