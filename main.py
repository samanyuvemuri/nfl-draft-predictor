import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('combine_data.csv')

def height_to_inches(height_str):
    try:
        feet, inches = height_str.split('-')
        return float(feet) * 12 + float(inches)
    except:
        return None
    
df['Height'] = df['Height'].apply(height_to_inches)

pos_corrections = {
    'OL': 'OL', 'OT': 'OL', 'OG': 'OL', 'C': 'OL',
    'EDGE': 'EDGE', 'DE': 'EDGE', 'OLB': 'EDGE',
    'DL': 'IDL', 'DT': 'IDL',
    'LB': 'LB', 'ILB': 'LB',
    'DB': 'DB', 'CB': 'DB', 'S': 'DB',
    'QB': 'QB', 'RB': 'RB', 'FB': 'FB', 'WR': 'WR', 'TE': 'TE',
    'K': 'K', 'P': 'P', 'LS': 'LS'
}
df['Pos'] = df['Pos'].map(pos_corrections)
df['Drafted'] = df['Drafted'].astype(int)


train_df = df[df['Year'] <= 2020]
val_df = df[(df['Year'] >= 2021) & (df['Year'] <= 2023)]
test_df = df[df['Year'] >= 2024]


numeric_features = ['Height', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle']
categorical_features = ['Pos', 'School']

X_train = train_df[numeric_features + categorical_features]
y_train_stage1 = train_df['Drafted']
y_train_stage2 = train_df['Pick']

X_val = val_df[numeric_features + categorical_features]
y_val_stage1 = val_df['Drafted']
y_val_stage2 = val_df['Pick']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

clf_stage1 = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
])

clf_stage1.fit(X_train, y_train_stage1)

y_val_pred_stage1 = clf_stage1.predict(X_val)
print("Stage 1 - Drafted vs Undrafted")
print(confusion_matrix(y_val_stage1, y_val_pred_stage1))
print(classification_report(y_val_stage1, y_val_pred_stage1))


X_train_stage2 = X_train[y_train_stage1==1]
y_train_stage2 = y_train_stage2[y_train_stage1==1]

X_val_stage2 = X_val[y_val_stage1==1]
y_val_stage2 = y_val_stage2[y_val_stage1==1]

reg_stage2 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
])

reg_stage2.fit(X_train_stage2, y_train_stage2)

y_val_pred_stage2 = reg_stage2.predict(X_val_stage2)
mae = mean_absolute_error(y_val_stage2, y_val_pred_stage2)
r2 = r2_score(y_val_stage2, y_val_pred_stage2)
print("\nStage 2 - Pick Number Regression")
print(f"MAE: {mae:.2f}")
print(f"R2: {r2:.2f}")

within_round = np.mean(np.abs(y_val_pred_stage2 - y_val_stage2) <= 32)
print(f"Within ±32 picks: {within_round*100:.2f}%")