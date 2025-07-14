#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler



# Data Preparation
df = pd.read_csv("../data/cleaned_reddit_posts.csv")

# Drop unwanted columns
drop_cols = ['score', 'upvote_ratio', 'sort_type', 'id', 'author', 'selftext']
df = df.drop(columns=drop_cols)




# Apply TF-IDF on title

tfidf = TfidfVectorizer(max_features=1000)
X_title_tfidf = tfidf.fit_transform(df['title']).toarray()




# Encode categorical features

label_enc_cols = ['subreddit', 'flair', 'media_type']
for col in label_enc_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))




# Encode target column
target_le = LabelEncoder()
df['popularity_bucket'] = target_le.fit_transform(df['popularity_bucket'])




# Convert TF-IDF to DataFrame
tfidf_df = pd.DataFrame(
    X_title_tfidf,
    columns=[f'tfidf_{i}' for i in range(X_title_tfidf.shape[1])],
    index=df.index
)




# Drop target and raw title (TF-IDF done separately)
structured_df = df.drop(columns=['popularity_bucket', 'title'])




# Split into features and labels
X = pd.concat([tfidf_df, structured_df], axis=1)
y = df['popularity_bucket']




# Split data into training + testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,                # input and output
    test_size=0.2,       # 20% for testing, 80% for training
    random_state=42,
    stratify=y           # ensures same label distribution in train & test
)




# Baseline: Naive majority class 

# 1. Find the most frequent class in y_train
majority_class = Counter(y_train).most_common(1)[0][0]
print("Majority class in training data:", majority_class)

# 2. Predict the majority class for all X_test samples
y_pred_baseline = np.full_like(y_test, fill_value=majority_class)

# 3. Evaluate performance
print("\nBaseline Performance (Majority Class):")
print("Accuracy:", round(accuracy_score(y_test, y_pred_baseline), 4))
print("F1 Score (macro):", round(f1_score(y_test, y_pred_baseline, average='macro'), 4))




# Train an XGBOOST Model

# Initialize the model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', tree_method="hist", device="cuda", random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)





# Evaluate on the test set

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate
print("Baseline XGBoost Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))




# Simple CV

# Step 1: Split into train (80%) and test (20%) - already done

# Now: Split train into train (80%) and val (20%)
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train  
)

# Step 2: Train on training set
xgb_model_simple_cv = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss')
xgb_model_simple_cv.fit(X_train_final, y_train_final)




# Step 3: Evaluate on validation set
y_val_pred = xgb_model_simple_cv.predict(X_val)


print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation F1 Score (macro):", f1_score(y_val, y_val_pred, average='macro'))




# 5-Fold CV

# Set up Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics for each fold
accuracies = []
f1_scores = []

# Loop through each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):

    # Split data into train and validation based on fold
    X_tr, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Initialize model
    model = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, tree_method="hist", device="cuda", random_state=42)

    # Train on training fold
    model.fit(X_tr, y_tr)

    # Predict on validation fold
    y_pred_fold = model.predict(X_val_fold)

    # Evaluate metrics
    acc = accuracy_score(y_val_fold, y_pred_fold)
    f1 = f1_score(y_val_fold, y_pred_fold, average='macro')

    accuracies.append(acc)
    f1_scores.append(f1)

    print(f"Fold {fold} - Accuracy: {acc:.4f} | F1 Score (macro): {f1:.4f}")

# After all folds
print("\nK-Fold CV Results (5 folds):")
print(f"Avg Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Avg F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")




# Hyperparameter tuning using optuna

# Objective function that Optuna will optimize
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
        'random_state': 42,
        #'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'tree_method':'hist', 
        'device':'cuda'
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average='macro')
    return f1                                           # Optuna will try to maximize this

# Create the study and run it
study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=MedianPruner())
study.optimize(objective, n_trials=30)

# Print best hyperparameters
print("Best trial:")
print(f"  F1 Score: {study.best_value}")
print("  Best hyperparameters:")
for key, val in study.best_params.items():
    print(f"    {key}: {val}")




# Retrain the model using the best parameters
best_params = study.best_trial.params
best_params.update({
    "objective": "multi:softprob",
    "num_class": len(target_le.classes_),
    "use_label_encoder": False,
    "eval_metric": "mlogloss"
})

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

y_test_pred_best = final_model.predict(X_test)


# Final test performance
print("Final Evaluation After Tuning:")
print("Accuracy:", accuracy_score(y_test, y_test_pred_best))
print("F1 Score (macro):", f1_score(y_test, y_test_pred_best, average='macro'))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_best))




# Confusion matrix plot
cm = confusion_matrix(y_test, y_test_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_le.classes_, yticklabels=target_le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()




# Feature importance plot
importances = final_model.feature_importances_
top_idx = importances.argsort()[-20:][::-1]   # Top 20 important features
top_features = X.columns[top_idx]

sns.barplot(x=importances[top_idx], y=top_features)
plt.title("Top 20 Feature Importances")
plt.show()

