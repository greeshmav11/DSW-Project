# .py files of notebooks to run on cluster


#!/usr/bin/env python
# coding: utf-8

# !pip install pandas
#  !pip install tensorflow
# !pip install scikeras


import pandas as pd             
import numpy as np
import shap
import contextlib
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, GlobalAveragePooling1D, Concatenate, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




# Step 1: Load the dataset
df = pd.read_csv("../data/cleaned_reddit_posts.csv")




# Show how many entries fall into each popularity bucket to understand class balance
print(df["popularity_bucket"].value_counts())




# Step 2: Drop unneeded columns
df = df.drop(columns=["id", "author", "score", "num_comments", "upvote_ratio"])




# Step 3: Encode labels (popularity_bucket)
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["popularity_bucket"])
y = to_categorical(df["label"])




# Step 4: Encode categorical features
cat_features = ["subreddit", "flair", "media_type"]
encoded_features = []

for col in cat_features:
    le = LabelEncoder()
    df[col] = df[col].fillna("unknown")
    encoded = le.fit_transform(df[col])
    encoded_features.append(encoded)

# Add binary features
encoded_features.append(df["is_self"].astype(int))
encoded_features.append(df["nsfw"].astype(int))
encoded_features.append(df["created_hour"].fillna(0).astype(int))

# Final non-text input
X = np.stack(encoded_features, axis=1) 




# Step 7: Build the model
def create_model(dropout_rate=0.3):
    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))  
    model.add(Dense(256, activation='relu'))  #128    
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))  #64   
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(3, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model




# Step 8: Compile and train
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Type of X_train:", type(X_train))
print("X_train shape:", X_train.shape)
print("X_train[0] shape:", np.array(X_train[0]).shape)




# 1. Update your model function to accept parameters from Optuna
def create_model_optuna(dropout_rate, learning_rate):
    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 2. Define the Optuna objective

def objective(trial):
    # Hyperparameter suggestions
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    epochs = trial.suggest_int("epochs", 10, 25)

    # K-Fold CV
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    val_accuracies = []

    for train_index, val_index in skf.split(X_train, np.argmax(y_train, axis=1)):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        model = create_model_optuna(dropout_rate, learning_rate)
        model.fit(X_tr, y_tr,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0)

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        val_accuracies.append(val_acc)

    return np.mean(val_accuracies)


# 3. Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# 4. Train final model using best parameters
best_params = study.best_params
print("\n Best Hyperparameters:")
print(best_params)





# Merge train + val (used inside Optuna's validation_split)
X_train_val = X_train  # because we used all of X_train for tuning
y_train_val = y_train

# Retrain final model on all training data (Optuna already used val split inside)
final_model = create_model_optuna(best_params["dropout_rate"], best_params["learning_rate"])
final_model.fit(X_train, y_train,
                batch_size=best_params["batch_size"],
                epochs=best_params["epochs"],
                verbose=1)


# Evaluate on test set
y_pred = final_model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test_labels, y_pred_labels)
prec = precision_score(y_test_labels, y_pred_labels, average="weighted", zero_division=0)
rec = recall_score(y_test_labels, y_pred_labels, average="weighted", zero_division=0)
f1 = f1_score(y_test_labels, y_pred_labels, average="weighted", zero_division=0)

print("\n Final Test Set Performance (after retraining):")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")




# Naive Baseline 

# Find the most frequent class in the training set
most_common_class = np.argmax(np.sum(y_train, axis=0))

# Predict that class for all test samples
naive_predictions = np.full(shape=(y_test.shape[0],), fill_value=most_common_class)

# Convert one-hot y_test to class labels
y_test_labels = np.argmax(y_test, axis=1)

# Compute metrics
naive_accuracy = accuracy_score(y_test_labels, naive_predictions)
naive_precision = precision_score(y_test_labels, naive_predictions, average='weighted', zero_division=0)
naive_recall = recall_score(y_test_labels, naive_predictions, average='weighted', zero_division=0)
naive_f1 = f1_score(y_test_labels, naive_predictions, average='weighted', zero_division=0)

print("\n=== Naive Baseline Metrics ===")
print(f"Most Frequent Class: {most_common_class} ({label_encoder.inverse_transform([most_common_class])[0]})")
print(f"Accuracy: {naive_accuracy:.4f}")
print(f"Precision: {naive_precision:.4f}")
print(f"Recall: {naive_recall:.4f}")
print(f"F1 Score: {naive_f1:.4f}")

# Print confusion matrix for naive baseline
print("\nNaive Baseline Confusion Matrix:")
print(confusion_matrix(y_test_labels, naive_predictions))



# Use SHAP DeepExplainer (suitable for Keras models)
explainer = shap.DeepExplainer(final_model, X_train[:100])  # Use a small background dataset

# Suppress verbose TensorFlow logs
with contextlib.redirect_stdout(io.StringIO()):
    shap_values = explainer.shap_values(X_test[:50])  # SHAP expects raw input (not one-hot encoded)

shap.summary_plot(shap_values, X_test[:50], feature_names=cat_features + ['is_self', 'nsfw', 'created_hour'], show=False)
plt.tight_layout()
plt.savefig("DLModel_shap_summary.png")
plt.close()

final_model.save("DLModel_CategoricalFeatures.h5")
