import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
from time import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# 📌 Load dataset (merged from multiple sources)
df_scraped = pd.read_csv('./tweets/labeled_tweets.csv')
df_public = pd.read_csv('./tweets/public_data_labeled.csv')

# 🔹 Clean dataset
df_scraped.drop_duplicates(inplace=True)
df_scraped.drop('id', axis='columns', inplace=True)
df_public.drop_duplicates(inplace=True)

# 🔹 Merge datasets
df = pd.concat([df_scraped, df_public])
df['label'] = df.label.map({'Offensive': 1, 'Non-offensive': 0})

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['full_text'], df['label'], random_state=42)

# 🔹 Vectorization
count_vector = CountVectorizer(stop_words='english', lowercase=True)
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

# 🔹 Define models
models = [
    MultinomialNB()
]

# 📌 Train models and evaluate performance
def pipeline(learner_list, X_train, y_train, X_test, y_test):
    results = []
    
    for learner in learner_list:
        model_name = learner.__class__.__name__
        print(f"Training {model_name}...")

        start = time()
        learner.fit(X_train, y_train)
        train_time = time() - start

        start = time()
        predictions_test = learner.predict(X_test)
        prediction_time = time() - start

        result = {
            'Algorithm': model_name,
            'Accuracy: Test': accuracy_score(y_test, predictions_test),
            'Precision: Test': precision_score(y_test, predictions_test),
            'Recall: Test': recall_score(y_test, predictions_test),
            'F1 Score: Test': f1_score(y_test, predictions_test),
            'Training Time': train_time,
            'Prediction Time': prediction_time
        }

        results.append(result)
        print(f"✅ {model_name} trained successfully!\n")

    return pd.DataFrame(results)

# 🔹 Train and evaluate models
results = pipeline(models, training_data, y_train, testing_data, y_test)

# 📌 Hyperparameter tuning
def param_tuning(clf, param_dict, X_train, y_train, X_test, y_test):
    scorer = make_scorer(f1_score)
    grid_obj = GridSearchCV(estimator=clf, param_grid=param_dict, scoring=scorer, cv=5)
    grid_fit = grid_obj.fit(X_train, y_train)

    best_clf = grid_obj.best_estimator_
    best_predictions = best_clf.predict(X_test)

    print(f"\nOptimized Model - {clf.__class__.__name__}")
    print("Best Parameters:", grid_obj.best_params_)
    print("Accuracy:", accuracy_score(y_test, best_predictions))
    print("F1-score:", f1_score(y_test, best_predictions))
    print("Precision:", precision_score(y_test, best_predictions))
    print("Recall:", recall_score(y_test, best_predictions))

    return best_clf

# 🔹 Tune selected models
param_grid_sgd = {'alpha': [0.095, 0.0002, 0.0003], 'max_iter': [2500, 3000, 4000]}
clf_sgd = param_tuning(SGDClassifier(), param_grid_sgd, training_data, y_train, testing_data, y_test)

param_grid_lr = {'C': [1, 1.2, 1.3, 1.4]}
clf_lr = param_tuning(LogisticRegression(), param_grid_lr, training_data, y_train, testing_data, y_test)

param_grid_rf = {'n_estimators': [50, 150], 'min_samples_leaf': [1, 5], 'min_samples_split': [2, 5]}
clf_rf = param_tuning(RandomForestClassifier(), param_grid_rf, training_data, y_train, testing_data, y_test)

# 📌 Ensure 'models/' directory exists
if not os.path.exists("models"):
    os.makedirs("models")
    print("📁 Created 'models/' directory.")

# 📌 Save the best model
model_path = "models/cb_sgd_final.sav"
joblib.dump(clf_sgd, model_path)
print(f"✅ Model saved as {model_path}")

# 📌 Save the trained vectorizer
vectorizer_path = "models/count_vect.sav"
joblib.dump(count_vector, vectorizer_path)
print(f"✅ Vectorizer saved as {vectorizer_path}")
