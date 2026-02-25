# Load cleaned dataset
import pandas as pd
df = pd.read_csv("Data/cleaned_pollution_dataset.csv")
print(df.shape)
print(df["unhealthy"].value_counts())

#Define X and Y
x = df[["PM10", "CO", "NO2", "SO2", "Humidity", "Temperature"]]
y = df["unhealthy"]

#Train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)

#Apply SMOTE on training data to handle class imbalance
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_resampled.value_counts())

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train_resampled, y_train_resampled)
#Evaluate logistic regression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
y_pred = log_model.predict(x_test)
y_prob = log_model.predict_proba(x_test)[:,1]
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_prob))

#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train_resampled, y_train_resampled)
#Evaluate RandomForestClassifier
rf_pred = rf_model.predict(x_test)
rf_proba = rf_model.predict_proba(x_test)[:,1]
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
print(roc_auc_score(y_test, rf_proba))

#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(x_train_resampled, y_train_resampled)
#Evaluate Gradient Boosting Classifier
gb_pred = gb_model.predict(x_test)
gb_prob = gb_model.predict_proba(x_test)[:, 1]
print(confusion_matrix(y_test, gb_pred))
print(classification_report(y_test, gb_pred))
print(roc_auc_score(y_test, gb_prob))


# Model Selection:
# Logistic Regression achieved highest AUC (0.991),
# but Random Forest demonstrates better balance between
# precision and recall and captured nonlinear interactions.
# Therefore, Random Forest was selected for tuning.

# Tuning Random Forest Classifier
from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier(random_state=42,class_weight="balanced")#imbalanced 
# Define parameter grid
param_grid = {"n_estimators": [100, 200],
              "max_depth": [None, 10, 20],
              "min_samples_split": [2, 5],
              "min_samples_leaf": [1, 2],
              "max_features": ["sqrt", "log2"]}
#Run GridSearchCV
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=param_grid,
                       scoring="roc_auc",
                       cv=5,
                       n_jobs=-1,
                       verbose=1)
grid_rf.fit(x_train_resampled, y_train_resampled)
#Get the best model
best_rf = grid_rf.best_estimator_
print("Best parameters:", grid_rf.best_params_)
#Evaluate tuned model
rft_pred = best_rf.predict(x_test)
rft_proba = best_rf.predict_proba(x_test)[:, 1]
print(confusion_matrix(y_test, rft_pred))
print(classification_report(y_test, rft_pred))
print("ROC AUC:", roc_auc_score(y_test, rft_proba))

#FEATURE IMPORTANCE
import matplotlib.pyplot as plt
#Extract feature importance scores from tuned random forest
importances = best_rf.feature_importances_
features = x_train_resampled.columns
#Create a dataframe for better visualization and sort features by importance
feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances}).sort_values(by="Importance", ascending=False)

#Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feat_df["Feature"], feat_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance (Tuned Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()

# Interpretation:
# PM10 and CO emerged as the strongest predictors of
# unhealthy PM2.5 levels, suggesting strong pollutant interrelationships.
# This supports the hypothesis that other pollutants
# can be used to predict PM2.5 exceedance events.

#Save tuned model
import joblib
joblib.dump(best_rf, "Model/Tuned_random_forest.pkl")
print("Model saved successfully.")