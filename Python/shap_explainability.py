#Import libraries
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split

#Setup logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

#Load processed data
df = pd.read_csv("Data/cleaned_pollution_dataset.csv")

#Define features and target
x = df[["PM10", "CO", "NO2", "SO2", "Humidity", "Temperature"]]
y = df["unhealthy"]
logging.info("Features and target seperated")

#Train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y)

#Load trained model
MODEL_PATH = "Model/Tuned_random_forest.pkl"
model = joblib.load(MODEL_PATH)
logging.info("Tuned Random Forest model loaded successfully.")

#Intialize SHAP explainer
explainer = shap.TreeExplainer(model)
logging.info("SHAP TreeExplainer initialized.")

#Compute SHAP values
shap_values = explainer.shap_values(X_test)

# Simple fix
if isinstance(shap_values, list):
    shap_values_positive = shap_values[1]
else:
    shap_values_positive = shap_values
logging.info("SHAP values computed.")

#Global interpretation
#Summary_plot
#Where to save shap_plots
Save_path = "SHAP Plots"
plt.figure()
shap.summary_plot(shap_values_positive,
                  X_test, show=False)
plt.tight_layout()
plt.savefig(Save_path + "shap_summary_plot.png", dpi=300, bbox_inches="tight")
plt.close()

#Bar importance plot
plt.figure()
shap.summary_plot(shap_values_positive,
                  X_test,
                  plot_type="bar",show=False)
plt.tight_layout()
plt.savefig(Save_path + "shap_bar_plot.png", dpi=300, bbox_inches="tight")
plt.close()

#Feature interaction insight
#Dependence plot
# Ensure shap_values_positive is 2D
if len(shap_values_positive.shape) == 3:
    shap_values_positive = shap_values_positive[:, :, 1]

# Ensure X_test is a DataFrame with correct columns
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test, columns=feature_names)

# Align shapes strictly
assert shap_values_positive.shape[0] == X_test.shape[0], "Row mismatch!"
assert shap_values_positive.shape[1] == X_test.shape[1], "Column mismatch!"

# Get most important feature
mean_abs_shap = np.abs(shap_values_positive).mean(axis=0)
top_feature_index = np.argmax(mean_abs_shap)
top_feature_name = X_test.columns[top_feature_index]

# Plot dependence
shap.dependence_plot(
    top_feature_name,
    shap_values_positive,
    X_test,
    interaction_index=None,
    show=False
)

plt.tight_layout()
plt.savefig("shap_dependence_plot.png")
plt.close()

print("SHAP dependence plot saved successfully.")

#Local explanation
#Forceplot
# Explain first test observation
sample_index = 0
logging.info("Generating local force plot")
shap.initjs()
force_plot = shap.force_plot(explainer.expected_value[1],
                             shap_values_positive[sample_index],
                             X_test.iloc[sample_index])

# If running in VSCode, save as HTML
shap.save_html(Save_path + "shap_force_plot.html", force_plot)

