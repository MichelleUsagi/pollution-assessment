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
# For classification, SHAP returns values for each class.
# We focus on class 1 (unhealthy air).
shap_values = explainer.shap_values(X_test)
# For binary classification:
# shap_values[1] corresponds to the positive class.
shap_values_positive = shap_values[1]
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
# Choose most important feature automatically
mean_abs_shap = np.abs(shap_values_positive).mean(axis=0)
top_feature_index = np.argmax(mean_abs_shap)
top_feature_name = x.columns[top_feature_index]
logging.info(f"Generating dependence plot for: {top_feature_name}")
plt.figure()
shap.dependence_plot(top_feature_name,
                     shap_values_positive,
                     X_test,show=False)
plt.tight_layout()
plt.savefig(Save_path + "shap_dependence_plot.png", dpi=300, bbox_inches="tight")
plt.close()

#Local explanation
#Forceplot
# Explain first test observation
sample_index = 0
logging.info("Generating local force plot")
shap.initjs()
force_plot = shap.force_plot(explainer.expected_value[1],
                             shap_values_positive[sample_index],
                             x_test.iloc[sample_index])

# If running in VSCode, save as HTML
shap.save_html(Save_path + "shap_force_plot.html", force_plot)

