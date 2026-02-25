library(tidyverse)
library(corrplot)
library(pROC)

#Loading dataset
df <- read.csv("Data/cleaned_pollution_dataset.csv")
dim(df)
head(df)
summary(df)

#DISTRIBUTION ANALYSIS
#Histogram of target variable
ggplot(df, aes(x = PM2.5)) + 
     geom_histogram(bins = 30, fill = "steelblue") + theme_minimal()
ggsave("PM25_distribution.png", width = 8, height = 6, dpi = 300)

#Density plot
ggplot(df, aes(x = PM2.5)) + geom_density(fill = "#0055ff2d", alpha = 0.5) + theme_minimal()
ggsave("PM 2.5_Densityplot.png", width = 8, height = 6, dpi = 300)

#CORRELATION ANALYSIS
# Select numeric columns
numeric_df <- df %>% select(where(is.numeric))
cor_matrix <- cor(numeric_df)
# Open bigger plotting window
windows(width = 16, height = 14)  # for Windows

# Plot
corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.cex = 0.8,      # smaller text
         tl.col = "black",  # black labels
         cl.cex = 1.2       # color legend bigger)


#HYPOTHESIS TESTING
t.test().test(`PM2.5` ~ unhealthy, data = df)

# LOGISTIC REGRESSION
df$unhealthy <- as.factor(df$unhealthy)
log_model <- glm(unhealthy ~ PM2.5 + PM10 + CO + NO2 + SO2 + Humidity + Temperature, data = df, family = "binomial")
summary(log_model)


library(logistf)
# Run the Firth Penalized Logistic Regression
# use logistf() instead of glm()
firth_model <- logistf(unhealthy ~ PM2.5 + PM10 + C O + NO2 + SO2 + Humidity + Temperature, data = df)

# 3. View the stable results
summary(firth_model)

#Odds ratio
exp(coef(firth_model))

#ROC_CURVE
probabilities <- predict(firth_model, type = "response")
roc_curve <- roc(df$unhealthy, probabilities)
plot(roc_curve, col = "blue")
auc(roc_curve)
