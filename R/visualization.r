
# Load libraries
library(tidyverse)
library(corrplot)
library(GGally)
library(readr)

# 1. Load Cleaned Dataset
data <- read_csv("Data/cleaned_pollution_dataset.csv")

# Inspect structure
glimpse(data)

# Define target variable
target_name <- "unhealthy"

# 2. Target Distribution
p_target <- ggplot(data, aes(x = factor(.data[[target_name]]))) +
  geom_bar(fill = "steelblue") + theme_minimal() + labs(
    title = "Air Quality Target Distribution",
    x = "Unhealthy (0 = No, 1 = Yes)",
    y = "Count")
ggsave("target_distribution.png", plot = p_target)

# 3. Numeric Feature Distributions
numeric_data <- data %>% select(where(is.numeric))

numeric_data %>%
  pivot_longer( cols = everything(),
                names_to = "Feature",
                values_to = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30, fill = "darkgreen", alpha = 0.7) +
  facet_wrap(~Feature, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of Numeric Features")
ggsave("feature_histograms.png", width = 12, height = 9)

# 4. Correlation Heatmap
cor_matrix <- cor(numeric_data, use = "complete.obs")

png("correlation_heatmap.png", width = 1000, height = 800)
corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.cex = 0.7,
        col = colorRampPalette(c("blue", "white", "red"))(200))
dev.off()

# 5. Boxplots (Feature vs Target)
for (feature in setdiff(names(numeric_data), target_name)) {

  p <- ggplot(
    data,
    aes(
      x = factor(.data[[target_name]]),
      y = .data[[feature]])) +
    geom_boxplot(fill = "orange") + theme_minimal() +
    labs(title = paste(feature, "vs Unhealthy"),
         x = "Unhealthy (0 = No, 1 = Yes)",
        y = feature)

  ggsave(paste0(feature, "_vs_unhealthy_boxplot.png"), plot = p)}

# 6. Pair Plot (Small Feature Set)
small_sample <- numeric_data %>% select(1:4)

png("pairplot.png", width = 1000, height = 800)
ggpairs(small_sample)
dev.off()

# 7. Feature Variance Plot
variance_df <- numeric_data %>%
  summarise(across(everything(), var)) %>%
  pivot_longer(cols = everything(),
               names_to = "Feature",
               values_to = "Variance")

p_variance <- ggplot(variance_df,
                    aes(x = reorder(Feature, Variance), y = Variance)) +
                    geom_col(fill = "purple") +
                    coord_flip() + theme_minimal() +
  labs(title = "Feature Variance", x = "Feature", y = "Variance")
ggsave("feature_variance.png", plot = p_variance)