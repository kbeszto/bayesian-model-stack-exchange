# Install necessary packages
install.packages("tidyverse")
install.packages("pROC")

# STAT 447C Bayesian Statistics Final Project
library(tidyverse)
library(pROC)
library(dplyr)
library(stringr)
library(lubridate)
library(rstan)
require(rstan)

# Read data
stack_exch_data <- read.csv("~/Downloads/stack-exchange-questions.csv")
head(stack_exch_data)
nrow(stack_exch_data)

# Ensuring correct data types
stack_exch_data$CreationDate <- ymd_hms(stack_exch_data$CreationDate)

# Feature 1: Title/Question Length
stack_exch_data <- stack_exch_data %>%
                    mutate(TitleLength = nchar(Title))

# Feature 2: Number of Tags
stack_exch_data <- stack_exch_data %>%
  mutate(TagCount = str_count(Tags, "<") )  # Note: Tags are in the format "<tag1><tag2><tag3>"

# Feature 3: Day of the Week
stack_exch_data <- stack_exch_data %>%
  mutate(DayOfWeek = wday(CreationDate, label = TRUE))

# Feature 4: Hour of the Day
stack_exch_data <- stack_exch_data %>%
  mutate(HourOfDay = hour(CreationDate))

# Define target variable: HighPopularity (binary classification)
# Score is the the number of upvotes on a question/answer minus the number of downvotes.
stack_exch_data <- stack_exch_data %>%
  mutate(HighPopularity = ifelse(Score >= 1, 1, 0))

# Checking distribution of target (contingency table)
# Note: threshold Score >= 1 or > 0 (0: 7860, 1: 2140)
# Note: threshold Score >= 0 (0: 2116, 1: 7884)
# May need to adjust threshold as it seems many posts do not get any votes and/or the votes cancel out to 0
# Likely sticking with 1 as measure of popularity, anything else either insignificant or negative
table(stack_exch_data$HighPopularity)
summary(stack_exch_data)

# Save cleaned data and save into new variable
write.csv(stack_exch_data, "stackexchange_cleaned.csv", row.names = FALSE)
stack_exch_data_cleaned <- read.csv("stackexchange_cleaned.csv")

# Note: Standardize predictors and convert DayOfWeek to numeric because Stan requires numeric input  
stack_exch_data_cleaned$DayOfWeek <- as.numeric(factor(stack_exch_data_cleaned$DayOfWeek))
stack_exch_data_cleaned <- stack_exch_data_cleaned %>%
                            mutate(TagCount = scale(TagCount),
                                    TitleLength = scale(TitleLength),
                                    DayOfWeek = scale(DayOfWeek),
                                    HourOfDay = scale(HourOfDay))
head(stack_exch_data_cleaned)

# Check for NAs
sum(is.na(stack_exch_data_cleaned))

X <- as.matrix(stack_exch_data_cleaned %>% select(TagCount, TitleLength, DayOfWeek, HourOfDay))
N <- nrow(stack_exch_data_cleaned)
K <- ncol(X)
y <- stack_exch_data_cleaned$HighPopularity

# Check for NAs
sum(is.na(X))

# Compare Bayesian Logistic Regression w/ appropriate priors to Frequentist Logistic Regression
# Create model for Frequentist LR
stck_exch_freq_model <- glm(HighPopularity ~ TagCount + TitleLength + DayOfWeek + HourOfDay, 
                            data = stack_exch_data_cleaned, family = binomial)
summary(stck_exch_freq_model)

# Read in Stan for Bayesian LR
suppressPackageStartupMessages(require(ggplot2))
suppressPackageStartupMessages(require(dplyr))

stack_exch_stan_model <- stan_model(file = "~/Downloads/stan_templates/from_r_scripts/stack_exch.stan")

stan_data <- list(
  N = N,
  K = K,
  y = y,
  X = X
)

fit <- sampling(
  stack_exch_stan_model,
  seed = 123,
  data = stan_data,
  iter = 2000,
  chains = 4,
  warmup = 500,
  control = list(adapt_delta = 0.95)  # Improves stability
)

print(fit)
print(fit, pars = "beta", probs = c(0.025, 0.5, 0.975))

# Get posterior sample and inspect posterior mean
posterior_samples <- extract(fit)
posterior_means <- apply(posterior_samples$beta, 2, mean)
print(posterior_means)

# Plot posterior distributions
coef_names <- colnames(X)
posterior_df <- data.frame(posterior_samples$beta)
colnames(posterior_df) <- coef_names

posterior_df_long <- posterior_df %>%
  pivot_longer(cols = everything(), names_to = "Coefficient", values_to = "Value")

ggplot(posterior_df_long, aes(x = Value, fill = Coefficient)) +
  geom_histogram(bins = 40, alpha = 0.6, position = "identity") +
  facet_wrap(~ Coefficient, scales = "free") +
  theme_minimal() +
  labs(title = "Posterior Distributions of Coefficients", x = "Coefficient Value", y = "Frequency")

# Extract predicted probabilities from Stan
# Compute mean posterior predicted probability for each observation
# Create plot of Actual vs. Predicted
posterior <- rstan::extract(fit)
posterior_preds <- posterior$y_pred_prob  

posterior_mean_preds <- colMeans(posterior_preds)
df_results <- stack_exch_data_cleaned
df_results$PosteriorMeanPred <- posterior_mean_preds

ggplot(df_results, aes(x = PosteriorMeanPred, y = HighPopularity)) +
  geom_jitter(height = 0.05, width = 0, alpha = 0.3) +
  geom_smooth(method = "loess", se = FALSE, color = "blue") +
  labs(title = "Posterior Predictive Check: Actual vs. Predicted",
       x = "Predicted Probability (Posterior Mean)",
       y = "Actual HighPopularity")

# Calibration Analysis and Plot
df_calibration <- df_results %>%
  mutate(prob_bin = cut(PosteriorMeanPred, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)) %>%
  group_by(prob_bin) %>%
  summarise(
    bin_center = mean(as.numeric(sub("\\((.+),.+", "\\1", as.character(prob_bin))) + 0.05),
    observed = mean(HighPopularity),
    predicted = mean(PosteriorMeanPred),
    count = n()
  )

ggplot(df_calibration, aes(x = bin_center)) +
  geom_line(aes(y = observed), color = "red") +
  geom_point(aes(y = observed, size = count)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") + # Perfect calibration line
  labs(title = "Calibration Plot",
       x = "Predicted Probability Bin Center",
       y = "Observed Proportion of High Popularity") +
  theme_minimal()


# AUC check
roc_obj <- roc(df_results$HighPopularity, df_results$PosteriorMeanPred)
auc(roc_obj)

df_results <- df_results %>%
  mutate(PredictedClass = ifelse(PosteriorMeanPred > 0.5, 1, 0))

mean(df_results$PredictedClass == df_results$HighPopularity)



