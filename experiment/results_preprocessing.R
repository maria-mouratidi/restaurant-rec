#library(effectsize)
#library(lsr)

# Load the dataset
#setwd("!Github/new_mair/mair-rrds/experiment")
dat <- read.csv("result_numeric_2023-10-31.csv")


#to run for Maria only
#current_directory <- getwd()
#subdirectory <- "mair-rrds/experiment"
#file_path <- file.path(current_directory, specific_subdirectory, "result_numeric_10242023.csv")
#dat <- read.csv(file_path)

# Drop the first two rows
dat <- dat[-1,]
dat <- dat[-1,]

# Correct wrongly filled in participant numbers
dat['Q2.1_1'][dat['StartDate'] == '2023-10-19 12:58:51'] <- 17
dat['Q2.1_1'][dat['StartDate'] == '2023-10-21 09:24:41'] <- 37

toString(dat$Q2.1_1)
dat <- dat[order(as.numeric(as.character(dat$Q2.1_1))),]

# Define ordered conditions for each prompt
prompt_1 <- c('A', 'A', 'A', 'B', 'B', 'B')
prompt_2 <- c('A', 'B', 'B', 'A', 'A', 'B')
prompt_3 <- c('B', 'A', 'B', 'A', 'B', 'A')

# Repeat and truncate the prompts to match the number of rows in dat
num_rows <- nrow(dat)
dat$cond_1 <- head(rep(prompt_1, times = ceiling(num_rows / length(prompt_1))), num_rows)
dat$cond_2 <- head(rep(prompt_2, times = ceiling(num_rows / length(prompt_2))), num_rows)
dat$cond_3 <- head(rep(prompt_3, times = ceiling(num_rows / length(prompt_3))), num_rows)

### REFINING DATAFRAME FOR USAGE
# Remove unnecessary columns
dat <- subset(dat, select = -c(StartDate, EndDate, Status, Progress,
                               Finished, RecordedDate,
                               ResponseId, DistributionChannel, UserLanguage,
                               Q_RecaptchaScore, Q3.1, Q4.1, Q5.1))

# Rename columns
names(dat)[names(dat) == "Q1.1"] <- "Consent"
names(dat)[names(dat) == "Q2.1_1"] <- "n_Participant"
names(dat)[names(dat) == "Q8.2_1"] <- "Age"
names(dat)[names(dat) == "Q8.3"] <- "Gender"
names(dat)[names(dat) == "Q8.4_1"] <- "English_difficulty"
names(dat)[names(dat) == "Q8.5_1"] <- "AI_familiarity"

names(dat)[names(dat) == "Q7.1_1"] <- "Rec_satisfaction_1"
names(dat)[names(dat) == "Q7.1_2"] <- "Choice_satisfaction_1"
names(dat)[names(dat) == "Q7.1_3"] <- "Overwhelm_1"

names(dat)[names(dat) == "Q7.2_1"] <- "Rec_satisfaction_2"
names(dat)[names(dat) == "Q7.2_2"] <- "Choice_satisfaction_2"
names(dat)[names(dat) == "Q7.2_3"] <- "Overwhelm_2"

names(dat)[names(dat) == "Q7.3_1"] <- "Rec_satisfaction_3"
names(dat)[names(dat) == "Q7.3_2"] <- "Choice_satisfaction_3"
names(dat)[names(dat) == "Q7.3_3"] <- "Overwhelm_3"

# Remove failed test(s)
dat <- dat[as.numeric(dat$n_Participant) != 7,]
dat <- dat[as.numeric(dat$n_Participant) < 100,]

### RANDOMLY PICK OUT OF REPEATED CONDITIONS

# Function to randomly select conditions based on the provided sequence
random_select <- function(seq) {
  a_indices <- which(seq == 'A')
  b_indices <- which(seq == 'B')
  
  if (length(a_indices) == 2) {
    selected_a <- sample(a_indices, 1)
    selected_b <- b_indices
  } else if (length(b_indices) == 2) {
    selected_a <- a_indices
    selected_b <- sample(b_indices, 1)
  } else {
    indices <- sample(1:3, 2)
    selected_a <- indices[seq[indices] == 'A']
    selected_b <- indices[seq[indices] == 'B']
  }
  
  values <- c(seq[selected_a], seq[selected_b])
  names <- c(selected_a, selected_b)
  return(data.frame(A_col = values[1], B_col = values[2], A_prompt = names[1], B_prompt = names[2]))
}

#boxplot(dat_combined$Rec_A, dat_combined$Rec_B, 
#        main="Recommendation Satisfaction", 
#        xlab="Condition", 
#        ylab="Satisfaction Score",
#        names=c("A", "B"))

#boxplot(dat_combined$Choice_A, dat_combined$Choice_B, 
#        main="Choice Satisfaction", 
#        xlab="Condition", 
#        ylab="Satisfaction Score",
#        names=c("A", "B"))

#boxplot(dat_combined$Overwhelm_A, dat_combined$Overwhelm_B, 
#        main="Overwhelm", 
#        xlab="Condition", 
#        ylab="Score",
#        names=c("A", "B"))

# Loop for t-tests
num_loops <- 500
results_df <- data.frame(matrix(ncol = 21, nrow = num_loops))
colnames(results_df) <- c("Rec_p_value", "Choice_p_value", "Overwhelm_p_value", 
                          "Rec_statistic", "Choice_statistic", "Overwhelm_statistic",
                          "Rec_parameter", "Choice_parameter", "Overwhelm_parameter",
                          "Rec_A_mean", "Rec_B_mean", 
                          "Choice_A_mean", "Choice_B_mean", 
                          "Overwhelm_A_mean", "Overwhelm_B_mean",
                          "Rec_mean_diff", "Choice_mean_diff", "Overwhelm_mean_diff",
                          "Rec_d", "Choice_d", "Overwhelm_d")

cohens.d <- function(gr1, gr2) {
  var1 <- var(gr1)
  var2 <- var(gr2)
  denom <- sqrt((var1+var2)/2)
  nom <- mean(gr2) - mean(gr1)
  res <- nom / denom
  return(res)
}

for (i in 1:num_loops) {
  # Randomly pick out of repeated conditions
  results <- apply(dat[, c("cond_1", "cond_2", "cond_3")], 1, random_select)
  dat_new <- merge(dat, do.call(rbind, results), by=0)
  dat_combined <- merge(dat, dat_new)
  dat_combined <- dat_combined[order(as.numeric(as.character(dat_combined$n_Participant))),]
  
  # Convert response variables to numeric
  dat_combined$Rec_satisfaction_1 <- as.numeric(as.character(dat_combined$Rec_satisfaction_1))
  dat_combined$Rec_satisfaction_2 <- as.numeric(as.character(dat_combined$Rec_satisfaction_2))
  dat_combined$Rec_satisfaction_3 <- as.numeric(as.character(dat_combined$Rec_satisfaction_3))
  dat_combined$Choice_satisfaction_1 <- as.numeric(as.character(dat_combined$Choice_satisfaction_1))
  dat_combined$Choice_satisfaction_2 <- as.numeric(as.character(dat_combined$Choice_satisfaction_2))
  dat_combined$Choice_satisfaction_3 <- as.numeric(as.character(dat_combined$Choice_satisfaction_3))
  dat_combined$Overwhelm_1 <- as.numeric(as.character(dat_combined$Overwhelm_1))
  dat_combined$Overwhelm_2 <- as.numeric(as.character(dat_combined$Overwhelm_2))
  dat_combined$Overwhelm_3 <- as.numeric(as.character(dat_combined$Overwhelm_3))
  
  # Handle missing values
  dat_combined <- na.omit(dat_combined)
  
  # Create a column for each response for each condition
  dat_combined$Rec_A <- ifelse(dat_combined$A_prompt == 1, dat_combined$Rec_satisfaction_1,
                               ifelse(dat_combined$A_prompt == 2, dat_combined$Rec_satisfaction_2,
                                      dat_combined$Rec_satisfaction_3))
  dat_combined$Rec_B <- ifelse(dat_combined$B_prompt == 1, dat_combined$Rec_satisfaction_1,
                               ifelse(dat_combined$B_prompt == 2, dat_combined$Rec_satisfaction_2,
                                      dat_combined$Rec_satisfaction_3))
  dat_combined$Choice_A <- ifelse(dat_combined$A_prompt == 1, dat_combined$Choice_satisfaction_1,
                                  ifelse(dat_combined$A_prompt == 2, dat_combined$Choice_satisfaction_2,
                                         dat_combined$Choice_satisfaction_3))
  dat_combined$Choice_B <- ifelse(dat_combined$B_prompt == 1, dat_combined$Choice_satisfaction_1,
                                  ifelse(dat_combined$B_prompt == 2, dat_combined$Choice_satisfaction_2,
                                         dat_combined$Choice_satisfaction_3))
  dat_combined$Overwhelm_A <- ifelse(dat_combined$A_prompt == 1, dat_combined$Overwhelm_1,
                                     ifelse(dat_combined$A_prompt == 2, dat_combined$Overwhelm_2,
                                            dat_combined$Overwhelm_3))
  dat_combined$Overwhelm_B <- ifelse(dat_combined$B_prompt == 1, dat_combined$Overwhelm_1,
                                     ifelse(dat_combined$B_prompt == 2, dat_combined$Overwhelm_2,
                                            dat_combined$Overwhelm_3))
  
  # T-tests
  rec_t_test <- t.test(dat_combined$Rec_A, dat_combined$Rec_B, paired=TRUE)
  choice_t_test <- t.test(dat_combined$Choice_A, dat_combined$Choice_B, paired=TRUE)
  overwhelm_t_test <- t.test(dat_combined$Overwhelm_A, dat_combined$Overwhelm_B, paired=TRUE)
  
  # Bayesian effect size
  rec_mean_diff <- mean(dat_combined$Rec_B - dat_combined$Rec_A)
  choice_mean_diff <- mean(dat_combined$Choice_B - dat_combined$Choice_A)
  overwhelm_mean_diff <- mean(dat_combined$Overwhelm_B - dat_combined$Overwhelm_A)
  
  # Cohen's d
  rec_d <- cohens.d(dat_combined$Rec_A, dat_combined$Rec_B)
  choice_d <- cohens.d(dat_combined$Choice_A, dat_combined$Choice_B)
  overwhelm_d <- cohens.d(dat_combined$Overwhelm_A, dat_combined$Overwhelm_B)
  
  # Storing results
  results_df[i, ] <- c(rec_t_test$p.value, choice_t_test$p.value, overwhelm_t_test$p.value,
                       rec_t_test$statistic, choice_t_test$statistic, overwhelm_t_test$statistic,
                       rec_t_test$parameter, choice_t_test$parameter, overwhelm_t_test$parameter,
                       mean(dat_combined$Rec_A, na.rm = TRUE), mean(dat_combined$Rec_B, na.rm = TRUE),
                       mean(dat_combined$Choice_A, na.rm = TRUE), mean(dat_combined$Choice_B, na.rm = TRUE),
                       mean(dat_combined$Overwhelm_A, na.rm = TRUE), mean(dat_combined$Overwhelm_B, na.rm = TRUE),
                       rec_mean_diff, choice_mean_diff, overwhelm_mean_diff,
                       rec_d, choice_d, overwhelm_d)
}

# Show the results
#mean(results_df$Rec_statistic)
#mean(results_df$Choice_statistic)
#mean(results_df$Overwhelm_statistic)
#
#mean(results_df$Rec_parameter)
#mean(results_df$Choice_parameter)
#mean(results_df$Overwhelm_parameter)
#
#mean(results_df$Rec_p_value)
#mean(results_df$Choice_p_value)
#mean(results_df$Overwhelm_p_value)
#
#hist(results_df$Rec_p_value, breaks = seq(-0.1, 0.5, by=0.01))
#abline(v = 0.05, col='red')
#hist(results_df$Choice_p_value, breaks = seq(-0.1, 1, by=0.01))
#abline(v = 0.05, col='red')
#hist(results_df$Overwhelm_p_value, breaks = seq(-0.1, 1, by=0.01))
#abline(v = 0.05, col='red')
#
#hist(results_df$Rec_mean_diff, breaks = seq(-1, 1, by=0.01))
#abline(v = 0, col='red')
#hist(results_df$Choice_mean_diff, breaks = seq(-1, 1, by=0.01))
#abline(v = 0, col='red')
#hist(results_df$Overwhelm_mean_diff, breaks = seq(-1, 1, by=0.01))
#abline(v = 0, col='red')

add_eff_sizes <- function(){
  abline(v=0, col='black')
  cols <- c('forestgreen', 'chocolate', 'firebrick')
  i <- 1
  for (tick in c(0.2, 0.5, 0.8)){
    abline(v = tick, col=cols[i])
    axis(side=1, at=tick, col.axis=cols[i])
    i <- i + 1
  }
}

# Plot and calculate effect sizes according to Cohen's d
hist(results_df$Rec_d, breaks = seq(-0.3, 1, by=0.01), col='darkseagreen', border='darkseagreen',
     main="Histogram of Cohen's d for overall system satisfaction",
     xlab="Cohen's d for overall system satisfaction",
     ylab="Frequency over 500 iterations")
add_eff_sizes()
hist(results_df$Choice_d, breaks = seq(-0.3, 1, by=0.01), col='darkseagreen', border='darkseagreen',
     main="Histogram of Cohen's d for choice satisfaction",
     xlab="Cohen's d for choice satisfaction",
     ylab="Frequency over 500 iterations")
add_eff_sizes()
hist(results_df$Overwhelm_d, breaks = seq(-0.3, 1, by=0.01), col='darkseagreen', border='darkseagreen',
     main="Histogram of Cohen's d for overwhelmedness",
     xlab="Cohen's d for overwhelmedness",
     ylab="Frequency over 500 iterations")
add_eff_sizes()

expected <- rnorm(500,mean=0,sd=1)
lines(expected)

sml_Rec <- nrow(results_df[results_df$Rec_d > 0.2,]) / nrow(results_df)
med_Rec <- nrow(results_df[results_df$Rec_d > 0.5,]) / nrow(results_df)

sml_Choice <- nrow(results_df[results_df$Choice_d > 0.2,]) / nrow(results_df)
med_Choice <- nrow(results_df[results_df$Choice_d > 0.5,]) / nrow(results_df)

sml_Overwhelm <- nrow(results_df[results_df$Overwhelm_d > 0.2,]) / nrow(results_df)
med_Overwhelm <- nrow(results_df[results_df$Overwhelm_d > 0.5,]) / nrow(results_df)

sprintf("In overall system satisfaction, the probability of an at least small effect size was % s, whereas the probability of an at least medium effect size was % s", sml_Rec, med_Rec)
sprintf("In choice satisfaction, the probability of an at least small effect size was % s, whereas the probability of an at least medium effect size was % s", sml_Choice, med_Choice)
sprintf("In overwhelmedness, the probability of an at least small effect size was % s, whereas the probability of an at least medium effect size was % s", sml_Overwhelm, med_Overwhelm)
