library(dplyr)
library(tidyr)
library(tidyverse)

# Load the dataset
#setwd("!Github/mair-rrds/experiment")
dat <- read.csv("C:/Users/alimo/Downloads/result_numeric_10242023.csv")

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

# Applying the function to each row of the dataframe
results <- apply(dat[, c("cond_1", "cond_2", "cond_3")], 1, random_select)

# Creating a new dataframe to store the results
dat_new <- merge(dat, do.call(rbind, results), by=0)

# Merge dat_new with the original dataframe dat
dat_combined <- cbind(dat, dat_new)

### PREPARE T-TEST VALUES
# Convert response variables to numeric
dat_combined$Rec_satisfaction_1 <- as.numeric(dat_combined$Rec_satisfaction_1)
dat_combined$Rec_satisfaction_2 <- as.numeric(dat_combined$Rec_satisfaction_2)
dat_combined$Rec_satisfaction_3 <- as.numeric(dat_combined$Rec_satisfaction_3)
dat_combined$Choice_satisfaction_1 <- as.numeric(dat_combined$Choice_satisfaction_1)
dat_combined$Choice_satisfaction_2 <- as.numeric(dat_combined$Choice_satisfaction_2)
dat_combined$Choice_satisfaction_3 <- as.numeric(dat_combined$Choice_satisfaction_3)
dat_combined$Overwhelm_1 <- as.numeric(dat_combined$Overwhelm_1)
dat_combined$Overwhelm_2 <- as.numeric(dat_combined$Overwhelm_2)
dat_combined$Overwhelm_3 <- as.numeric(dat_combined$Overwhelm_3)

# Handle missing values (if any) - here we are removing rows with any missing value
dat_combined <- na.omit(dat_combined)
# Create a column for each response for each condition
dat_combined$Rec_A <- ifelse(
  dat_combined$A_prompt == 1,
  dat_combined$Rec_satisfaction_1,
  ifelse(
    dat_combined$A_prompt == 2,
    dat_combined$Rec_satisfaction_2,
    dat_combined$Rec_satisfaction_3
  )
)
dat_combined$Rec_B <- ifelse(
  dat_combined$A_prompt == 1,
  dat_combined$Choice_satisfaction_1,
  ifelse(
    dat_combined$A_prompt == 2,
    dat_combined$Choice_satisfaction_2,
    dat_combined$Choice_satisfaction_3
  )
)
dat_combined$Choice_A <- ifelse(
  dat_combined$A_prompt == 1,
  dat_combined$Overwhelm_1,
  ifelse(
    dat_combined$A_prompt == 2,
    dat_combined$Overwhelm_2,
    dat_combined$Overwhelm_3
  )
)

dat_combined$Choice_B <- ifelse(
  dat_combined$B_prompt == 1,
  dat_combined$Rec_satisfaction_1,
  ifelse(
    dat_combined$B_prompt == 2,
    dat_combined$Rec_satisfaction_2,
    dat_combined$Rec_satisfaction_3
  )
)
dat_combined$Overwhelm_A <- ifelse(
  dat_combined$B_prompt == 1,
  dat_combined$Choice_satisfaction_1,
  ifelse(
    dat_combined$B_prompt == 2,
    dat_combined$Choice_satisfaction_2,
    dat_combined$Choice_satisfaction_3
  )
)
dat_combined$Overwhelm_B <- ifelse(
  dat_combined$B_prompt == 1,
  dat_combined$Overwhelm_1,
  ifelse(
    dat_combined$B_prompt == 2,
    dat_combined$Overwhelm_2,
    dat_combined$Overwhelm_3
  )
)

# Now you can use dat_combined for your t-tests
rec_t_test <- t.test(dat_combined$Rec_A, dat_combined$Rec_B)
choice_t_test <- t.test(dat_combined$Choice_A, dat_combined$Choice_B)
overwhelm_t_test <- t.test(dat_combined$Overwhelm_A, dat_combined$Overwhelm_B)

cat("Rec satisfaction:", "p-value =", rec_t_test$p.value, "\n")
cat("Choice satisfaction:", "p-value =", choice_t_test$p.value, "\n")
cat("Overwhelm:", "p-value =", overwhelm_t_test$p.value, "\n\n")

mean(dat_combined$Rec_A)
mean(dat_combined$Rec_B)

mean(dat_combined$Choice_A)
mean(dat_combined$Choice_B)

mean(dat_combined$Overwhelm_A)
mean(dat_combined$Overwhelm_B)

boxplot(dat_combined$Rec_A, dat_combined$Rec_B, 
        main="Recommendation Satisfaction", 
        xlab="Condition", 
        ylab="Satisfaction Score",
        names=c("A", "B"))

boxplot(dat_combined$Choice_A, dat_combined$Choice_B, 
        main="Choice Satisfaction", 
        xlab="Condition", 
        ylab="Satisfaction Score",
        names=c("A", "B"))

boxplot(dat_combined$Overwhelm_A, dat_combined$Overwhelm_B, 
        main="Overwhelm", 
        xlab="Condition", 
        ylab="Score",
        names=c("A", "B"))


write.csv(dat_combined, "test.csv", row.names=FALSE)
# Loop for t-tests
results_df <- data.frame(matrix(ncol = 9, nrow = 10))
colnames(results_df) <- c("Rec_p_value", "Choice_p_value", "Overwhelm_p_value", "Rec_A_mean", "Rec_B_mean", "Choice_A_mean", "Choice_B_mean", "Overwhelm_A_mean", "Overwhelm_B_mean")

set.seed(123)  # Replace 123 with your chosen seed value

for (i in 1:10) {
  # Randomly pick out of repeated conditions
  results <- apply(dat[, c("cond_1", "cond_2", "cond_3")], 1, random_select)
  dat_new <- merge(dat, do.call(rbind, results), by=0)
  dat_combined <- cbind(dat, dat_new)
  
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
  rec_t_test <- t.test(dat_combined$Rec_A, dat_combined$Rec_B)
  choice_t_test <- t.test(dat_combined$Choice_A, dat_combined$Choice_B)
  overwhelm_t_test <- t.test(dat_combined$Overwhelm_A, dat_combined$Overwhelm_B)

  # Storing results
  results_df[i, ] <- c(rec_t_test$p.value, choice_t_test$p.value, overwhelm_t_test$p.value,
                       mean(dat_combined$Rec_A, na.rm = TRUE), mean(dat_combined$Rec_B, na.rm = TRUE),
                       mean(dat_combined$Choice_A, na.rm = TRUE), mean(dat_combined$Choice_B, na.rm = TRUE),
                       mean(dat_combined$Overwhelm_A, na.rm = TRUE), mean(dat_combined$Overwhelm_B, na.rm = TRUE))
}

# Show the results
print(results_df)
