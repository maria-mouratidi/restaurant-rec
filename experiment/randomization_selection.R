# Load necessary libraries
library(dplyr)

# Read the data
dat <- read.csv("C:/Users/alimo/Downloads/result_numeric_10242023.csv")

# Dropping the first two rows
dat <- dat[-c(1,2), ]

# Updating specific values in 'Q2.1_1' based on 'StartDate'
dat$Q2.1_1[dat$StartDate == '2023-10-19 12:58:51'] <- 17
dat$Q2.1_1[dat$StartDate == '2023-10-21 09:24:41'] <- 37

# Convert 'Q2.1_1' to numeric and order the dataframe based on this column
dat$Q2.1_1 <- as.numeric(as.character(dat$Q2.1_1))
df <- dat[order(dat$Q2.1_1), ]

# Ordered conditions for each prompt
prompt_1 <- c('A', 'A', 'A', 'B', 'B', 'B')
prompt_2 <- c('A', 'B', 'B', 'A', 'A', 'B')
prompt_3 <- c('B', 'A', 'B', 'A', 'B', 'A')

# Repeat and truncate the prompts to match the number of rows in df
num_rows <- nrow(df)
p1_rep <- head(rep(prompt_1, times = ceiling(num_rows / length(prompt_1))), num_rows)
p2_rep <- head(rep(prompt_2, times = ceiling(num_rows / length(prompt_2))), num_rows)
p3_rep <- head(rep(prompt_3, times = ceiling(num_rows / length(prompt_3))), num_rows)

# Add the conditions to the dataframe
df$cond_1 <- p1_rep
df$cond_2 <- p2_rep
df$cond_3 <- p3_rep

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
  return(data.frame(cond_1 = values[1], cond_2 = values[2], cond_1_name = names[1], cond_2_name = names[2]))
}

# Applying the function to each row of the dataframe
results <- apply(df[, c("cond_1", "cond_2", "cond_3")], 1, random_select)

# Creating a new dataframe to store the results
df_new <- do.call(rbind, results)

# Viewing the modified dataframe
head(df_new)
