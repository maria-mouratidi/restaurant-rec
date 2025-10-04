# install.packages("patchwork")
# install.packages("RColorBrewer")

library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(patchwork) 

# DATA LOADING =================================================================
dat_original = read.csv('result_numeric_10242023.csv')

dat = read.csv('result_numeric_10242023.csv')

dat <- dat[-c(1,2),]

dat['Q2.1_1'][dat['StartDate'] == '2023-10-19 12:58:51'] <- 17
dat['Q2.1_1'][dat['StartDate'] == '2023-10-21 09:24:41'] <- 37

toString(dat$Q2.1_1)
df <- dat[order(as.numeric(as.character(dat$Q2.1_1))),]
df <- df[as.numeric(df$Q2.1_1) <= 50,]


# DEMOGRAPHICS =================================================================

participant_numbers <- as.numeric(df$Q2.1_1)
ages <- as.numeric(df$Q8.2_1)
genders <- as.numeric(df$Q8.3)
english_levels <- as.numeric(df$Q8.4_1)
ai_familiarity <- as.numeric(df$Q8.5_1)

vector_list <- list('Age' = ages, 'English Level' = english_levels, 'AI Familiarity' = ai_familiarity)


for (vec_name in names(vector_list)) {
    my_df <- data.frame(id = 1:length(vector_list[[vec_name]]), value = vector_list[[vec_name]])
    
    vec_data <- vector_list[[vec_name]]
    mean_val <- mean(vec_data, na.rm = TRUE)
    median_val <- median(vec_data, na.rm = TRUE)
    stdev_val <- sd(vec_data, na.rm = TRUE)
    
    subtitle_text <- paste("Mean: ", round(mean_val, 2),
                           " | Median: ", round(median_val, 2),
                           " | Std Dev: ", round(stdev_val, 2))
    
    x_label <- ifelse(vec_name == 'Age', 'Age', 'Likert Scale') 
    
    axis_limits <- range(vector_list[[vec_name]], na.rm = TRUE)
    axis_limits[1] <- axis_limits[1] - 0.5
    axis_limits[2] <- axis_limits[2] + 0.5  
    
    plt1 <- my_df %>% select(value) %>%
        ggplot(aes(x="", y = value)) +
        geom_boxplot(fill = "darkseagreen3", color = "black") +
        xlab("") +
        ylab(x_label) +
        coord_flip(ylim = axis_limits) +
        theme_classic() +
        theme(axis.text.y=element_blank(),
              axis.ticks.y=element_blank(),
              plot.title = element_text(size = 20),  # Adjust the size as needed
              plot.subtitle = element_text(size = 16)) +
        ggtitle(paste("Boxplot of", vec_name))
    
    plt2 <- my_df %>% select(id, value) %>%
        ggplot() +
        geom_histogram(aes(x = value, y = (..count..)/sum(..count..)),
                       position = "identity", binwidth = 1, 
                       fill = "darkseagreen3", color = "black") +
        ylab("Frequency Ratio") +
        xlab('') +
        theme_classic() +
        theme(plot.title = element_text(size = 20),  # Adjust the size as needed
            plot.subtitle = element_text(size = 16)) +
        ggtitle(paste("Histogram of", vec_name)) +
        coord_cartesian(xlim = axis_limits)  
    
    plt2 <- plt2 + labs(subtitle = subtitle_text)
    
    custom_theme <- theme_classic(base_size = 22)
    
    # Apply the custom theme to your plots
    plt1 <- plt1 + custom_theme
    plt2 <- plt2 + custom_theme
    
    final_plot <- plt2 / plt1 + plot_layout(nrow = 2, heights = c(3, 1))
    print(final_plot)
}

# Gender pie chart =============================================================

library(RColorBrewer)

gender_summary$fraction = gender_summary$Count / sum(gender_summary$Count)
gender_summary$ymax = cumsum(gender_summary$fraction)
gender_summary$ymin = c(0, head(gender_summary$ymax, n=-1))

pie_chart_gender <- ggplot(gender_summary, aes(ymax=ymax, ymin=ymin, xmax=4, xmin=3, fill=Gender)) +
    geom_rect(color="white") + 
    coord_polar(theta="y") +
    guides(fill = guide_legend(override.aes = list(label = ''))) +  # Remove the "a" from the legend
    scale_fill_brewer(palette="Pastel2") 

pie_chart_gender <- pie_chart_gender +
    geom_label(aes(x= ifelse(fraction < 0.05, 3.7, 3.5), y = ((ymin+ymax)/2), label = paste(round(Percentage, 1), "% (", Count, ")", sep = "")), size = 4 ) +
    ggtitle("Gender Distribution") +
    theme_void() +
  theme(plot.title = element_text(size = 20)) 
  

print(pie_chart_gender)
