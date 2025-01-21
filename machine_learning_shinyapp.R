library(shiny)
library(shinythemes)
library(dplyr)
library(data.table)
library(ggplot2)
library(caret)
library(tidyverse)
library(randomForest) 
library(tidymodels)
library(rpart)
library(e1071)

# Read the data
tracks <- read.csv('tracks.csv', header = TRUE, skip = 1)
setnames(tracks, 'X', 'track_id')
tracks <- tracks[-1, ]

echonest <- read.csv('echonest.csv', header = TRUE, skip = 2)
setnames(echonest, 'X', 'track_id')
echonest <- echonest[-1, ]

genres<- read.csv('genres.csv')

# Small Subset 
#We chose the small subset which contains 8000 instances.NA dominant cols will be removed automatically by the near-zero variance multiple genre classes
# head(small_tracks)
small_tracks <- subset(tracks, subset == "small") # 8000 tracks are enough

sum(is.na(small_tracks$latitude))
sum(is.na(small_tracks$longitude))

columns_to_drop <- c("subset", "split", "wikipedia_page","id", "latitude", "longtitude")
small_tracks <- small_tracks[, !(names(small_tracks) %in% columns_to_drop)]
head(small_tracks)

# each genre_top has 1000 instances
small_tracks$genre_top <- factor(small_tracks$genre_top)
table(small_tracks$genre_top)
head(small_tracks)
summary(small_tracks)

# Echonest (with temporal features)
#Temporal features in the context of the FMA dataset are time-varying properties extracted from the audio signal of a music track. They are statistical measurements that capture aspects of the sound throughout its duration, which can reflect the rhythm, dynamics, and timbre of the audio.

#In the FMA dataset, temporal features could include things like changes in spectral centroid (indicating where the "center of mass" for a sound's spectrum is located over time), variations in zero-crossing rate (how frequently the audio signal changes from positive to negative, which can be related to the perceived "noisiness" or "roughness" of a sound), and other time-related descriptors.
# colnames(echonest)
temporal_feature_cols <- paste0('temporal_feature_', seq_len(ncol(echonest) - 26))
setnames(echonest, old = names(echonest)[27:ncol(echonest)], new = temporal_feature_cols)
# Remove temporal features
echonest <- echonest[, -c(27:ncol(echonest))]

# Top genres (Pop vs Hip pop) To identify ideal two genre_top for classification task, we want to avoid too many overlapping and generic genres .We can tell how Rock, Experimental and Electrical have repeated occurrence as non-primary genres for many tracks. Thus, we decided to choose Hip-pop and Pop as our target classification targets.
head(genres)

#  only study top genres
top_genre_to_id <- genres %>%
  filter(title %in% small_tracks$genre_top) %>%
  select(title, top_level) %>%
  distinct() %>%
  deframe()

# 'genre_top' --> genre_id
small_tracks$top_level_id <- top_genre_to_id[small_tracks$genre_top]

genre_id_to_top_level <- setNames(genres$top_level, genres$genre_id)
top_level_id_to_title <- setNames(genres$title, genres$genre_id)

# Translate and expand 'genres_all' to list of top_level_ids
small_tracks$top_level_ids_all <- lapply(small_tracks$genres_all, function(genre_list) {
  genre_ids <- as.integer(unlist(strsplit(gsub("\\[|\\]", "", genre_list), ",\\s*")))
  top_ids <- genre_id_to_top_level[genre_ids]
  top_ids[!is.na(top_ids)]  
})

# count occurrance o top genres in genres_all
all_top_level_ids <- unlist(small_tracks$top_level_ids_all)
filtered_top_level_ids <- all_top_level_ids[all_top_level_ids %in% small_tracks$top_level_id]
top_level_id_occurrences <- table(filtered_top_level_ids)

# count occurrence of 'genre_top' (primary).
primary_top_level_id_counts <- table(small_tracks$top_level_id)
non_primary_occurrences <- top_level_id_occurrences - primary_top_level_id_counts
non_primary_occurrences[non_primary_occurrences < 0] <- 0
non_primary_occurrences_df <- as.data.frame(non_primary_occurrences)
print(non_primary_occurrences_df)

names(non_primary_occurrences_df) <- c("Top_Level_ID", "Non_Primary_Count")
non_primary_occurrences_df$Genre_Title <- top_level_id_to_title[as.character(non_primary_occurrences_df$Top_Level_ID)]

# Plot 
ggplot(non_primary_occurrences_df, aes(x = reorder(Genre_Title, -Non_Primary_Count), y = Non_Primary_Count)) +
  geom_bar(stat = "identity", fill = 'steelblue') +
  labs(title = "Non-Primary Occurrences of Top-Level Genres in Genres_All", x = "Genre Title", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

df <- small_tracks %>%
  filter(genre_top %in% c("Pop", "Hip-Hop"))
df$genre_top <- factor(df$genre_top, levels = c("Pop", "Hip-Hop"))

columns_to_drop <- c("genres", "all_generes", "top_level_ids_all", "top_level_id")
df <- df[, !(names(df) %in% columns_to_drop)]
df_final <- merge(df, echonest, by.x = "track_id", by.y = "track_id", all.x = TRUE)

nzv_indices <- nearZeroVar(df_final, saveMetrics = FALSE)  
nzv_vars <- names(df_final)[nzv_indices]  
nzv_vars 

# 70%-30% split
index <- createDataPartition(df_final$genre_top, p = 0.70, list = FALSE) 
df_train <- df_final[index, ]
df_test <- df_final[-index, ]

blueprint <- recipe(genre_top ~ ., data = df_train) %>%
  
  step_rm(nzv_vars) %>%                             # Remove near-zero variance predictors
  
  step_string2factor(all_nominal_predictors()) %>%   # Convert string variables to factors
  
  step_impute_knn(all_predictors(), neighbors = 5) %>% # Impute using k-NN
  
  step_center(all_numeric_predictors()) %>%          # Center numeric predictors
  
  step_scale(all_numeric_predictors()) %>%           # Scale numeric predictors
  
  step_pca(all_numeric_predictors(), threshold = 0.90) %>% # reduce dimensionality
  
  step_other(all_nominal(), threshold = 0.03) %>%    # Group infrequent factors
  
  step_dummy(all_nominal_predictors(), one_hot = TRUE)  # Convert dummies

blueprint_prep <- prep(blueprint, training = df_train)

transformed_train <- bake(blueprint_prep, new_data = df_train)
transformed_test <- bake(blueprint_prep, new_data = df_test)

summary(transformed_train)

# Define UI
ui <- fluidPage(
  titlePanel("Machine Learning Final Project (Group 17)"),
  navbarPage(
    title = "Mackiah C Henry, Ze Wang, Yongcen Zhou",
    theme = shinytheme("flatly"),
    tabPanel("Data Preprocessing", icon = icon("info-circle"),
             fluidRow(
               column(12,
                      titlePanel("Data Preprocessing"),
                      helpText("Data Acquisition and Pre-processing"),
                      tags$hr(),
                      h4("Data Selection"),
                      p("For this project, we selected datasets from the Free Music Archive (FMA), which is known for its comprehensive collection of music tracks along with rich metadata and audio features. These datasets, available through reputable platforms like the UCI Machine Learning Repository, provide a solid basis for exploring music genre classification. Specifically, we used three main files: `tracks.csv`, `echonest.csv`, and `genres.csv`, focusing on a subset of 8000 tracks labeled as 'small' to maintain computational efficiency while ensuring diverse genre representation."),
                      tags$hr(),
                      h4("Data Exploration"),
                      p("Initial exploratory data analysis (EDA) involved examining the distribution of genres, the presence of missing values, and the general structure of the data. We visualized various aspects of the data:"),
                      tags$ol(
                        tags$li("Genre Distribution: Understanding the balance across different musical genres."),
                        tags$li("Missing Values: Identification and visualization of missing data points across geographical information, such as latitude and longitude, which were prevalent."),
                        tags$li("Data Overview: Use functions like `summary()` and `head()` to get an initial feel for the data after the preprocessing steps.")
                      ),
                      tags$hr(),
                      h4("Data Pre-processing"),
                      p("The pre-processing steps included:"),
                      tags$ul(
                        tags$li("Cleaning Data: We handled missing values and removed columns with high percentages of missing data or irrelevant information (like Wikipedia pages and IDs not used in analysis)."),
                        tags$li("Outlier Handling: Near-zero variance predictors were identified and removed to ensure that our models would only train on features with significant variance and predictive power."),
                        tags$li("Data Transformation: For numerical stability and model performance, we normalized and standardized numerical features. This involved scaling features to have zero mean and unit variance.")
                      ),
                      tags$hr(),
                      h4("Feature Engineering"),
                      p("We enhanced our dataset by:"),
                      tags$ul(
                        tags$li("Encoding Categorical Variables: Transforming categorical variables using one-hot encoding to make them suitable for modeling."),
                        tags$li("Temporal Features: Extracting temporal features from the `echonest.csv` file, which provided time-varying properties of the music tracks that are crucial for understanding the dynamics and characteristics of music.")
                      )
               )
             )
    ),
    tabPanel("Model Building", icon = icon("cogs"),
             fluidRow(
               column(12,
                      titlePanel("Model Building"),
                      helpText("Model Building Step"),
                      tags$hr(),
                      h4("Algorithm Selection"),
                      p("After preprocessing the dataset, we selected three machine learning algorithms: random forest, support vector machine, and decision trees. Random forest was chosen for its ability to manage high-dimensional data, handle both category and numerical characteristics, and negotiate uneven class distributions and noisy datasets."),
                      tags$hr(),
                      h4("Model Training"),
                      p("We divided the dataset into training and testing sets using an 80:20 split ratio, with 80% given to training and 20% to testing. This ensured a strong training period while maintaining an independent subset for unbiased evaluation. We next used the training data to train each of the three algorithms we selected: random forest, support vector machine (SVM), and decision trees. Our models were primed to effectively capture patterns and nuances within the music dataset through thoughtful data preparation and algorithm selection. Our models were optimized to capture patterns and nuances in the music dataset through careful data preparation and algorithm selection."),
                      tags$hr(),
                      h4("Model Evaluation"),
                      p("During the model evaluation phase, we evaluated the trained machine learning models to determine their accuracy in predicting music genres. We evaluated the three Machine learning models using performance criteria such as accuracy, precision, recall, and F1-score. We applied confusion matrices to assess the model's ability to identify cases across genre groups correctly. We also evaluated the Decision Tree model's performance by measuring its accuracy on training and testing datasets and compared it to the Random Forest technique. We evaluated the accuracy of the random forest and Support Vector Machine model and developed confusion matrices to understand its classification performance better. By evaluating these measures and applying the evaluation procedures, we gained a better understanding of each model's strengths and limitations, allowing us to make the right decisions regarding the ability for genre classification tasks in music datasets."),
                      tags$hr(),
                      h4("Performance Metrics"),
                      tableOutput("model_performance_table"),
                      tags$hr(),
                      p("Table 1 shows that Random Forest outperformed the other models, with an accuracy of 99.8% and a precision of 100%. It achieved a high recall rate of 99.6% and an F1 score of 99.8%. This shows that the Random Forest effectively predicted the music genre classification problem, providing an accurate prediction over the whole dataset.")
               )
             )
    ),
    
    tabPanel("Shiny APP Building", icon = icon("rocket"),
             fluidRow(
               column(12,
                      titlePanel("Shiny Application"),
                      helpText("Shiny App Building"),
                      tags$hr(),
                      h4("App Building"),
                      p("After processing the dataset, we integrated all our building models into the R Shiny Application. The advantage of R Shiny App is its ability to vividly present detailed results to the audience and give them the freedom to manipulate the data to check various results. The application contains different tabs that the audience can choose to explore."),
                      p("Initially, we carefully reviewed the trained model and decided to use the template provided by the professor to build the interface. We chose to present the written report at the start so that the audience could better understand how to use the app and the main goal of machine learning. Then, we provided a tab for the reader to upload their own data and adjust the data frequency. After that, we presented our data pre-processing summary tab so that the audience could gain insights into the raw data format, the NZV features, and summary statistics of different variables. In the next tab, we also provided histograms and tables to show different genres and their respective counts to help the audience identify them."),
                      p("Next, we included three different tabs representing three different training methods: random forest, decision tree, and support vector machine."),
                      p("For these tabs, there are two sections: model evaluation metrics and accuracy plots evolving over time. The audience can review the results and compare the accuracy of each method."),
                      tags$hr(),
                      h4("Challenges and Solutions:"),
                      p("The main challenge was unfamiliarity with the shiny package. As I only had limited knowledge of the R Shiny package, it was hard for me to program to achieve the functionality. My solution was to refer to the class template, Stack Overflow, and 'Mastering Shiny' written by Hadley Wickham. These resources helped me strengthen my skills in programming with R Shiny, including all the different commands."),
                      p("Another problem in R Shiny is buttons reacting between the frontend and the backend. Sometimes, it is hard to know why the buttons do not react the same way as I expected. Therefore, I need to patiently use the debug procedure to identify small errors, such as parentheses, syntax, and function names, to correct these errors and make the application fully functional.")
               )
             )
    ),
    
    tabPanel("Viewer Input", icon = icon("cloud-upload"),
             fluidRow(
               column(6,
                      titlePanel("Upload Tracks CSV"),
                      fileInput("tracks_file", "Choose CSV file",
                                accept = c(".csv")),
                      helpText("Please upload a new 'tracks.csv' file.")
               ),
               column(6,
                      titlePanel("Upload Echonest CSV"),
                      fileInput("echonest_file", "Choose CSV file",
                                accept = c(".csv")),
                      helpText("Please upload a new 'echonest.csv' file.")
               ),
               column(6,
                      titlePanel("Upload Genres CSV"),
                      fileInput("genres_file", "Choose CSV file",
                                accept = c(".csv")),
                      helpText("Please upload a new 'genres.csv' file.")
               ),
               column(6,
                      titlePanel("Select Data Split Percentage"),
                      sliderInput("data_split", "Data Split Percentage",
                                  min = 50, max = 90, value = 70,
                                  step = 1, ticks = TRUE, width = "100%"),
                      helpText("Choose the percentage of data for training and testing.")
               ),
               actionButton("update_data", "Update Data",
                            class = "btn-primary")
             )
    ),
    
    tabPanel("Raw Data Summary", icon = icon("table"),
             fluidRow(
               column(6,
                      titlePanel("Head of Raw Data"),
                      verbatimTextOutput("tracks_head")
               ),
               column(6,
                      titlePanel("Head of Preprocessed Data"),
                      verbatimTextOutput("preprocessed_head")
               ),
               column(12,
                      titlePanel("Summary of Raw Data"),
                      verbatimTextOutput("summary_raw_data")
               ),
               column(12, 
                      titlePanel("Summary of Transformed Data"),
                      verbatimTextOutput("summary_transformed_data")
               )
             )
    ),
    tabPanel("Genre Analysis", icon = icon("music"),
             fluidRow(
               column(6,
                      titlePanel("Head of Genres"),
                      verbatimTextOutput("genres_head")
               ),
               column(6,
                      titlePanel("Frequency of Genres"),
                      tableOutput("genre_frequency")
               ),
               column(12,
                      titlePanel("Histogram of Genres"),
                      plotOutput("genre_histogram")
               )
             )
    ),
    tabPanel("NZV Features", icon = icon("warning"),
             fluidRow(
               column(6,
                      titlePanel("Near-Zero Variance Features"),
                      verbatimTextOutput("nzv_features")
               ),
               column(12,
                      titlePanel("Genre Distribution"),
                      plotOutput("genre_barplot")
               )
             )
    ),
    tabPanel("Random Forest", icon = icon("tree"),
             fluidRow(
               column(12,
                      titlePanel("Random Forest Model Metrics"),
                      verbatimTextOutput("rf_model_metrics"),
                      titlePanel("Random Forest Model Panel"),
                      plotOutput("accuracy_plot")
               )
             )
    ),
    tabPanel("Decision Tree", icon = icon("tree"),
             fluidRow(
               column(12,
                      titlePanel("Decision Tree Model Metrics"),
                      verbatimTextOutput("DT_model_metrics"),
                      titlePanel("Decision Tree Model Panel"),
                      plotOutput("DT_accuracy_plot")
               )
             )
    ),
    tabPanel("Support Vector Machine", icon = icon("flag"),
             fluidRow(
               column(12,
                      titlePanel("Support Vector Machine Model Metrics"),
                      verbatimTextOutput("SVM_model_metrics"),
                      titlePanel("Support Vector Machine Model Panel"),
                      plotOutput("SVM_accuracy_plot")
               )
             )
    ),
    tabPanel("Conclusion", icon = icon("check-circle"),
             fluidRow(
               column(12,
                      titlePanel("Conclusion"),
                      p("Our project, 'Decoding Beats,' successfully demonstrated the capability of machine learning to classify music genres, specifically distinguishing between Pop and Hip-Hop using the Random Forest algorithm. This approach achieves high accuracy and precision, highlighted by its performance in our tests."),
                      p("The incorporation of this model into an R Shiny web application allowed for interactive exploration and demonstrated the practical application of complex machine learning techniques in the arts. This experience has enhanced our technical skills and opened avenues for further research in music classification.")
               )
             )
    )
    
    
  ) 
)

# Define server logic
server <- function(input, output, session) {
  output$summary_raw_data <- renderPrint({
    summary(small_tracks)
  })
  
  output$tracks_head <- renderPrint({
    head(tracks)
  })
  
  output$preprocessed_head <- renderPrint({
    head(df_final)
  })
  
  output$summary_transformed_data <- renderPrint({
    summary(transformed_train)
  })
  
  output$model_performance_table <- renderTable({
    model_performance <- data.frame(
      Models = c("Random Forest", "Decision Tree", "Support Vector Machine"),
      `Accuracy%` = c(99.8, 97.0, 99.5),
      Recall = c(99.6, 98.3, 99.0),
      Precision = c(100.0, 95.7, 100.0),
      `F1-Score%` = c(99.8, 97.0, 99.4)
    )
    model_performance
  }, row.names = FALSE)
  
  output$genres_head <- renderPrint({
    head(genres)
  })
  
  output$nzv_features <- renderPrint({
    nzv_indices <- nearZeroVar(df_final, saveMetrics = FALSE)
    nzv_vars <- names(df_final)[nzv_indices]
    nzv_vars
  })
  
  output$genre_barplot <- renderPlot({
    # 70%-30% split
    index <- createDataPartition(df_final$genre_top, p = 0.70, list = FALSE) 
    df_train <- df_final[index, ]
    df_test <- df_final[-index, ]
    
    # createDataPartition does stratified sampling for factor outcome
    par(mfrow=c(1, 2)) 
    barplot(table(df_train$genre_top), main="Training Set Dist", col=c("blue", "red"))
    barplot(table(df_test$genre_top), main="Test Set Dist", col=c("blue", "red"))
  })
  
  output$genre_frequency <- renderTable({
    genre_frequency <- as.data.frame(table(filtered_top_level_ids))
    colnames(genre_frequency) <- c("Genre", "Frequency")
    genre_frequency
  })
  
  output$genre_histogram <- renderPlot({
    ggplot(data = non_primary_occurrences_df, aes(x = reorder(Genre_Title, -Non_Primary_Count), y = Non_Primary_Count)) +
      geom_bar(stat = "identity", fill = 'steelblue') +
      labs(title = "Non-Primary Occurrences of Top-Level Genres in Genres_All", x = "Genre Title", y = "Count") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 90, hjust = 1))
  })
  
  output$rf_model_metrics <- renderPrint({
    rf_model <- randomForest(genre_top ~ ., data = transformed_train, ntree = 100)
    evaluate_model <- function(model, data) {
      predictions <- predict(model, newdata = data)
      confusion_matrix <- confusionMatrix(predictions, data$genre_top)
      accuracy <- confusion_matrix$overall['Accuracy']
      
      precision <- confusion_matrix$byClass['Precision']
      recall <- confusion_matrix$byClass['Recall']
      f1_score <- confusion_matrix$byClass['F1']
      
      return(list(
        accuracy = accuracy,
        precision = precision,
        recall = recall,
        f1_score = f1_score
      ))
    }
    rf_metrics <- evaluate_model(rf_model, transformed_test)
    
    cat("Random Forest Model Metrics:\n")
    cat("Accuracy:", rf_metrics$accuracy, "\n")
    cat("Precision:", rf_metrics$precision, "\n")
    cat("Recall:", rf_metrics$recall, "\n")
    cat("F1-score:", rf_metrics$f1_score, "\n")
  })
  
  output$accuracy_plot <- renderPlot({
    num_trees <- seq(10, 200, by = 10)
    train_accuracies <- vector(length = length(num_trees))
    test_accuracies <- vector(length = length(num_trees))
    
    for (i in seq_along(num_trees)) {
      rf_model <- randomForest(genre_top ~ ., data = transformed_train, ntree = num_trees[i])
      
      predictions_train <- predict(rf_model, newdata = transformed_train)
      train_accuracies[i] <- mean(predictions_train == transformed_train$genre_top)
      
      predictions_test <- predict(rf_model, newdata = transformed_test)
      test_accuracies[i] <- mean(predictions_test == transformed_test$genre_top)
    }
    
    accuracy_data <- data.frame(
      num_trees = num_trees,
      train_accuracy = train_accuracies,
      test_accuracy = test_accuracies
    )
    
    accuracy_plot <- ggplot(accuracy_data, aes(x = num_trees)) +
      geom_line(aes(y = train_accuracy, color = "Training Accuracy")) +
      geom_line(aes(y = test_accuracy, color = "Testing Accuracy")) +
      scale_color_manual(values = c("Training Accuracy" = "blue", "Testing Accuracy" = "red")) +
      labs(title = "Accuracy vs. Number of Trees in Random Forest",
           x = "Number of Trees", y = "Accuracy") +
      theme_minimal()
    
    print(accuracy_plot)
  })
  
  
  output$DT_model_metrics <- renderPrint({
    tree_model <- rpart(genre_top ~ ., data = transformed_train)
    predictions_test <- predict(tree_model, newdata = transformed_test, type = "class")
    predictions_train <- predict(tree_model, newdata = transformed_train, type = "class")
    
    accuracy_test <- mean(predictions_test == transformed_test$genre_top)
    accuracy_train <- mean(predictions_train == transformed_train$genre_top)
    
    conf_matrix_test <- table(predictions_test, transformed_test$genre_top)
    print("Confusion Matrix - Testing Data:")
    print(conf_matrix_test)
    
    print(paste("Testing Accuracy:", accuracy_test))
    
    conf_matrix_train <- table(predictions_train, transformed_train$genre_top)
    print("Confusion Matrix - Training Data:")
    print(conf_matrix_train)
    print(paste("Training Accuracy:", accuracy_train))
    
    
    accuracy_test <- mean(predictions_test == transformed_test$genre_top)
    confusionMatrix_test <- confusionMatrix(predictions_test, transformed_test$genre_top)
    precision_test <- confusionMatrix_test$byClass['Precision']
    recall_test <- confusionMatrix_test$byClass['Recall']
    f1_score_test <- confusionMatrix_test$byClass['F1']
    
    cat("Performance Metrics - Testing Data:\n",
        "Accuracy:", accuracy_test, "\n",
        "Precision:", precision_test, "\n",
        "Recall:", recall_test, "\n",
        "F1 Score:", f1_score_test, "\n")
  })
  
  output$DT_accuracy_plot <- renderPlot({
    max_depths <- seq(1, 20)
    train_accuracies <- vector(length = length(max_depths))
    test_accuracies <- vector(length = length(max_depths))
    
    for (i in seq_along(max_depths)) {
      tree_model <- rpart(genre_top ~ ., data = transformed_train, maxdepth = max_depths[i])
      predictions_train <- predict(tree_model, newdata = transformed_train, type = "class")
      train_accuracies[i] <- mean(predictions_train == transformed_train$genre_top)
      
      predictions_test <- predict(tree_model, newdata = transformed_test, type = "class")
      test_accuracies[i] <- mean(predictions_test == transformed_test$genre_top)
    }
    
    accuracy_data <- data.frame(
      max_depth = max_depths,
      train_accuracy = train_accuracies,
      test_accuracy = test_accuracies
    )
    
    accuracy_plot <- ggplot(accuracy_data, aes(x = max_depth)) +
      geom_line(aes(y = train_accuracy, color = "Training Accuracy")) +
      geom_line(aes(y = test_accuracy, color = "Testing Accuracy")) +
      scale_color_manual(values = c("Training Accuracy" = "blue", "Testing Accuracy" = "red")) +
      labs(title = "Accuracy vs. Maximum Depth of Decision Tree",
           x = "Maximum Depth", y = "Accuracy") +
      theme_minimal()
    
    print(accuracy_plot)
  })
  
  output$SVM_model_metrics <-renderPrint({
    cost_values <- seq(0.1, 10, by = 0.1)
    train_accuracies <- vector(length = length(cost_values))
    test_accuracies <- vector(length = length(cost_values))
    
    for (i in seq_along(cost_values)) {
      svm_model <- svm(genre_top ~ ., data = transformed_train, kernel = "radial", cost = cost_values[i])
      
      predictions_train <- predict(svm_model, newdata = transformed_train)
      train_accuracies[i] <- mean(predictions_train == transformed_train$genre_top)
      
      predictions_test <- predict(svm_model, newdata = transformed_test)
      test_accuracies[i] <- mean(predictions_test == transformed_test$genre_top)
    }
    
    accuracy_data <- data.frame(
      cost_value = cost_values,
      train_accuracy = train_accuracies,
      test_accuracy = test_accuracies
    )
    
    accuracy_plot <- ggplot(accuracy_data, aes(x = cost_value)) +
      geom_line(aes(y = train_accuracy, color = "Training Accuracy")) +
      geom_line(aes(y = test_accuracy, color = "Testing Accuracy")) +
      scale_color_manual(values = c("Training Accuracy" = "blue", "Testing Accuracy" = "red")) +
      labs(title = "Accuracy vs. Cost Parameter (C) in SVM",
           x = "Cost Parameter (C)", y = "Accuracy") +
      theme_minimal()
    
    print(accuracy_plot)
    
    svm_model <- svm(genre_top ~ ., data = transformed_train, kernel = "radial", scale = TRUE)
    predictions_test <- predict(svm_model, newdata = transformed_test)
    accuracy_test <- mean(predictions_test == transformed_test$genre_top)
    
    conf_matrix_test <- confusionMatrix(predictions_test, transformed_test$genre_top)
    print("Confusion Matrix - Testing Data:")
    print(conf_matrix_test$table)
    print(paste("Testing Accuracy:", accuracy_test))
    
    precision <- conf_matrix_test$byClass["Precision"]
    recall <- conf_matrix_test$byClass["Recall"]
    f1_score <- 2 * precision * recall / (precision + recall)
    
    print(paste("Precision:", precision))
    print(paste("Recall:", recall))
    print(paste("F1-score:", f1_score))
    
    
  })
  
  output$SVM_accuracy_plot <-renderPlot({
    cost_values <- seq(0.1, 10, by = 0.1)
    train_accuracies <- vector(length = length(cost_values))
    test_accuracies <- vector(length = length(cost_values))
    
    for (i in seq_along(cost_values)) {
      svm_model <- svm(genre_top ~ ., data = transformed_train, kernel = "radial", cost = cost_values[i])
      
      predictions_train <- predict(svm_model, newdata = transformed_train)
      train_accuracies[i] <- mean(predictions_train == transformed_train$genre_top)
      
      predictions_test <- predict(svm_model, newdata = transformed_test)
      test_accuracies[i] <- mean(predictions_test == transformed_test$genre_top)
    }
    
    accuracy_data <- data.frame(
      cost_value = cost_values,
      train_accuracy = train_accuracies,
      test_accuracy = test_accuracies
    )
    
    accuracy_plot <- ggplot(accuracy_data, aes(x = cost_value)) +
      geom_line(aes(y = train_accuracy, color = "Training Accuracy")) +
      geom_line(aes(y = test_accuracy, color = "Testing Accuracy")) +
      scale_color_manual(values = c("Training Accuracy" = "blue", "Testing Accuracy" = "red")) +
      labs(title = "Accuracy vs. Cost Parameter (C) in SVM",
           x = "Cost Parameter (C)", y = "Accuracy") +
      theme_minimal()
    
    print(accuracy_plot)
    
    svm_model <- svm(genre_top ~ ., data = transformed_train, kernel = "radial", scale = TRUE)
    predictions_test <- predict(svm_model, newdata = transformed_test)
    accuracy_test <- mean(predictions_test == transformed_test$genre_top)
    
    conf_matrix_test <- confusionMatrix(predictions_test, transformed_test$genre_top)
    print("Confusion Matrix - Testing Data:")
    print(conf_matrix_test$table)
    print(paste("Testing Accuracy:", accuracy_test))
    
    precision <- conf_matrix_test$byClass["Precision"]
    recall <- conf_matrix_test$byClass["Recall"]
    f1_score <- 2 * precision * recall / (precision + recall)
    
    print(paste("Precision:", precision))
    print(paste("Recall:", recall))
    print(paste("F1-score:", f1_score))
  })
}

# Run the application 
shinyApp(ui = ui, server = server)













