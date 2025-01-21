# Decoding Beats: A Machine Learning Approach to Music Genre Classification

## Group Members:
- Mackiah C Henry
- Ze Wang
- Yongcen Zhou

### Overview
This project explores music genre classification using machine learning techniques. The goal is to classify music tracks into genres like Pop and Hip-Hop using the Free Music Archive (FMA) dataset.

---

## 1. Data Acquisition and Pre-processing

### Data Selection:
We used datasets from the Free Music Archive (FMA), accessed through the UCI Machine Learning Repository. We focused on a subset of 8000 tracks labeled as 'small', ensuring diverse genre representation.

**Main files used:**
- `tracks.csv`
- `echonest.csv`
- `genres.csv`

### Data Exploration:
We explored genre distributions, missing values, and the general structure of the data.

- **Genre Distribution:** Understanding the balance across musical genres.
- **Missing Values:** Identified missing data, particularly in geographical information like latitude and longitude.
- **Data Overview:** We used `summary()` and `head()` functions for an initial data overview.

### Data Pre-processing:
We cleaned and transformed the data by:
- Removing irrelevant columns and handling missing data.
- Removing near-zero variance predictors.
- Normalizing and standardizing numerical features for model stability.

### Feature Engineering:
- **Categorical Encoding:** One-hot encoding was used for categorical variables.
- **Temporal Features:** Temporal features from `echonest.csv` were removed after initial experimentation.

### Target Labels:
We chose 'Pop' and 'Hip-Hop' for classification based on clear distinctions observed in the genre distribution.

---

## 2. Model Building

### Algorithm Selection:
We selected three machine learning algorithms for genre classification:
- Random Forest
- Support Vector Machine (SVM)
- Decision Trees

### Model Training:
The dataset was split into an 80:20 training-to-testing ratio, with 80% used for training and 20% for testing. We trained the models to capture patterns and nuances in the dataset.

### Model Evaluation:
We used accuracy, precision, recall, and F1-score to evaluate each model. The results were as follows:

| Model           | Accuracy (%) | Recall (%) | Precision (%) | F1-Score (%) |
|-----------------|--------------|------------|---------------|--------------|
| Random Forest   | 99.8         | 99.6       | 100.0         | 99.8         |
| Decision Tree   | 97.0         | 98.3       | 95.7          | 97.0         |
| Support Vector Machine | 99.5 | 99.0       | 100.0         | 99.4         |

**Conclusion:** Random Forest performed the best, achieving an accuracy of 99.8% and precision of 100%.

---

## 3. Web Application Development with R-Shiny

### Shiny Application:
We integrated the trained models into an R Shiny application to allow interactive exploration of music genre classification results.

### App Features:
- **Data Upload:** Users can upload their own data and adjust the frequency.
- **Data Pre-processing:** Summary statistics and missing values visualization.
- **Genre Visualization:** Histograms and tables for understanding genre distributions.
- **Model Comparison:** Tabs for Random Forest, Decision Tree, and SVM to evaluate model accuracy and compare results.

### Challenges and Solutions:
- **R Shiny Learning Curve:** As I was unfamiliar with R Shiny, I referred to resources like the class template, Stack Overflow, and "Mastering Shiny" by Hadley Wickham to improve my skills.
- **Button Interactivity Issues:** Debugging small errors in function calls and syntax helped resolve issues with button responsiveness between frontend and backend.

---

## Conclusion
The project successfully demonstrated the capability of machine learning in music genre classification. The Random Forest algorithm achieved high accuracy, and the R Shiny application provided a user-friendly interface for interactive exploration.

This work enhances our technical skills and paves the way for future research in music classification.
