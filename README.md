# DataModelingWithTidyverserse
This repository is created to host the finale project of my specialization cours on the Tidyverse Framework on coursera by John Hopkins university (https://www.jhu.edu/)

# Consumer Complaint Classification Project

## 📋 Project Overview
This project focuses on classifying consumer complaint narratives into product categories using natural language processing and machine learning techniques. The implementation leverages the tidyverse ecosystem for data processing and tidymodels for building a Random Forest classifier.

## 🚀 Features
- **Text Preprocessing:** Tokenization, lemmatization, and stop word removal  
- **Feature Engineering:** TF-IDF transformation for text representation  
- **Machine Learning:** Random Forest classification with careful parameter tuning  
- **Model Evaluation:** Comprehensive metrics and confusion matrix analysis  
- **Reproducible Workflow:** Consistent data splitting and seed setting  

## 📊 Dataset
The project uses two datasets:
- `data/data_complaints_train.csv`: Training data with consumer narratives and product labels  
- `data/data_complaints_test.csv`: Testing data for model evaluation  

## 🛠️ Installation & Dependencies
To run this project, you'll need the following R packages:

```r
install.packages(c(
  "tidyverse", 
  "tidytext", 
  "tm", 
  "dtplyr", 
  "textstem",
  "tidymodels",
  "ranger"
))
```

## 📁 Project Structure
```
├── data/
│   ├── data_complaints_train.csv      # Original training data
│   ├── data_complaints_test.csv       # Original testing data
│   ├── data_token.rds                 # Processed token data
│   ├── tf_idf_data.rds                # TF-IDF transformed data
│   ├── tf_idf_data_long.rds           # Wide format TF-IDF data
│   └── final_training_data.rds        # Final training dataset
├── script.R                           # Main analysis script
└── README.md                          # Project documentation
```

## 🔧 Implementation Details

### Data Preprocessing
- Tokenization of complaint narratives  
- Filtering of non-English words, numbers, and overly long words  
- Lemmatization using `textstem::lemmatize_words()`  
- Stop word removal using the `stop_words` dataset  

### Feature Engineering
- TF-IDF (Term Frequency-Inverse Document Frequency) transformation  
- Selection of top 20 most frequent words as features  
- Conversion to wide format for machine learning  

### Modeling
Random Forest classifier with:
- 500 trees  
- `mtry = 5`  
- `min_n = 5`  
- 80/20 train-test split with stratification  
- Comprehensive evaluation using accuracy metrics and confusion matrix  

## 📈 Results
The model achieves classification performance as measured by:
- Accuracy metrics  
- Confusion matrix analysis  
- Variable importance from Random Forest  

## 🎯 How to Use
1. Clone the repository  
2. Ensure all data files are in the `data/` directory  
3. Run the `script.R` file to:  
   - Preprocess the data  
   - Train the model  
   - Evaluate performance  
   - Generate predictions  

## 🔮 Future Improvements
- Experiment with different feature selection methods  
- Try alternative classification algorithms (e.g., XGBoost, SVM)  
- Implement deep learning approaches for text classification  
- Add hyperparameter tuning with cross-validation  
- Develop a Shiny app for interactive predictions  

## 📚 References
- Silge, J., & Robinson, D. (2017). *Text Mining with R: A Tidy Approach*. O'Reilly Media.  
- Kuhn, M., & Wickham, H. (2020). *Tidymodels: a collection of packages for modeling and machine learning using tidyverse principles*.  

## 👨‍💻 Author
This project was completed as part of the "Data Modeling with Tidyverse" specialization course.  

**Note:** This project demonstrates practical application of tidyverse principles for text classification tasks in a real-world scenario.
