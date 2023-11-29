# Bank Term Deposit Subscription Prediction
### Overview

This project develops a predictive model for XYZ Bank to identify potential clients who are likely to subscribe to a term deposit. The analysis leverages data from the bank's direct marketing campaigns, encompassing client demographics, campaign details, and economic indicators.

### Key Features

**Data Analysis:** Utilizes PySpark for handling large-scale data, ensuring efficient processing and analysis.

**Predictive Modeling:** Employs various machine learning models such as Logistic Regression, Random Forest, Gradient Boosted Trees, and Decision Tree Classifiers.

**Model Evaluation:** Focuses on metrics like AUC, accuracy, and F1 score to assess model performance.

**Customer Segmentation:** Explores customer segmentation using K-means clustering to identify distinct customer groups.

## Repository Contents

**Code:** Contains the complete Python code for data preprocessing, model training, evaluation, and customer segmentation (cluster analysis).

**Report:** A detailed analysis report is available, hosted as an HTML file on GitHub Pages. This report encompasses all phases of the project, from exploratory data analysis to model evaluation and conclusions. The report is a replacement for the Word document because we could make code blocks and scrollable elements in HTML instead of Word document along with other prettification objects such as an interactive table of contents. 

**Saved Model:** The best-performing model is saved by using the pyspark 'save' command which makes a folder of the saved model and the same can be found in the repository for further use or replication of the study.

## Model Insights

In the report of the project, multiple insights sections have been added after sections and subsections. A last word has also been added in the recommendation and conclusion section of the report. High-level insights into the project are as follows:

- The Gradient Boosted Trees model exhibited superior performance with the highest AUC, suggesting its robustness in predicting term deposit subscriptions.
- In-depth exploratory data analysis provided valuable insights into the factors influencing client decisions.
- Customer segmentation revealed distinct groups, aiding in targeted marketing strategies.

## Usage

The Python code can be run locally to replicate the study's findings. Requirements include PySpark and associated machine-learning libraries.

## Project Report

For a comprehensive understanding of the project, methodologies, and findings, please refer to the Project Report at: iimrraann.github.io/term_loan_website

## Contributions

This project is a collaborative effort for the BAN 5753 course, meticulously completed by our dedicated team. Our team members, each contributing their unique expertise and insights, include:

Haider, Muhammad Imran (muhaide@okstate.edu)

Joshi, Anant (anant.joshi@okstate.edu)

Kadiyala, Ruthvik (ruthvik.kadiyala@okstate.edu)

Litchfield, Ray Arthur (arthur.litchfield@okstate.edu)
