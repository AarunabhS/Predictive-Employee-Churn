# Employee Churn Prediction

## Overview

The project focuses on predicting employee churn, which refers to the rate at which employees leave a company within a certain period. This is a significant issue for companies as it incurs costs related to recruitment, training, and productivity loss. The project uses HR data consisting of 14,999 samples to study and predict employee churn. The data includes information about employees who stayed and those who left the company.

## Features

- **satisfaction_level**: Employee satisfaction score ranging from 0-1.
- **last_evaluation**: Performance evaluation score given by the employer, ranging from 0-1.
- **number_projects**: Number of projects assigned to an employee.
- **average_monthly_hours**: Average hours worked per month by an employee.
- **time_spent_company**: Number of years an employee has spent in the company.
- **work_accident**: Indicator of whether an employee has had a work accident.
- **promotion_last_5years**: Indicator of whether an employee was promoted in the last five years.
- **Departments**: Employee's working department.
- **Salary**: Salary level of the employee.
- **left**: Indicator of whether the employee has left the company.

## Data Analysis and Modelling

The project includes detailed analysis and visualization of the dataset, identifying key patterns and correlations among features. Clustering techniques were applied to categorize employees into different groups based on their characteristics and behaviors.

### Key Findings

1. **Satisfaction Level**: Employees with low satisfaction levels are more likely to leave. There's a notable churn among those with satisfaction levels around 0.1 and between 3.5 and 4.5.
2. **Workload**: The number of projects and average monthly hours worked are significant indicators of churn. Employees with high workloads or too few projects are more likely to leave.
3. **Cluster Analysis**: Identified clusters such as 'Stars', 'Slackers', 'Workaholics', 'Just-a-job', and 'Overworked' based on work hours and evaluation scores.

### Modelling Techniques

The dataset was split into training and testing sets (70:30 ratio). Various machine learning algorithms were used, including:

- **Gradient Boosting Classifier**
- **Logistic Regression**
- **Random Forest Classifier**
- **Deep Learning Neural Networks**

### Model Performance

- **Gradient Boosting Classifier** emerged as the best-performing model, outperforming others, including neural networks, which required a larger dataset for optimal performance.

## Conclusion

The Gradient Boosting Classifier was identified as the best model for predicting employee churn. The project highlights the importance of addressing employee satisfaction and workload to reduce turnover. Future work includes finding a more comprehensive dataset and employing advanced models for improved predictions.
