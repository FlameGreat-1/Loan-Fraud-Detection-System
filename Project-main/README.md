# Loan Fraud Detection System

## Overview

The Loan Fraud Detection System is a comprehensive tool designed to analyze loan data, detect potential fraud, and provide insights into loan repayment patterns. This project combines advanced data analysis techniques, machine learning algorithms, and a user-friendly graphical interface to assist in identifying and preventing loan fraud.

## Features

1. **Data Analysis and Visualization**
   - Upload and process CSV files containing loan data
   - Perform Exploratory Data Analysis (EDA)
   - Generate various visualizations including heatmaps, pie charts, and histograms

2. **Machine Learning Models**
   - Implement multiple classification algorithms:
     - Logistic Regression
     - Gaussian Naive Bayes
     - K-Nearest Neighbors
     - Support Vector Machine
     - Decision Tree
     - Random Forest
     - XGBoost
   - Perform SMOTE resampling for imbalanced datasets
   - Cross-validation and hyperparameter tuning

3. **Feature Importance Analysis**
   - Identify and visualize the most important features for fraud detection

4. **User-Friendly GUI**
   - Interactive Tkinter-based graphical user interface
   - Tabbed interface for easy navigation between different analysis stages

5. **Results and Reporting**
   - Generate detailed classification reports and confusion matrices
   - Export final predictions to CSV file

## Installation

1. Clone the repository:
   git clone https://github.com/FlameGreat-1/Loan-Fraud-Detection-System/edit/


2. Navigate to the project directory:
   cd loan-fraud-detection-system


3. Install the required packages:
   pip install -r requirements.txt


## Usage

1. Run the main script:
   python loan_fraud_detection_gui.py


2. Use the GUI to:
- Upload your CSV file containing loan data
- Process the data and view various analyses
- Train and evaluate machine learning models
- View feature importance and final predictions

## Dependencies

- Python 3.7+
- Tkinter
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (for SMOTE)
- YData Profiling (formerly Pandas Profiling)

## File Structure

- `loan_fraud_detection_gui.py`: Main script containing the GUI and analysis logic
- `requirements.txt`: List of required Python packages
- `README.md`: This file

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors and maintainers of the libraries used in this project
- Inspired by the need for better fraud detection in the financial sector
