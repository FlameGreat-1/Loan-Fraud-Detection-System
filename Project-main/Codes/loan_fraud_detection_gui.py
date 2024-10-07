import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import ydata_profiling as pf
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import warnings
import io

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

class LoanFraudDetectionGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Loan Fraud Detection System")
        self.master.geometry("1200x800")
        self.master.configure(bg='#f0f0f0')
        self.df = None
        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(self.master, text="Loan Fraud Detection System", font=("Arial", 20, "bold"), bg='#f0f0f0')
        title_label.pack(pady=20)

        self.upload_button = tk.Button(self.master, text="Upload CSV File", command=self.upload_file, font=("Arial", 12), bg='#4CAF50', fg='white')
        self.upload_button.pack(pady=10)

        self.file_label = tk.Label(self.master, text="No file selected", font=("Arial", 10), bg='#f0f0f0')
        self.file_label.pack()

        self.process_button = tk.Button(self.master, text="Process Data", command=self.process_data, font=("Arial", 12), bg='#008CBA', fg='white', state=tk.DISABLED)
        self.process_button.pack(pady=10)

        self.create_tabs()

    def create_tabs(self):
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        self.tabs = {
            "Data Overview": ScrollableFrame(self.notebook),
            "Data Cleaning": ScrollableFrame(self.notebook),
            "EDA": ScrollableFrame(self.notebook),
            "Visualizations": ScrollableFrame(self.notebook),
            "SMOTE Resampling": ScrollableFrame(self.notebook),
            "Model Results": ScrollableFrame(self.notebook),
            "Cross-Validation": ScrollableFrame(self.notebook),
            "Hyperparameter Tuning": ScrollableFrame(self.notebook),
            "Feature Importance": ScrollableFrame(self.notebook),
            "Final Predictions": ScrollableFrame(self.notebook)
        }

        for tab_name, tab in self.tabs.items():
            self.notebook.add(tab, text=tab_name)

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=f"Selected file: {file_path.split('/')[-1]}")
            self.process_button.config(state=tk.NORMAL)

    def process_data(self):
        if self.df is None:
            try:
                self.df = pd.read_csv(self.file_path)
                self.display_data_overview()
                self.perform_data_cleaning()
                self.perform_eda()
                self.create_visualizations()
                self.perform_smote_resampling()
                self.train_models()
                self.perform_cross_validation()
                self.perform_hyperparameter_tuning()
                self.display_feature_importance()
                self.display_final_predictions()
                messagebox.showinfo("Success", "Data processing completed successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def display_data_overview(self):
        try:
            tab = self.tabs["Data Overview"].scrollable_frame
            
            tk.Label(tab, text="First Five Rows:", font=("Arial", 12, "bold")).pack(anchor="w")
            text_widget = tk.Text(tab, height=10, width=100)
            text_widget.pack(pady=10)
            text_widget.insert(tk.END, self.df.head().to_string())

            tk.Label(tab, text=f"Shape: {self.df.shape}", font=("Arial", 12)).pack(anchor="w")

            tk.Label(tab, text="Null Values:", font=("Arial", 12, "bold")).pack(anchor="w")
            text_widget = tk.Text(tab, height=10, width=100)
            text_widget.pack(pady=10)
            text_widget.insert(tk.END, self.df.isnull().sum().to_string())

            tk.Label(tab, text="DataFrame Info:", font=("Arial", 12, "bold")).pack(anchor="w")
            text_widget = tk.Text(tab, height=10, width=100)
            text_widget.pack(pady=10)
            buffer = io.StringIO()
            self.df.info(buf=buffer)
            text_widget.insert(tk.END, buffer.getvalue())

            tk.Label(tab, text="Descriptive Statistics:", font=("Arial", 12, "bold")).pack(anchor="w")
            text_widget = tk.Text(tab, height=10, width=100)
            text_widget.pack(pady=10)
            text_widget.insert(tk.END, self.df.describe().T.to_string())
        except Exception as e:
            print(f"Error in display_data_overview: {str(e)}")

    def perform_data_cleaning(self):
        try:
            tab = self.tabs["Data Cleaning"].scrollable_frame

            # Handle missing values
            tk.Label(tab, text="Handling Missing Values:", font=("Arial", 12, "bold")).pack(anchor="w")
            text_widget = tk.Text(tab, height=5, width=100)
            text_widget.pack(pady=10)
            missing_before = self.df.isnull().sum()
            self.df = self.df.dropna()  # or use appropriate imputation method
            missing_after = self.df.isnull().sum()
            text_widget.insert(tk.END, f"Missing values before: {missing_before.sum()}\n")
            text_widget.insert(tk.END, f"Missing values after: {missing_after.sum()}\n")

            # Handle duplicate rows
            tk.Label(tab, text="Handling Duplicate Rows:", font=("Arial", 12, "bold")).pack(anchor="w")
            text_widget = tk.Text(tab, height=5, width=100)
            text_widget.pack(pady=10)
            duplicates_before = self.df.duplicated().sum()
            self.df = self.df.drop_duplicates()
            duplicates_after = self.df.duplicated().sum()
            text_widget.insert(tk.END, f"Duplicate rows before: {duplicates_before}\n")
            text_widget.insert(tk.END, f"Duplicate rows after: {duplicates_after}\n")

            # Display cleaned data info
            tk.Label(tab, text="Cleaned Data Info:", font=("Arial", 12, "bold")).pack(anchor="w")
            text_widget = tk.Text(tab, height=10, width=100)
            text_widget.pack(pady=10)
            buffer = io.StringIO()
            self.df.info(buf=buffer)
            text_widget.insert(tk.END, buffer.getvalue())

        except Exception as e:
            print(f"Error in perform_data_cleaning: {str(e)}")

    def perform_eda(self):
        try:
            tab = self.tabs["EDA"].scrollable_frame
            
            profile = pf.ProfileReport(self.df)
            profile.to_file("EDA_Report.html")
            
            tk.Label(tab, text="EDA Report generated and saved as 'EDA_Report.html'", font=("Arial", 12)).pack(pady=20)

            tk.Label(tab, text="Correlation Matrix:", font=("Arial", 12, "bold")).pack(anchor="w")
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(self.df.corr(), ax=ax, cmap='coolwarm', annot=True)
            self.display_figure(fig, tab)
        except Exception as e:
            print(f"Error in perform_eda: {str(e)}")

    def create_visualizations(self):
        try:
            tab = self.tabs["Visualizations"].scrollable_frame

            visualizations = [
                self.plot_loan_purpose_repayment,
                self.plot_credit_policy_fico,
                self.plot_loan_repayment_fico,
                self.plot_loan_purpose_pie,
                self.plot_loan_repayment_pie,
                self.plot_correlation_heatmap
            ]

            for viz_func in visualizations:
                fig = viz_func()
                self.display_figure(fig, tab)
        except Exception as e:
            print(f"Error in create_visualizations: {str(e)}")

    def display_figure(self, fig, tab):
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
        canvas.get_tk_widget().pack(pady=10)

    def plot_loan_purpose_repayment(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(x='purpose', hue='not.fully.paid', data=self.df, ax=ax)
        ax.set_title('Count of Customer based on Loan Purpose and Repayment')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.tight_layout()
        return fig

    def plot_credit_policy_fico(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.df[self.df['credit.policy'] == 1]['fico'].hist(ax=ax1, bins=30, alpha=0.5)
        ax1.set_title('Distribution of Credit Policy [1] & FICO')
        self.df[self.df['credit.policy'] == 0]['fico'].hist(ax=ax2, bins=30, alpha=0.5)
        ax2.set_title('Distribution of Credit Policy [0] & FICO')
        plt.tight_layout()
        return fig

    def plot_loan_repayment_fico(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.df[self.df['not.fully.paid'] == 1]['fico'].hist(ax=ax1, bins=30, alpha=0.5)
        ax1.set_title('Distribution of Not Fully Paid & FICO')
        self.df[self.df['not.fully.paid'] == 0]['fico'].hist(ax=ax2, bins=30, alpha=0.5)
        ax2.set_title('Distribution of Fully Paid & FICO')
        plt.tight_layout()
        return fig

    def plot_loan_purpose_pie(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        self.df['purpose'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_title('Pie Representation on Percentage of Loan Purpose')
        return fig

    def plot_loan_repayment_pie(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        self.df['not.fully.paid'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, labels=['Paid', 'Not Paid'])
        ax.set_title('Proportion of Customer with Paid and Not Paid')
        return fig

    def plot_correlation_heatmap(self):
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(self.df.corr(), ax=ax, cmap='coolwarm', annot=True)
        ax.set_title('Correlation Heat Map')
        return fig

    def perform_smote_resampling(self):
        try:
            tab = self.tabs["SMOTE Resampling"].scrollable_frame

            # Prepare data
            df1 = pd.get_dummies(self.df, columns=['purpose'], drop_first=True)
            X = df1.drop(['not.fully.paid'], axis=1)
            y = df1['not.fully.paid']

            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Display results
            tk.Label(tab, text="SMOTE Resampling Results:", font=("Arial", 12, "bold")).pack(anchor="w")
            text_widget = tk.Text(tab, height=10, width=100)
            text_widget.pack(pady=10)
            text_widget.insert(tk.END, f"Original dataset shape: {X.shape}\n")
            text_widget.insert(tk.END, f"Resampled dataset shape: {X_resampled.shape}\n")
            text_widget.insert(tk.END, f"Original class distribution:\n{y.value_counts()}\n")
            text_widget.insert(tk.END, f"Resampled class distribution:\n{pd.Series(y_resampled).value_counts()}\n")

            # Visualize class distribution before and after SMOTE
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            y.value_counts().plot(kind='bar', ax=ax1, title='Before SMOTE')
            pd.Series(y_resampled).value_counts().plot(kind='bar', ax=ax2, title='After SMOTE')
            plt.tight_layout()
            self.display_figure(fig, tab)

            # Store resampled data for further processing
            self.X_resampled, self.y_resampled = X_resampled, y_resampled

        except Exception as e:
            print(f"Error in perform_smote_resampling: {str(e)}")

    def train_models(self):
        try:
            tab = self.tabs["Model Results"].scrollable_frame

            # Prepare data
            X_train, X_test, y_train, y_test = train_test_split(self.X_resampled, self.y_resampled, test_size=0.3, random_state=42)

            # Define models
            models = {
                'Logistic Regression': LogisticRegression(),
                'Gaussian Naive Bayes': GaussianNB(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Support Vector Machine': SVC(),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'XGBoost': XGBClassifier()
            }

            # Train and evaluate models
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                tk.Label(tab, text=f"{name} Results:", font=("Arial", 12, "bold")).pack(anchor="w")
                text_widget = tk.Text(tab, height=10, width=100)
                text_widget.pack(pady=10)
                text_widget.insert(tk.END, f"Accuracy: {accuracy:.2%}\n\n")
                text_widget.insert(tk.END, f"Classification Report:\n{classification_report(y_test, y_pred)}\n")

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(f'Confusion Matrix - {name}')
                self.display_figure(fig, tab)

        except Exception as e:
            print(f"Error in train_models: {str(e)}")

    def perform_cross_validation(self):
        try:
            tab = self.tabs["Cross-Validation"].scrollable_frame

            # Prepare data
            X, y = self.X_resampled, self.y_resampled

            # Define models
            models = {
                'Logistic Regression': LogisticRegression(),
                'Gaussian Naive Bayes': GaussianNB(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Support Vector Machine': SVC(),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'XGBoost': XGBClassifier()
            }

            # Perform cross-validation
            for name, model in models.items():
                scores = cross_val_score(model, X, y, cv=5)
                tk.Label(tab, text=f"{name} Cross-Validation Results:", font=("Arial", 12, "bold")).pack(anchor="w")
                text_widget = tk.Text(tab, height=5, width=100)
                text_widget.pack(pady=10)
                text_widget.insert(tk.END, f"Mean Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})\n")

        except Exception as e:
            print(f"Error in perform_cross_validation: {str(e)}")

    def perform_hyperparameter_tuning(self):
        try:
            tab = self.tabs["Hyperparameter Tuning"].scrollable_frame

            # Prepare data
            X, y = self.X_resampled, self.y_resampled

            # Define model and parameters for tuning (example with Random Forest)
            model = RandomForestClassifier()
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10]
            }

            # Perform GridSearchCV
            grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
            grid_search.fit(X, y)

            # Display results
            tk.Label(tab, text="Hyperparameter Tuning Results (Random Forest):", font=("Arial", 12, "bold")).pack(anchor="w")
            text_widget = tk.Text(tab, height=10, width=100)
            text_widget.pack(pady=10)
            text_widget.insert(tk.END, f"Best parameters: {grid_search.best_params_}\n")
            text_widget.insert(tk.END, f"Best cross-validation score: {grid_search.best_score_:.2f}\n\n")
            text_widget.insert(tk.END, "All results:\n")
            for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
                text_widget.insert(tk.END, f"{params}: {mean_score:.2f}\n")

        except Exception as e:
            print(f"Error in perform_hyperparameter_tuning: {str(e)}")

    def display_feature_importance(self):
        try:
            tab = self.tabs["Feature Importance"].scrollable_frame

            # Prepare data
            X, y = self.X_resampled, self.y_resampled

            # Train Random Forest model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

            # Get feature importance
            feature_importance = model.feature_importances_
            feature_names = X.columns
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

            # Display feature importance
            tk.Label(tab, text="Feature Importance:", font=("Arial", 12, "bold")).pack(anchor="w")
            text_widget = tk.Text(tab, height=10, width=80)
            text_widget.pack(pady=10)
            text_widget.insert(tk.END, feature_importance_df.to_string(index=False))

            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            feature_importance_df.plot(x='feature', y='importance', kind='bar', ax=ax)
            ax.set_title('Feature Importance')
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            plt.xticks(rotation=90)
            plt.tight_layout()
            self.display_figure(fig, tab)

        except Exception as e:
            print(f"Error in display_feature_importance: {str(e)}")

    def display_final_predictions(self):
        try:
            tab = self.tabs["Final Predictions"].scrollable_frame

            # Prepare data
            X, y = self.X_resampled, self.y_resampled

            # Train final model (using Random Forest with best parameters from hyperparameter tuning)
            model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2, random_state=42)
            model.fit(X, y)

            # Make predictions on original dataset
            original_X = pd.get_dummies(self.df.drop(['not.fully.paid'], axis=1), columns=['purpose'], drop_first=True)
            final_predictions = model.predict(original_X)

            # Create final dataframe
            final_df = self.df.copy()
            final_df['Predicted_Loan_Status'] = final_predictions
            final_df['Predicted_Loan_Status'] = final_df['Predicted_Loan_Status'].map({0: 'Paid', 1: 'Not Paid'})
            final_df['Actual_Loan_Status'] = final_df['not.fully.paid'].map({0: 'Paid', 1: 'Not Paid'})

            # Display final predictions
            tk.Label(tab, text="Final Predictions (First 10 rows):", font=("Arial", 12, "bold")).pack(anchor="w")
            text_widget = tk.Text(tab, height=20, width=100)
            text_widget.pack(pady=10)
            text_widget.insert(tk.END, final_df[['Actual_Loan_Status', 'Predicted_Loan_Status']].head(10).to_string())

            # Save predictions
            final_df.to_csv('Loan_Status_Prediction.csv', index=False)
            tk.Label(tab, text="Predictions saved to 'Loan_Status_Prediction.csv'", font=("Arial", 12)).pack(pady=10)

            # Display model performance on original dataset
            accuracy = accuracy_score(final_df['not.fully.paid'], final_predictions)
            tk.Label(tab, text=f"Model Accuracy on Original Dataset: {accuracy:.2%}", font=("Arial", 12, "bold")).pack(pady=10)

            # Confusion matrix
            cm = confusion_matrix(final_df['not.fully.paid'], final_predictions)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix - Final Predictions')
            self.display_figure(fig, tab)

        except Exception as e:
            print(f"Error in display_final_predictions: {str(e)}")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = LoanFraudDetectionGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

