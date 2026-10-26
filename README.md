Housing Value Prediction Tool

Project Overview
The goal of this project is to predict residential property prices in North Carolina using machine learning models. The project combines individual housing data with regional and economic data to understand and explore the factors associated with house sale prices.

We use Linear Regression and Random Forest models to compare performance.

Project Status: In progress. Data Collection, cleaning and processing, and initial model training have been completed.

Data Sources
1. North Carolina Housing Dataset: Property characteristics (bedrooms, bathrooms, square footage, year built, location, and historical prices).
2. U.S. Census Bureau: County-level median income, population, rent, and housing units.
3. Federal Housing Finance Agency(FHFA): County_level House Price Index and annual housing price changes.
4. Federal Reserve Economic Data(FRED): Historical 30-year mortgage interest rates.

Installation Requirements
1. Python 3.11 or later
2. pandas
3. numpy
4. scikit-learn
5. requests
6. openpyxl
7. joblib

Setup
Clone the repository:
git clone YOUR_GITHUB_REPOSITORY_URL
cd DTSC-3601-2-Project

Create and activate a virtual environment:
python3 -m venv .venv
source .venv/bin/activate

Install the required libraries:
pip install -r requirements.txt

Getting Started
The project requires a processed North Carolina housing dataset
Once the required file is available at 
data/processed/nc_housing_combined_2024.csv, run:
python3 src/train_nc_combined_models.py

The script then trains and evaluates four models:
1. Baseline Linear Regression
2. Baseline Random Forest
3. Enhanced Linear Regression
4. Enhanced Random Forest

The baseline models use property related variables and location. While the enhanced models include Census, FHFA, and FRED information

The trained models are saved in the models/ directory, and evaluation results are saved in data/processed/nc_model_comparison.csv.

Initial Model Results
Baseline Linear Regression:
MAE: $151,694    R^2: 0.4263
BaseLine Random Forest: 
MAE: $88,779     R^2: 0.6758
Enhanced Linear Regression:
MAE: $142,806    R^2: 0.4660
Enhanced Random Forest: 
MAE: $87,965     R^2: 0.6814

The Enhanced Random Forest models performed better in this initial evaluation.

The results are not yet completed. Some annual economic features were calculated using information from the full year, making the evaluation uncompelete and non-leakage free.

Repository Structure
DTSC-3601-2-Project/
README.md
LICENSE
requirements.txt
scr/: Scripts for data collection, cleaning, and model training
bin/: Example code to run the project
docs/: Additional documentation
data/: Raw and processed dtatsets
models/: Machine learning models

Documentation
Detailed explanations for the major function will be provided in docs/function.med

License
This project carries an MIT license.

Contact
For further information or question about this project, please contact:
Happy Crabs group
Gabriel Tewell, Medha Modini, Assmaa Hassan
