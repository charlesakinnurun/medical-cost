# %% [markdown]
# Import the neccesary libraries

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,r2_score

# %% [markdown]
# Data Loading

# %%
df = pd.read_csv("insurance.csv")
print("Data loaded sucessfully")
print("First 5 rows of the data")
print(df.head())

# %% [markdown]
# Data Preprocessing and Feature Engineering

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("Missing Values")
print(df_missing)

# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("Duplicated Rows")
print(df_duplicated)

print("Number of rows before dropping")
print(len(df))

# Drop duplicated rows
df.drop_duplicates(inplace=True)

print("Number of rows after dropping")
print(len(df))

# Define the features (X) and the target variable (y)
X = df.drop("charges",axis=1) # Features are all columns except "charges"
y = df["charges"] # The target variable is "charges"

# Identify the types of features
categorical_features = ["sex","smoker","region"]
numerical_features = ["age","bmi","children"]

# Create a preprocessor using ColumnTransformer. This applies different transformations
# to different columns in a single step
preprocessor = ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),numerical_features),
        ("cat",OneHotEncoder(),categorical_features)
    ]
)

# %% [markdown]
# Data Spilting

# %%
# We split the data into training and testing sets to evaluate the model's performance on the unseen data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %% [markdown]
# Model Training

# %%
# We create a pipeline to cahin preprocessing and model training together for the linear model
linear_model_pipeline = Pipeline(steps=[("preprocessor",preprocessor),
                                        ("regressor",LinearRegression())])

# Train the model on the training data
linear_model_pipeline.fit(X_train,y_train)

# Make predictions on the test set
y_pred_linear = linear_model_pipeline.predict(X_test)

# Evaluate the linear model's performance using MAE and R-squared
mae_linear = mean_absolute_error(y_test,y_pred_linear)
r2_linear = r2_score(y_test,y_pred_linear)
print(mae_linear)
print(r2_linear)

# %%
# Display model performance results
print("Linear Regression Model Performance:")
print("=" * 40)
print(f"Mean Absolute Error: ${mae_linear:,.2f}")
print(f"R-squared Score: {r2_linear:.4f}")
print(f"R-squared Percentage: {r2_linear*100:.2f}%")

# Interpretation
print("\nModel Interpretation:")
print("-" * 20)
if r2_linear > 0.8:
    print("✓ Excellent model performance!")
elif r2_linear > 0.6:
    print("✓ Good model performance")
elif r2_linear > 0.4:
    print("⚠ Moderate model performance")
else:
    print("⚠ Poor model performance - consider feature engineering or different algorithms")


# %%
# Additional Analysis and Visualization
import matplotlib.pyplot as plt

# Create a comparison plot of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Charges ($)')
plt.ylabel('Predicted Charges ($)')
plt.title('Actual vs Predicted Medical Charges')
plt.grid(True, alpha=0.3)
plt.show()

# Calculate and display residuals
residuals = y_test - y_pred_linear
print(f"\nResidual Analysis:")
print(f"Mean of residuals: ${residuals.mean():,.2f}")
print(f"Standard deviation of residuals: ${residuals.std():,.2f}")

# Feature importance (coefficients)
feature_names = numerical_features + list(linear_model_pipeline.named_steps['preprocessor']
                                        .transformers_[1][1].get_feature_names_out(categorical_features))
coefficients = linear_model_pipeline.named_steps['regressor'].coef_

print(f"\nFeature Importance (Coefficients):")
print("-" * 35)
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature:20}: {coef:8.2f}")


# %%
# Example Predictions
print("Example Predictions:")
print("=" * 50)

# Create some example cases
examples = [
    {"age": 30, "sex": "male", "bmi": 25.5, "children": 2, "smoker": "no", "region": "northeast"},
    {"age": 45, "sex": "female", "bmi": 32.0, "children": 0, "smoker": "yes", "region": "southwest"},
    {"age": 25, "sex": "female", "bmi": 22.0, "children": 1, "smoker": "no", "region": "northwest"}
]

for i, example in enumerate(examples, 1):
    # Convert to DataFrame for prediction
    example_df = pd.DataFrame([example])
    prediction = linear_model_pipeline.predict(example_df)[0]
    
    print(f"\nExample {i}:")
    print(f"  Age: {example['age']}, Sex: {example['sex']}, BMI: {example['bmi']}")
    print(f"  Children: {example['children']}, Smoker: {example['smoker']}, Region: {example['region']}")
    print(f"  Predicted Medical Charges: ${prediction:,.2f}")



