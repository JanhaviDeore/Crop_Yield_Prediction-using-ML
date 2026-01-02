import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def evaluate_model_in_memory(model, X_test_transformed, y_test):
    """
    Evaluates a trained model (already in memory) on transformed test data.
    
    Args:
        model: The trained scikit-learn compatible model.
        X_test_transformed: The preprocessed (transformed) test features.
        y_test: The true target values for the test set.
        
    Returns:
        A dictionary containing RMSE, MAE, and R2 score.
    """
    # Predict on the already-transformed test data
    preds = model.predict(X_test_transformed)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def evaluate_model(model_path, X_test, y_test, preprocessor):
    """
    Loads a saved model from disk and evaluates it on raw test data.
    
    Args:
        model_path (str): Path to the saved .joblib model file.
        X_test: The raw, untransformed test features (pandas DataFrame).
        y_test: The true target values for the test set.
        preprocessor: The fitted ColumnTransformer object.
        
    Returns:
        A dictionary containing RMSE, MAE, and R2 score.
    """
    model = joblib.load(model_path)
    X_test_trans = preprocessor.transform(X_test)
    preds = model.predict(X_test_trans)

    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def create_synthetic_data(num_samples=1000):
    """
    Generates a synthetic DataFrame simulating crop yield data.
    
    Args:
        num_samples: The number of data points to generate.
        
    Returns:
        A pandas DataFrame.
    """
    np.random.seed(42) # for reproducibility
    data = {
        'temperature': np.random.uniform(10, 35, num_samples),
        'rainfall_mm': np.random.uniform(50, 500, num_samples),
        'soil_type': np.random.choice(['Loam', 'Clay', 'Sand'], num_samples),
        'fertilizer_type': np.random.choice(['Type A', 'Type B', 'Type C'], num_samples)
    }
    
    # Create a plausible target variable with some noise
    data['hg/ha_yield'] = (
        data['temperature'] * 2.5 + 
        data['rainfall_mm'] * 0.8 +
        np.random.normal(0, 25, num_samples) +
        [100 if s == 'Loam' else 50 if s == 'Clay' else 25 for s in data['soil_type']] +
        [75 if f == 'Type A' else 50 if f == 'Type B' else 25 for f in data['fertilizer_type']]
    )
    
    return pd.DataFrame(data)

def main():
    # 1. Load Data (using our synthetic function)
    df = create_synthetic_data()
    target_col = 'hg/ha_yield'
    
    # 2. Separate features (X) and target (y)
    y = df[target_col]
    X = df.drop(target_col, axis=1)
    
    # 3. Define Preprocessing
    num_cols = ['temperature', 'rainfall_mm']
    cat_cols = ['soil_type', 'fertilizer_type']
    
    # Create transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Create the column transformer (preprocessor)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='passthrough' # Keep any other columns unchanged
    )
    
    # 4. Split Data (BEFORE fitting the preprocessor)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Fit Preprocessor and Transform Data
    # Fit *only* on the training data
    X_train_trans = preprocessor.fit_transform(X_train)
    
    # Transform the test data using the *fitted* preprocessor
    X_test_trans = preprocessor.transform(X_test)
    
    # 6. Define and Train Models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, verbosity=0, use_label_encoder=False, eval_metric='rmse')
    }
    
    results = {}
    print("Training and evaluating models...")
    for name, model in models.items():
        # Fit on the TRANSFORMED training data
        model.fit(X_train_trans, y_train)
        
        # Evaluate on the TRANSFORMED test data using the in-memory evaluator
        metrics = evaluate_model_in_memory(model, X_test_trans, y_test)
        results[name] = metrics
        print(f"{name}: {metrics}")
        
    # 7. Find and Save Best Model
    best_model_name = min(results, key=lambda x: results[x]['rmse'])
    best_model = models[best_model_name]
    
    # Create directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the best model
    best_model_path = 'models/best_model.joblib'
    joblib.dump(best_model, best_model_path)
    
    # Save the FITTED preprocessor
    preprocessor_path = 'models/preprocessor.joblib'
    joblib.dump(preprocessor, preprocessor_path)
    
    print(f"\nBest model '{best_model_name}' saved to {best_model_path}")
    print(f"Fitted preprocessor saved to {preprocessor_path}")

    # 8. Evaluate the saved model from disk using the new function
    print("\nVerifying saved model performance by loading from disk...")
    # We can re-use the raw X_test and y_test, and the fitted 'preprocessor' object
    saved_model_metrics = evaluate_model(
        model_path=best_model_path,
        X_test=X_test, # Pass the raw X_test
        y_test=y_test,
        preprocessor=preprocessor # Pass the fitted preprocessor
    )
    print(f"Metrics from saved model: {saved_model_metrics}")

if __name__ == '__main__':
    main()

