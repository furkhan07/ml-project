"""
Preprocessing helper for Data_Train.csv
- Replaces Route tokens: BLR -> Banglore, DEL -> Delhi (case-insensitive)
- Drops unwanted columns and NA rows
- Builds a ColumnTransformer + Pipeline and evaluates a regressor
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def preprocess_and_train(csv_path='Data_Train.csv'):
    df = pd.read_csv(csv_path)

    # Basic cleaning
    df = df.dropna()
    for c in ['Date_of_Journey', 'Additional_Info']:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Replace Route tokens (case-insensitive)
    if 'Route' in df.columns:
        df['Route'] = df['Route'].astype(str)
        df['Route'] = df['Route'].str.replace(r"\\bBLR\\b", 'Banglore', regex=True, flags=re.IGNORECASE)
        df['Route'] = df['Route'].str.replace(r"\\bDEL\\b", 'Delhi', regex=True, flags=re.IGNORECASE)

        print('Unique Route values after replacement:')
        print(df['Route'].unique()[:50])

    # Split features/target
    if 'Price' not in df.columns:
        raise KeyError('Expected target column "Price" not found in dataframe')

    X = df.drop('Price', axis=1)
    y = df['Price']

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()

    print(f'Numeric cols: {num_cols}')
    print(f'Categorical cols: {cat_cols}')

    # Preprocessing transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), cat_cols),
        ],
        remainder='drop'
    )

    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    pipeline.fit(X_train, y_train)

    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)

    y_pred_test = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)

    print(f'Train R^2: {train_score:.4f}')
    print(f'Test R^2:  {test_score:.4f} (sklearn score)')
    print(f'Test R^2 (r2_score): {r2:.4f}')

    # Return pipeline and processed dataframe sample
    return pipeline, df


if __name__ == '__main__':
    pipeline, df = preprocess_and_train()


                