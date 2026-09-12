# Apache Spark Restaurant Revenue Forecasting

A machine-learning project for predicting restaurant revenue using **Java, Apache Spark MLlib, and MongoDB**.

The project loads restaurant data from MongoDB, performs data cleaning and feature engineering, trains multiple regression models, compares their performance, and writes prediction results back to MongoDB.

## Main Features

- MongoDB data ingestion with Apache Spark
- Data cleaning and filtering
- Numerical and categorical feature processing
- Feature engineering
- Linear Regression
- Random Forest Regression
- Gradient-Boosted Trees (GBT)
- One-Hot Encoding
- String Indexing
- Train / validation splitting
- Basic hyperparameter tuning
- Model comparison
- Prediction storage in MongoDB

## Models

The project evaluates three regression approaches:

### Linear Regression

Uses numerical restaurant features such as:

- Rating
- Marketing Budget
- Number of Reviews
- Average Meal Price
- Ambience Score
- Chef Experience Years

### Random Forest Regression

Uses the same structured numerical features to capture non-linear relationships between restaurant characteristics and revenue.

### Gradient-Boosted Trees

The GBT pipeline includes additional preprocessing and feature engineering:

- Cuisine
- Location
- Parking Availability
- Service Quality Score
- Social Media Followers
- Rating × Marketing Budget
- Ambience Score × Chef Experience Years
- Log-transformed revenue

Categorical variables are processed with:

```text
StringIndexer
      ↓
OneHotEncoder
      ↓
VectorAssembler
      ↓
GBTRegressor
```

## Architecture

```text
MongoDB
   |
   v
Apache Spark DataFrame
   |
   v
Data Cleaning
   |
   v
Feature Engineering
   |
   +-----------------------------+
   |              |              |
   v              v              v
Linear       Random Forest      GBT
Regression     Regression     Regression
   |              |              |
   +--------------+--------------+
                  |
                  v
          Model Evaluation
                  |
                  v
        Prediction Results
                  |
                  v
              MongoDB
```

## Model Evaluation

The models are evaluated using:

- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error
- **R² Score**
- Estimated prediction accuracy derived from MAPE

The project also includes a model-comparison runner that prints a summary table for:

```text
Linear Regression
Random Forest
GBT Optimized
```

## Technologies

- Java
- Apache Spark
- Spark SQL
- Spark DataFrames
- Spark MLlib
- MongoDB
- MongoDB Spark Connector
- Linear Regression
- Random Forest
- Gradient-Boosted Trees
- Feature Engineering
- Machine Learning
- Regression Analysis

## Project Components

### `MongoSparkLoader.java`

Implements the baseline pipeline:

- Reads restaurant data from MongoDB
- Builds numerical feature vectors
- Trains a Linear Regression model
- Evaluates predictions
- Saves predictions to MongoDB

### `ModelComparisonRunner.java`

Compares:

- Linear Regression
- Random Forest Regression
- Gradient-Boosted Trees

It evaluates each model using RMSE, MAPE, estimated accuracy, and R².

### `GBTWithFeatures.java`

Builds a more advanced GBT pipeline using:

- Categorical features
- One-hot encoding
- Additional numerical features
- Train / test splitting
- TrainValidationSplit
- Basic hyperparameter tuning

### `GBTOptimizedFinal.java`

Implements an optimized GBT pipeline with:

- Engineered interaction features
- Log-transformed revenue
- Train / validation splitting
- RMSE evaluation
- MAPE evaluation
- R² evaluation
- MongoDB prediction output

## Example Data Flow

```text
Restaurant Data
      |
      v
MongoDB
      |
      v
Apache Spark
      |
      v
Preprocessing
      |
      v
Feature Engineering
      |
      v
Regression Models
      |
      v
Evaluation
      |
      v
Predicted Revenue
```

## Example Features

The project uses restaurant attributes such as:

```text
Rating
Marketing Budget
Number of Reviews
Average Meal Price
Ambience Score
Chef Experience Years
Service Quality Score
Social Media Followers
Cuisine
Location
Parking Availability
```

## Output

Depending on the pipeline, predictions are either:

- displayed in the console,
- written to CSV,
- or stored in MongoDB collections such as:

```text
predictions_linear
predictions_gbt
```

## Goal

The goal of the project is to explore how distributed data processing and machine-learning models can be used together to predict restaurant revenue and compare different regression approaches.

## Author

**Abhinaba Dutta**

MSc in Data, Algorithms and Machine Intelligence  
University of Palermo, Italy
