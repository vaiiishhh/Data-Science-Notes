# DataSci Notes

## Core Libraries & Data Structures

### Essential Libraries
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `plotly.express`, `matplotlib`, `seaborn`
- **Machine Learning**: `sklearn.models`, `sklearn.metrics`, `sklearn.impute`, `sklearn.pipeline`
- **Database**: `pymongo`, `sqlite3`
- **Time Series**: `statsmodels.graphics.tsaplots`, `statsmodels.tsa.ar_model`, `statsmodels.tsa.arima.model`
- **Utility**: `gzip`, `json`, `pickle`, `glob`
- **Sampling**: `imblearn.over_sampling`, `imblearn.under_sampling`
- **Ensemble Methods**: `sklearn.ensemble`
- **Clustering**: `sklearn.cluster.KMeans`

### Core Data Structures
- **Lists & Dictionaries**: Basic Python structures for data organization
- **Pandas DataFrame**: Primary structure for tabular data manipulation
- **CSV Import**: Using `csv` module and `glob` for multiple files

---

## Data Preparation & Cleaning

### Data Cleaning Essentials
- **Null Values**: Identify and handle missing data appropriately
- **Symbol Removal**: Clean unwanted characters from data
- **Data Type Conversion**: Use `astype()` to convert data types
- **Outlier Detection**: Remove observations outside [0.1-0.9] quantile range

### Exploratory Data Analysis (EDA)
- **Scatter Mapbox**: Geographic visualization of data points
- **Value Counts**: Frequency distribution analysis
- **`.describe()`**: Statistical summary of numerical features
- **Mean vs Median**: Understanding data distribution (mean > median indicates right skew)
- **Visualizations**:
  - **Histogram**: Frequency distribution visualization
  - **Box Plot**: Visual representation of `.describe()` statistics
  - **Time Series Plot**: Line plots for temporal data

### Data Quality Considerations
- **High vs Low Cardinality**: For categorical features, consider top 10 categories and bucket remaining together
- **Data Leakage**: Ensure future information doesn't leak into training data
- **Multicollinearity**: Check for highly correlated features
- **Feature Scaling**: Normalize features when necessary

---

## Machine Learning Workflow

### The Five-Step ML Process
1. **Split**: Divide data vertically (features vs target) and horizontally (train vs test)
2. **Baseline**: Establish minimum performance benchmark
3. **Iterate**: Build, train, and refine models
4. **Evaluate**: Assess model performance on test data
5. **Communicate**: Present results with visualizations and interpretations

### Data Splitting Strategy
- **Vertical Split**: Separate features (X) from target variable (y)
- **Horizontal Split**: Divide into training and testing sets
- **Validation Split**: Further split training data for model validation
- **Reproducibility**: Use consistent random seed values

### Baseline Models
- **Regression Baseline**: Predict the mean for all observations
- **Performance Metric**: Mean Absolute Error (MAE) - average error across observations
- **Visualization**: Plot mean baseline against feature relationships

---

## Linear Regression & Preprocessing

### Model Building Process
1. **Model Initialization**: `LinearRegression()`
2. **Training**: `model.fit(X_train, y_train)`
3. **Prediction**: `model.predict(X_train)` for training evaluation
4. **Evaluation**: Compare training MAE to baseline MAE

### Data Preprocessing Pipeline
- **Imputation**: Handle missing values using transformers like `SimpleImputer`
- **Encoding**: Convert categorical variables using `OneHotEncoder`
- **Pipeline Structure**: Chain preprocessing steps for consistent application
- **Best Practice**: Apply same transformations to training and testing data

### Model Interpretation
- **Linear Equation**: `target = intercept + coefficient * feature`
- **Multiple Features**: Display top 10-15 feature coefficients in bar chart
- **Deployment Options**:
  1. Wrap model in function for programmatic use
  2. Create interactive dashboard using Jupyter widgets

### Regularization
- **Purpose**: Prevent overfitting in high-dimensional data
- **Ridge Regression**: Penalizes extreme coefficient values
- **Benefit**: Reduces model complexity and improves generalization

---

## Database Integration

### MongoDB Operations
- **Connection**: Create `MongoClient` to establish database connection
- **Database Structure**: Port → Database → Collection → Documents
- **Document Format**: JSON-like structures where each document is an observation
- **Aggregation**: Use `.aggregate()` method with `group` and `match` operations
- **Data Import**: Use `.find()` method with projection to import into DataFrame

---

## Time Series Analysis

### Data Preparation
- **Index**: Set timestamps as DataFrame index
- **Timezone Localization**: Handle different timezone requirements
- **Resampling**: Create desired time intervals for analysis
- **Forward Fill**: Impute missing values using closest previous observation
- **Rolling Windows**: Calculate moving averages with fixed-size shifting windows

### Feature Engineering
- **Lag Features**: Create features from previous time units using `shift(1)`
- **Autocorrelation**: Measure correlation between time series and its lagged values
- **Temporal Patterns**: Visualize data fluctuations across different time windows

### Time Series Modeling
- **Linear Regression Approach**: Apply standard regression techniques to time series
- **Performance Baseline**: Use mean prediction for all observations
- **Evaluation**: Compare predicted vs actual values using line plots
- **Model Equation**: Express relationship as linear function of temporal features

---

## Advanced Time Series: ACF & PACF

### Autocorrelation Function (ACF)
- **Purpose**: Measures correlation between observations at different lag lengths
- **Interpretation**: 
  - Slow decay suggests non-stationary data requiring differencing
  - Peaks at regular intervals indicate seasonality
- **Use Case**: Determining Moving Average (MA) order in ARIMA models

### Partial Autocorrelation Function (PACF)
- **Purpose**: Measures direct correlation at specific lags, controlling for intermediate lags
- **Key Insight**: Removes "echo" effect from previous correlations
- **Model Selection**: ACF vs PACF comparison determines optimal model:
  - Slow ACF decay → AR model more appropriate
  - Slow PACF decay → MA model more appropriate

### Autoregressive (AR) Models
- **Concept**: Regression of time series on its own previous values
- **Training Process**: Fit model, make predictions, analyze residuals
- **Residual Analysis**: Should be close to zero and normally distributed
- **Diagnostic Plots**: Line plot, histogram, and ACF of residuals

### Walk Forward Validation
- **Problem**: Standard train/test split fails for time series
- **Solution**: Continuously retrain model with expanding window of data
- **Process**: Drop oldest data, include latest observations for each new prediction
- **Benefit**: More accurate performance estimation for temporal data

---

## ARIMA Models & Hyperparameter Tuning

### ARIMA Components
- **AR (p)**: Number of autoregressive terms (use PACF for guidance)
- **MA (q)**: Number of moving average terms (use ACF for guidance)
- **I (d)**: Degree of differencing to achieve stationarity

### Hyperparameter Optimization
- **Grid Search**: Test all combinations of p and q parameters
- **Evaluation Metric**: Mean Absolute Error across different parameter combinations
- **Trade-offs**: Balance between model complexity (computational cost) and performance
- **Validation**: Use walk-forward validation for time series data

### Model Diagnostics
- **Residual Analysis**: Essential for validating model assumptions
- **Diagnostic Plots**: Multiple visualizations to assess model fit
- **Parameter Interpretation**: Understand coefficient meanings and significance

---

## SQL Integration

### Database Setup
- **SQLite**: Serverless database for local development
- **IPython-SQL**: Enable SQL queries directly in Jupyter notebooks
- **Magic Commands**: Use `%sql` or `%%sql` for query execution
- **Connection**: Establish database connection using magic functions

### Query Execution
- **Exploration**: Use `%sql` for interactive database exploration
- **Data Import**: Use `pd.read_sql()` to import query results as DataFrame
- **Indexing**: Ensure proper indexing for query performance
- **Integration**: Seamlessly combine SQL queries with Python analysis

---

## Binary Classification: Logistic Regression

### Problem Setup
- **Target Variable**: Boolean/binary outcomes (0/1, True/False, Yes/No)
- **Sigmoid Function**: Maps linear regression output to probability [0,1]
- **Decision Threshold**: Typically 0.5 for classification boundary
- **Probability Interpretation**: Output represents likelihood of positive class

### Exploratory Analysis
- **Class Balance**: Crucial EDA step using bar charts
- **Majority vs Minority Class**: Identify class distribution imbalances
- **Normalization**: Convert counts to proportions for better understanding
- **Pivot Tables**: Analyze categorical relationships with target variable

### Model Training
- **Algorithm**: Uses sigmoid function with linear regression core
- **Convergence**: Set max_iter parameter to avoid warnings
- **Baseline**: Accuracy of predicting majority class for all observations
- **Performance**: Compare model accuracy to baseline accuracy

### Results Interpretation
- **Predicted Probabilities**: Use model's probability outputs
- **Odds Ratios**: Exponential of coefficients shows likelihood multipliers
- **Interpretation**: "X times more likely" statements using odds ratios
- **Visualization**: Bar charts for coefficient comparison across groups

---

## Decision Trees & Encoding

### Ordinal Encoding
- **Purpose**: Convert categorical variables to integers without creating new columns
- **Advantage**: Avoids curse of dimensionality with high-cardinality features
- **Risk**: May imply false ordering in features
- **Compatibility**: Works well with tree-based algorithms that use splits

### Decision Tree Characteristics
- **Structure**: Tree-like model with questions leading to binary decisions
- **Flexibility**: More flexible than linear models for complex patterns
- **Interpretability**: Easy to explain to non-technical stakeholders
- **Algorithm**: Progressively asks questions to separate data into classes

### Overfitting Management
- **Problem**: Trees can memorize training data without generalizing
- **Solution**: Hyperparameter tuning, especially tree depth
- **Validation Curve**: Plot performance vs depth to find optimal complexity
- **Early Stopping**: Identify point where validation performance peaks

### Model Interpretation
- **Tree Visualization**: Plot the actual decision tree structure
- **Gini Impurity**: Measure of node purity in decision process
- **Feature Importance**: Bar chart showing most influential features
- **Decision Path**: Follow tree branches to understand predictions

---

## Handling Imbalanced Data

### Data Assessment
- **Feature Distribution**: Examine distribution by class for each feature
- **Missing Values**: Identify patterns in missing data by class
- **Skewness**: Detect skewed features requiring special handling
- **Imputation Strategy**: Use median for skewed features instead of mean

### Resampling Techniques
- **Important Rule**: Only resample training data, never test data
- **Oversampling**: Increase minority class samples
- **Undersampling**: Reduce majority class samples
- **Baseline Adjustment**: High baseline accuracy expected with imbalanced data

### Evaluation Considerations
- **Confusion Matrix**: Critical for imbalanced data evaluation
- **Performance Metrics**: Accuracy alone insufficient for imbalanced datasets
- **Class-Specific Metrics**: Analyze performance for each class separately
- **Model Persistence**: Save and load models using pickle library

---

## Ensemble Methods: Random Forest

### Ensemble Concepts
- **Definition**: Combine multiple predictors for improved performance
- **Types**: Bagging (parallel), Boosting (sequential), Blending
- **Advantage**: Generally outperform single predictors
- **Random Forest**: Ensemble of decision trees using bagging

### Cross-Validation
- **K-Fold Process**: Divide data into k subsets
- **Training Strategy**: Use k-1 folds for training, 1 for validation
- **Rotation**: Each fold serves as validation once
- **Performance Estimate**: Average results across all k iterations
- **Generalization**: Close scores across folds indicate good generalization

### Hyperparameter Optimization
- **Grid Search**: Test all combinations of hyperparameter values
- **GridSearchCV**: Combines grid search with cross-validation
- **Process**: Train and evaluate models for each parameter combination
- **Selection**: Choose combination with best cross-validation score
- **Efficiency**: Balance between performance improvement and computational cost

---

## Gradient Boosting & Performance Metrics

### Gradient Boosting Trees
- **Concept**: Sequential ensemble where each tree corrects previous errors
- **Training Process**: Stage-wise addition of trees to minimize loss
- **Residual Learning**: Each new tree predicts where previous trees failed
- **Gradient Descent**: Uses gradient descent principles for optimization

### Performance Metrics Deep Dive

#### Recall
- **Definition**: "High recall, you gotta catch them all"
- **Formula**: TP / (TP + FN)
- **Purpose**: Measures ability to identify positive class observations
- **Threshold Relationship**: Lower threshold → better recall
- **Trade-off**: Often inversely related to precision

#### Precision
- **Definition**: "High precision, make careful decisions"
- **Formula**: TP / (TP + FP)
- **Purpose**: Measures quality of positive predictions
- **Focus**: Minimizes false positive errors
- **Application**: Critical when false positives are costly

#### Practical Applications
- **Medical Diagnosis**: High recall to catch all diseases
- **Spam Detection**: High precision to avoid blocking legitimate emails
- **Financial Fraud**: Balance between catching fraud and user experience

---

## Unsupervised Learning: Clustering

### Problem Setup
- **Data Structure**: Only feature matrix (X), no target vector (y)
- **Goal**: Segment population into groups based on proximity
- **Applications**: Customer segmentation, social stratification
- **Exploration**: Scatterplots reveal cluster patterns instead of linear relationships

### K-Means Algorithm
- **Initialization**: Randomly place k centroids
- **Assignment**: Group data points to nearest centroid
- **Update**: Recalculate centroids as mean of assigned points
- **Iteration**: Repeat until centroids stabilize
- **Visualization**: Color-code clusters and mark centroids with different symbols

### Evaluation Metrics
- **Inertia**: Sum of squared distances from points to cluster centers
- **Silhouette Score**: Measures density and separation of clusters
- **Formula**: [a-b]/max(a,b) where a=intra-cluster distance, b=inter-cluster distance
- **Range**: -1 to 1, with 1 being optimal
- **Goal**: Maximize silhouette score while minimizing inertia

### Optimal Cluster Selection
- **Elbow Method**: Plot inertia vs number of clusters
- **Sweet Spot**: Find point of diminishing returns in inertia reduction
- **Multiple Solutions**: May have several appropriate k values
- **Validation**: Unlike supervised learning, multiple valid solutions exist

### Communication of Results
- **Visualization**: Scatterplot with cluster-colored data points
- **Analysis**: Side-by-side bar charts of feature means per cluster
- **Strategy**: Use cluster characteristics for targeted approaches
- **Interpretation**: Analyze feature proportions and patterns across clusters

---

## Model Deployment & Best Practices

### Model Persistence
- **Pickle Library**: Save trained models to files
- **Loading**: Restore models for production use
- **Versioning**: Track model versions for reproducibility

### Validation Strategies
- **Training Validation**: Ensure model learns from training data
- **Test Performance**: Evaluate generalization on unseen data
- **Cross-Validation**: Robust performance estimation
- **Walk-Forward**: Special validation for time series data

### Ethical Considerations
- **Bias Detection**: Identify unfair model behaviors
- **Transparency**: Ensure model decisions are explainable
- **Fairness**: Consider impact on different population groups
- **Accountability**: Establish responsibility for model decisions

### Performance Optimization
- **Feature Selection**: Choose most relevant features
- **Hyperparameter Tuning**: Optimize model settings
- **Regularization**: Prevent overfitting
- **Ensemble Methods**: Combine models for better performance

---

## Key Takeaways & Best Practices

### Data Quality First
- Clean, well-prepared data is foundational to successful models
- Spend significant time on EDA and preprocessing
- Handle missing values and outliers appropriately
- Validate data quality throughout the pipeline

### Model Selection Strategy
- Start with simple baselines before complex models
- Use cross-validation for robust performance estimation
- Consider the trade-offs between interpretability and performance
- Choose metrics appropriate for your specific problem

### Validation & Testing
- Never test on training data
- Reserve test set until final evaluation
- Use appropriate validation strategies for your data type
- Consider class imbalance when selecting metrics

### Communication & Deployment
- Visualize results effectively for stakeholders
- Document model decisions and limitations
- Plan for model monitoring and updates in production
- Consider ethical implications of model decisions

This comprehensive guide covers the essential concepts and practical techniques needed for effective data science work, from basic data manipulation through advanced machine learning models and deployment considerations.
