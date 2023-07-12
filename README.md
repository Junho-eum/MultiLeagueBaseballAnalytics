# Baseball_Analytics

## Data Collection Process
- This project involves data from multiple baseball leagues, each with their unique playing styles. The data for these analyses was collected from the following sources:

### Major League Baseball (MLB) Data

- The MLB data was sourced from the Kaggle MLB Player Digital Engagement Forecasting competition. The relevant data includes team box scores, team win/loss probabilities, and team data:

- **Download Data**: The data can be downloaded directly from the Kaggle competition page. Make sure to download the 'train.csv' file.

- **Extract and Transform Data**: The downloaded data is in nested JSON format within a CSV file. To handle this data, each row was read in and transformed from JSON into a pandas DataFrame. For each date in the dataset, a new row was created for each team, including metrics such as runs scored, runs allowed, and other stats.
  - First, the dataset was loaded into a pandas DataFrame from the CSV file named "train.csv".
    ```
    import pandas as pd
    import json
    df = pd.read_csv("./train.csv")
    ```
  - The "games" column from the DataFrame, which contains the JSON data of interest, was isolated and stored in a variable. This data was then transformed from a string representation of JSON to actual JSON objects.
    ```
    teamBoxScores_data = df["games"].dropna()
    teamBoxScores_data = teamBoxScores_data.apply(lambda x: json.loads(x.replace('null', 'None')) if pd.notnull(x) else None)
    ```
  - The JSON data was then converted to a pandas DataFrame using the json_normalize function.
    ```
    teamBoxScores_df = pd.json_normalize(teamBoxScores_nested_json)
    ```
  - The resulting DataFrame was saved to a CSV file named "mlb_data.csv".
    ```
    teamBoxScores_df.to_csv('./mlb_data.csv', index=False)
    ```
- **Preprocessing the Data**: After the data extraction and conversion, some preprocessing steps were applied. The "gameDate" column, which contains the dates of games, was converted to the datetime format.
  ```
  data = pd.read_csv('mlb_data.csv')
  data['gameDate'] = pd.to_datetime(data['gameDate'])
  ```
  - The year from each game date was extracted and stored in a new column, "year".
  ```
  data['year'] = data['gameDate'].dt.year
  ```
  - Finally, the data was grouped by year and by team ID to compute yearly statistics for each team. The computed yearly team stats were saved to separate CSV files for each year.
  - With the data now unpacked from its nested JSON format and grouped by year and team, it is ready for easier analysis and modeling.
  ```
  for year in data['year'].unique():
    yearly_data = data[data['year'] == year]
    yearly_team_stats = yearly_data.groupby('teamId').sum()
    yearly_team_stats.to_csv(f'year_{year}.csv')
  ```
- The extraction and transformation process was carried out using mlb_collect.py using pandas and json libraries. The scripts read the nested JSON, transformed it into a more readable format, and created separate datasets for different statistics such as box scores and team win/loss probabilities.



## extract_mlb_game_data.py
- This script loads, parses, and transforms a dataset containing information about baseball games. The source data is assumed to be in CSV format, with one of the columns containing JSON strings that encapsulate detailed game data.
  
  1. Data Loading
  The script starts by loading a CSV file into a Pandas DataFrame.
  
  2. Dataframe Initialization
  An empty DataFrame is initialized. This will be used to store the processed game data.
  
  3. Row Iteration
  The script iterates over each row in the original DataFrame, checking for any missing or malformed data in the 'games' column.
  
  4. JSON Data Parsing
  For rows with valid 'games' data, the script attempts to parse the JSON data into a Python object. It handles any potential JSON decoding errors gracefully by printing an error message and continuing with the next row.
  
  5. JSON to DataFrame Conversion
  Upon successful parsing, the script converts the Python object (derived from the JSON data) into a DataFrame. This process is known as 'flattening' the JSON structure.
  
  6. Appending to the Main DataFrame
  The new DataFrame is then appended to the main DataFrame, ensuring a continuous index across the whole dataset.
  
  7. Saving the DataFrame
  Finally, the complete DataFrame, containing the parsed and transformed game data, is saved to a new CSV file.
  
  This script is particularly useful for scenarios where game data is stored in JSON format within a CSV file, and where this data needs to be extracted, transformed, and saved in a flat, tabular format. Note that the script specifically looks for a 'games' column in the source CSV file and expects this to contain valid JSON strings.
  
## PreprocessModelData Class

The PreprocessModelData class is a Python utility that allows for the preprocessing of structured datasets (e.g., CSV files). The class provides functions for handling missing data, separating features from target variables, scaling numerical features, and splitting datasets into training and testing subsets.

  - **Initialization**
    - The class is initialized with a dictionary (impute_strategy_dict) that specifies the imputation strategy ('mean', 'median', or 'mode') for each column in the dataframe that has missing data.
  
  - **handle_missing_data method**
    - This method goes through each column in the dataframe and fills missing values according to the specified strategy.
  
  - **separate_features_target method**
    - This method separates the features and the target column in the dataframe. If the target column is 'win_loss_percentage', it creates a new column 'categorical_win_loss_prob' which categorizes 'win_loss_percentage' into 'low', 'medium', or 'high' categories.
  
  - **scale_features method**
    - This method standardizes the features in the dataframe using the StandardScaler from scikit-learn. This is often a necessary step before using many machine learning algorithms.
  
  - **split_data method**
    - This method splits the dataset into training and testing subsets. It uses scikit-learn's train_test_split function, which shuffles the dataset and splits it. The default test size is 20% of the total dataset.
  
  - The PreprocessData class can be used in combination with the extract_mlb_game_data.py script. The script extracts and transforms the game data, and the class preprocesses it for machine learning applications. The imputation of missing values, scaling of features, and splitting of data are all common steps in a machine learning pipeline, and this class provides an easy and reusable way to perform these tasks.


## LDA Class
  - **Initialization**
    - The class doesn't require any parameters during initialization.
    
  - **fit_lda method**
    - This method fits the LDA model on the training data. It takes the training data (X_train and y_train) and the number of components to keep (n_components) as input and returns the fitted LDA model and the transformed X_train.
    
  - **plot_lda method**
    - This method creates a scatter plot of the data in the first two linear discriminant spaces. The points are colored according to their class labels. This visualization helps understand how well the classes are separated by the LDA.

- **lda_coef method**
    - This method returns a dataframe that contains the coefficients of the discriminant function for each class. It can help interpret the impact of the features on the classification.

- **Usage in Main Script**
    - Both PreprocessModelData and LDA_explore classes are utilized in the main script for data preprocessing and analysis respectively. Here's a brief step-by-step rundown of the main script:

  1. The script first initializes the PreprocessModelData and LDA_explore classes.
  
  2. The script reads a CSV file into a dataframe and preprocesses it by handling missing values using the specified strategy.
  
  3. The features and target variable are separated from the dataframe. The target variable 'win_loss_percentage' is transformed into a categorical variable.
  
  4. The features are scaled using the scale_features method.
  
  5. The preprocessed dataset is split into a training set and a test set.
  
  6. LDA is fitted on the training data using the fit_lda method.
  
  7. A scatter plot of the data in the first two linear discriminant spaces is displayed.
  
  8. The lda_coef method is used to get a dataframe that contains the coefficients of the discriminant function for each class. These coefficients are saved to a CSV file.
  
  9. By incorporating both preprocessing and analysis in the script, we can efficiently prepare our data for machine learning and perform exploratory data analysis.

## pythagorean_expectation_modeling.py
### Preprocessing the Data
  1. The script starts by loading two data files: team_win_loss_probabilities.csv and teams.csv. The former is presumed to contain historical win-loss probability data for baseball teams, while the latter contains team identifiers and corresponding team names. A dictionary mapping team IDs to team names is created from the teams.csv data.
  
  ```
  team_mapping = teams.set_index('id')['name'].to_dict()
  ```
  
  2. Then, the script enters a loop that ranges from the year 2018 to 2021 (the last year is exclusive in Python's range function). For each year in the range:
  
  ```
  for year in range(2018, 2022):
  ```
  
  3. The script reads in the yearly team box score data from a CSV file named in the format year_{year}.csv (for example, year_2018.csv for 2018). This data file is presumed to contain box score data (e.g., runs scored and runs allowed) for each team for that year. It then creates a column year and sets it to the current year being processed.
  
  The box score data is grouped by team ID and year, and the total (sum) runs scored and runs allowed for each team in that year is computed.
  ```
  team_boxscore_grouped = team_boxscore.groupby(['teamId', 'year']).agg({
    'runsScored': 'sum',
    'runsPitching': 'sum',  
  }).reset_index()
  ```
  4. Finally, it merges the processed box score data with the win-loss probability data based on team name and year. If the merged data is empty (which could be due to missing win-loss probability data for that year or team), the script will again skip the rest of the current loop iteration and move to the next year.
  ```
  dataset = pd.merge(team_boxscore_grouped, win_loss_prob, left_on=['teamName', 'year'], right_on=['homeName', 'season'])
  if dataset.empty:
      print(f'Merged DataFrame is empty for year {year}. Check your merge operation.')
      continue
  ```
  
  At the end of this preprocessing step, for each year in the given range, you will have a dataset that includes total runs scored, total runs allowed, team names, and corresponding win-loss probabilities for each team. These datasets are then saved as pythagorean_output_{year}.csv.


## Pythagorean Expectation Calculation and Modeling
- The Pythagorean expectation is a formula that estimates a team's win percentage based on the number of runs it scores and allows. In baseball analytics, it's often used to evaluate a team's performance compared to their actual win percentage.

- The script computes the Pythagorean expectation for each team in the processed data and adds it as a new column, pythag_expect.
  ```
  df['pythag_expect'] = (df[runsScored] ** 2) / (df[runsScored] ** 2 + df[runsAllowed] ** 2)
  ```

- The script then prepares the data for the machine learning model. It creates a 2-D feature matrix X with runs scored and runs allowed, and a target variable y with the win-loss probability.
  ```
  X = df[[runsScored, runsAllowed]]
  y = df[win_loss_probability]
  ```

- Next, the script uses scikit-learn's PolynomialFeatures to generate polynomial and interaction features from the runs scored and runs allowed. This expanded feature matrix X_poly is then split into training and testing datasets, with 80% of the data used for training and 20% for testing.
  ```
  poly = PolynomialFeatures(degree=2)
  ```

- A linear regression model is trained using the training data. The model's performance is evaluated on the test data using the mean squared error (MSE) and the R-squared (R²) statistic, which are printed out for each year.
  ```
  model = LinearRegression()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  mse_pythag = mean_squared_error(y_test, y_pred)
  r2_pythag = r2_score(y_test, y_pred)
  print(f'Pythagorean Expectation: MSE = {mse_pythag}, R^2 = {r2_pythag}')
  ```

- This process is repeated for each year in the range, resulting in a separate model and performance metrics for each year. It allows you to see how the model's performance varies from year to year and potentially spot any trends or anomalies.
