# Baseball_Analytics

## Data Collection Process
- This project involves data from multiple baseball leagues, each with their unique playing styles. The data for these analyses was collected from the following source:
https://www.kaggle.com/competitions/mlb-player-digital-engagement-forecasting/data?select=train.csv

## Extracting Yearly Team Box Score Data (mlb_collect.py)
- This script is used to process and transform the MLB data sourced from the Kaggle MLB Player Digital Engagement Forecasting competition. The raw data includes nested JSON within a CSV file, which is then extracted and transformed into a more usable format.

- Here's a breakdown of the steps:
  1. **Download Data**: The raw data can be downloaded directly from the Kaggle competition page. Make sure to download the 'train.csv' file.

  2. **Load Data**: The dataset is loaded into a pandas DataFrame from the 'train.csv' file.
     
  ```
  import pandas as pd
  import json
  df = pd.read_csv("./train.csv")
  ```
  
  3. **Extract and Transform Data**: The "games" column from the DataFrame, which contains the JSON data of interest, is isolated. The JSON string is transformed into actual JSON objects and then converted to a pandas DataFrame using the json_normalize function.
     
  ```
  teamBoxScores_data = df["games"].dropna()
  teamBoxScores_data = teamBoxScores_data.apply(lambda x: json.loads(x.replace('null', 'None')) if pd.notnull(x) else None)
  teamBoxScores_df = pd.json_normalize(teamBoxScores_nested_json)
  ```
  
  4. **Preprocessing the Data**: The "gameDate" column, which contains the dates of games, is converted to datetime format. The year from each game date is extracted and stored in a new column, "year".
     
    ```
    data = pd.read_csv('mlb_data.csv')
    data['gameDate'] = pd.to_datetime(data['gameDate'])
    data['year'] = data['gameDate'].dt.year
    ```
    
  5. **Yearly Team Box Score Data Extraction**: The data is grouped by year and by team ID to compute yearly statistics for each team. The computed yearly team stats are saved to separate CSV files for each year, which provides us with the yearly team box score data.
     
   ```
   for year in data['year'].unique():
    yearly_data = data[data['year'] == year]
    yearly_team_stats = yearly_data.groupby('teamId').sum()
    yearly_team_stats.to_csv(f'year_{year}.csv')
   ```
   
  To run this script, use the following command:
  
   ```
   python mlb_collect.py
   ```
   
   Please ensure that the input CSV file ('train.csv') is in the correct path before running the script. After the script has run, check the output CSV files to see the extracted yearly team box score data.

Remember to check and confirm that the output files ('year_{year}.csv') are in the desired output directory after running the script.

Keep in mind that the structure of your input data file ('train.csv') should conform to the structure expected by the script for it to run successfully. Check the example data provided for the expected format and structure.

##  Standardize different datasets from different leagues into a consistent and structured format (team_season_stat_gen.py)

  - The function preprocess_league_data inside team_season_stat_gen.py script is used to standardize different datasets from different leagues into a consistent and structured format. This is crucial in ensuring the integrity and compatibility of the data during subsequent analysis.
  
  - The function takes as input three CSV files: the first containing data about each game (including the season, home team, away team, home team wins, and home team losses), the second containing more detailed statistics about each game, and the third containing information about the teams.
    
  ```
  df1 = pd.read_csv(df1_path)
  df2 = pd.read_csv(df2_path)
  df_team = pd.read_csv(team_info_path)
  ```

The function performs several transformations and aggregations on the data:

  - It calculates the total wins and losses for each team in each season, both for home games and away games.
    
  - It then calculates the win-loss percentage for each team in each season.
    
  - It averages the detailed statistics for each team in each season.
    
  - It merges these different dataframes into a final dataframe.
    
  - The resulting dataframe provides a complete and detailed view of the performance of each team in each season. Each row represents one team in one season and includes information about the number of wins and losses, the win-loss percentage, and other statistics averaged over all the games in that season.

  ```
  df1_home = df1.groupby(['season', 'homeId']).agg({
      'homeWins': 'sum',
      'homeLosses': 'sum'
  }).reset_index().rename(columns={'homeId': 'teamId', 'homeWins': 'wins', 'homeLosses': 'losses'})
  
  df1_away = df1.groupby(['season', 'awayId']).agg({
      'awayWins': 'sum',
      'awayLosses': 'sum'
  }).reset_index().rename(columns={'awayId': 'teamId', 'awayWins': 'wins', 'awayLosses': 'losses'})
  
  team_season_data = pd.concat([df1_home, df1_away])
  
  team_season_data = team_season_data.groupby(['season', 'teamId']).agg({'wins': 'sum', 'losses': 'sum'}).reset_index()
  ```

  - Finally, the function also renames the columns in the final dataframe to match a standard format. This makes the data easier to work with, as the column names are consistent across different leagues and datasets.

  ```
  column_mapping = {...}
  final_df.rename(columns=column_mapping, inplace=True)
  final_df = final_df.drop(columns='id')
  ```

  - By preparing the data in this way, the preprocess_league_data function makes it easier to analyze the performance of different teams across different seasons and even across different leagues.

## Expected Output
  - The script will output the Mean Squared Error (MSE) and R-Squared (R^2) for each year in the data. Here are some example outputs you might expect:
    ```
    Pythagorean Expectation: MSE = 0.043825052872218384, R^2 = 0.04972759343921895
    Pythagorean Expectation: MSE = 0.03606235134234921, R^2 = -0.17769687170829518
    Pythagorean Expectation: MSE = 0.004137536723744783, R^2 = 0.6199216545300004
    Pythagorean Expectation: MSE = 0.0035629027629842376, R^2 = -0.37132119335757463
    ```
- Additionally, the script will create an LDA scatter plot. An example of such a plot might look something like this:

![user_input_1](https://github.com/Junho-eum/Baseball_Analytics/assets/74083204/0a961a73-b4be-4af6-ad89-00c69217ac4e)

## extract_mlb_game_data.py
- This script loads, parses, and transforms a dataset containing information about ba!
seball games. The source data is assumed to be in CSV format, with one of the columns containing JSON strings that encapsulate detailed game data.
  
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
