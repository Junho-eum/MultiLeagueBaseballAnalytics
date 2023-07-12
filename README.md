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

## pythagorean_expectation_modeling.py
### Preprocessing the Data
  The script starts by loading two data files: team_win_loss_probabilities.csv and teams.csv. The former is presumed to contain historical win-loss probability data for baseball teams, while the latter contains team identifiers and corresponding team names. A dictionary mapping team IDs to team names is created from the teams.csv data.
  
  ```
  team_mapping = teams.set_index('id')['name'].to_dict()
  ```
  
  Then, the script enters a loop that ranges from the year 2018 to 2021 (the last year is exclusive in Python's range function). For each year in the range:
  
  ```
  for year in range(2018, 2022):
  ```
  
  The script reads in the yearly team box score data from a CSV file named in the format year_{year}.csv (for example, year_2018.csv for 2018). This data file is presumed to contain box score data (e.g., runs scored and runs allowed) for each team for that year. It then creates a column year and sets it to the current year being processed.
  
  The box score data is grouped by team ID and year, and the total (sum) runs scored and runs allowed for each team in that year is computed.
  ```
  team_boxscore_grouped = team_boxscore.groupby(['teamId', 'year']).agg({
    'runsScored': 'sum',
    'runsPitching': 'sum',  
  }).reset_index()
  ```
  Finally, it merges the processed box score data with the win-loss probability data based on team name and year. If the merged data is empty (which could be due to missing win-loss probability data for that year or team), the script will again skip the rest of the current loop iteration and move to the next year.
  ```
  dataset = pd.merge(team_boxscore_grouped, win_loss_prob, left_on=['teamName', 'year'], right_on=['homeName', 'season'])
  if dataset.empty:
      print(f'Merged DataFrame is empty for year {year}. Check your merge operation.')
      continue
  ```
  
  At the end of this preprocessing step, for each year in the given range, you will have a dataset that includes total runs scored, total runs allowed, team names, and corresponding win-loss probabilities for each team. These datasets are then saved as pythagorean_output_{year}.csv.

### Pythagorean Expectation Calculation and Modeling
The Pythagorean expectation is a formula that estimates a team's win percentage based on the number of runs it scores and allows. In baseball analytics, it's often used to evaluate a team's performance compared to their actual win percentage.

The script computes the Pythagorean expectation for each team in the processed data and adds it as a new column, pythag_expect.
```
df['pythag_expect'] = (df[runsScored] ** 2) / (df[runsScored] ** 2 + df[runsAllowed] ** 2)
```

The script then prepares the data for the machine learning model. It creates a 2-D feature matrix X with runs scored and runs allowed, and a target variable y with the win-loss probability.
```
X = df[[runsScored, runsAllowed]]
y = df[win_loss_probability]
```

Next, the script uses scikit-learn's PolynomialFeatures to generate polynomial and interaction features from the runs scored and runs allowed. This expanded feature matrix X_poly is then split into training and testing datasets, with 80% of the data used for training and 20% for testing.
```
poly = PolynomialFeatures(degree=2)
```

A linear regression model is trained using the training data. The model's performance is evaluated on the test data using the mean squared error (MSE) and the R-squared (R²) statistic, which are printed out for each year.
```
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse_pythag = mean_squared_error(y_test, y_pred)
r2_pythag = r2_score(y_test, y_pred)
print(f'Pythagorean Expectation: MSE = {mse_pythag}, R^2 = {r2_pythag}')
```

This process is repeated for each year in the range, resulting in a separate model and performance metrics for each year. It allows you to see how the model's performance varies from year to year and potentially spot any trends or anomalies.
