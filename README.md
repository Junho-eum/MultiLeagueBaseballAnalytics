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
    
- **Data Concatenation**: The resulting DataFrames were then concatenated to create a complete dataset for analysis.



- The extraction and transformation process was carried out using custom Python scripts using pandas and json libraries. The scripts read the nested JSON, transformed it into a more readable format, and created separate datasets for different statistics such as box scores and team win/loss probabilities.
