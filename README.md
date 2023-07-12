# Baseball_Analytics

## Data Collection Process
- This project involves data from multiple baseball leagues, each with their unique playing styles. The data for these analyses was collected from the following sources:

### Major League Baseball (MLB) Data

- The MLB data was sourced from the Kaggle MLB Player Digital Engagement Forecasting competition. The relevant data includes team box scores, team win/loss probabilities, and team data:

- **Download Data**: The data can be downloaded directly from the Kaggle competition page. Make sure to download the 'train.csv' file.

- **Extract and Transform Data**: The downloaded data is in nested JSON format within a CSV file. To handle this data, each row was read in and transformed from JSON into a pandas DataFrame. For each date in the dataset, a new row was created for each team, including metrics such as runs scored, runs allowed, and other stats.

- **Data Concatenation**: The resulting DataFrames were then concatenated to create a complete dataset for analysis.

- The extraction and transformation process was carried out using custom Python scripts using pandas and json libraries. The scripts read the nested JSON, transformed it into a more readable format, and created separate datasets for different statistics such as box scores and team win/loss probabilities.
