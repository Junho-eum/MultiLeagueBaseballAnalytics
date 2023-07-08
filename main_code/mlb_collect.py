import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from scipy.stats import norm
import warnings
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from dataclasses import dataclass
matplotlib_axes_logger.setLevel('ERROR')
warnings.filterwarnings('ignore')

import json

df = pd.read_csv("./train.csv")

teamBoxScores_data = df["games"].dropna()

# Convert the string representation of JSON to actual JSON objects
teamBoxScores_data = teamBoxScores_data.apply(lambda x: json.loads(x.replace('null', 'None')) if pd.notnull(x) else None)

# Access the nested JSON field
teamBoxScores_nested_json = teamBoxScores_data.apply(lambda x: x[0] if len(x) > 0 else None)

# Convert to DataFrame
teamBoxScores_df = pd.json_normalize(teamBoxScores_nested_json) # or teamBoxScores_df = pd.json_normalize(teamBoxScores_nested_json) if you have pandas version 1.0.0 or higher.

# Save as CSV
teamBoxScores_df.to_csv('./mlb_data.csv', index=False)
