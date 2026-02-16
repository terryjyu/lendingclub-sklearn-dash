import pandas as pd
import json
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "lc_cleaned_combined.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "lendingclub-modern", "backend", "models")

def calculate_stats():
    print(f"Loading data from {DATA_PATH}...")
    df2 = pd.read_csv(DATA_PATH, low_memory=False)
    
    # Region Logic
    west = ['CA', 'OR', 'UT','WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID']
    south_west = ['AZ', 'TX', 'NM', 'OK']
    south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN' ]
    mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
    north_east = ['CT', 'NY', 'PA', 'NJ', 'RI','MA', 'MD', 'VT', 'NH', 'ME']

    def finding_regions(state):
        if state in west: return 'West'
        elif state in south_west: return 'SouthWest'
        elif state in south_east: return 'SouthEast'
        elif state in mid_west: return 'MidWest'
        elif state in north_east: return 'NorthEast'
        return 'Other'

    print("Processing regions...")
    df2['region'] = df2['addr_state'].apply(finding_regions)
    df2['year'] = pd.to_datetime(df2['issue_d'], format='%b-%Y').dt.year
    
    # 1. Region Stats
    print("Generating stats_region.json...")
    stats_region = df2.groupby(['region', 'year'], as_index=False)[['loan_amnt','funded_amnt','funded_amnt_inv']].sum()
    stats_region.to_json(os.path.join(OUTPUT_DIR, "stats_region.json"), orient="records")
    
    # 2. State Stats
    print("Generating stats_state.json...")
    stats_state = df2.groupby(['addr_state', 'year'], as_index=False)[['loan_amnt','funded_amnt','funded_amnt_inv']].sum()
    stats_state.to_json(os.path.join(OUTPUT_DIR, "stats_state.json"), orient="records")
    
    # 3. Map Stats
    print("Generating stats_map.json...")
    df2['Charged_Off'] = [1 if x=='Charged Off' else 0 for x in df2['loan_status']]
    df2['Fully_Paid'] = [1 if x=='Fully Paid' else 0 for x in df2['loan_status']]
    stats_map = df2.groupby(['addr_state', 'year'], as_index=False)[['Charged_Off','Fully_Paid']].sum()
    stats_map['Fully_Paid_percentage'] = (stats_map['Fully_Paid']/(stats_map['Fully_Paid']+stats_map['Charged_Off'])).round(4)*100
    stats_map.to_json(os.path.join(OUTPUT_DIR, "stats_map.json"), orient="records")
    
    # 4. Sample Data (for Boxplots)
    print("Generating sample.json (2000 rows)...")
    cols = ['grade', 'home_ownership', 'purpose', 'emp_length', 'int_rate', 'annual_inc', 'loan_amnt', 'loan_status']
    sample = df2[cols].sample(n=2000)
    sample.to_json(os.path.join(OUTPUT_DIR, "sample.json"), orient="records")
    
    print("Done! Stats saved to", OUTPUT_DIR)

if __name__ == "__main__":
    calculate_stats()
