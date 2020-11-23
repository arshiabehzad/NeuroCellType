import pandas as pd

df = pd.read_csv("cell_types_specimen_details.csv")
new_df = df.filter(['specimen__id','specimen__hemisphere','structure__name','ef__avg_firing_rate','tag__dendrite_type','structure__name', 'donor__species'])
print(new_df.head())