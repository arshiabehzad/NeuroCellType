import pandas as pd


def main():
    df = pd.read_csv("cell_types_specimen_details.csv")
    new_df = df.filter(['specimen__id','specimen__hemisphere','structure__name','ef__avg_firing_rate','tag__dendrite_type', 'donor__species'])
    new_df.dropna(inplace = True)
    print(new_df.head())
    new_df.info()

if __name__ == '__main__':
    main()

