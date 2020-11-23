import pandas as pd


def main():
    #reads csv file
    df = pd.read_csv("cell_types_specimen_details.csv")
    #filter to columns we want
    new_df = df.filter(['specimen__id','specimen__hemisphere','structure__name','ef__avg_firing_rate','tag__dendrite_type', 'donor__species'])
    #drops rows with null values
    new_df.dropna(inplace = True)
    #drops rows with sparsly spiny as dendrite type
    new_df.drop(new_df.index[new_df['tag__dendrite_type'] == 'sparsely spiny'], inplace=True)
    new_df['specimen__hemisphere_left'] = new_df.specimen__hemisphere.map({'left': 1, 'right': 0})
    print(new_df.head())
    new_df.info()

if __name__ == '__main__':
    main()

