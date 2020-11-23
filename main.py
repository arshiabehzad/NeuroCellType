import pandas as pd


def main():
    #reads csv file
    df = pd.read_csv("cell_types_specimen_details.csv")
    #filter to columns we want
    new_df = df.filter(['specimen__id','specimen__hemisphere','structure__name', 'structure_parent__acronym','ef__avg_firing_rate','tag__dendrite_type', 'donor__species'])
    #drops rows with null values
    new_df.dropna(inplace = True)
    #drops rows with sparsly spiny as dendrite type
    new_df.drop(new_df.index[new_df['tag__dendrite_type'] == 'sparsely spiny'], inplace=True)
    #creates new column with left as 1 and right as 0
    new_df['specimen__hemisphere_left'] = new_df.specimen__hemisphere.map({'left': 1, 'right': 0})
    #creates new column with spiny as 1 and aspiny as 0
    new_df['tag__dendrite_type_spiny'] = new_df.tag__dendrite_type.map({'spiny': 1, 'aspiny': 0})
    #creates new column with human as 1 and mouse as 0
    new_df['donor_species_human'] = new_df.donor__species.map( {'Homo Sapiens': 1, 'Mus musculus': 0})
    pd.set_option('display.max_columns', None)
    averages(new_df)

def averages(new_df):
    
    print(new_df.groupby('tag__dendrite_type')['ef__avg_firing_rate'].mean())


if __name__ == '__main__':
    main()

