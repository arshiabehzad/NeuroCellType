import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def average(new_df, group_by_column, columns_to_average):
    avg = new_df.groupby(group_by_column)[columns_to_average].mean()
    return avg

def heatmapgen(new_df):

    #initialize new dataframe
    average_df = pd.DataFrame(new_df,columns=['structure_parent__acronym','ef__avg_firing_rate'])

    #sort data into buckets
    average_df.loc[average_df['ef__avg_firing_rate'] > 10, 'bucket'] = '10-20'
    average_df.loc[average_df['ef__avg_firing_rate'] > 20, 'bucket'] = '20-30'
    average_df.loc[average_df['ef__avg_firing_rate'] > 30, 'bucket'] = '30-40'
    average_df.loc[average_df['ef__avg_firing_rate'] > 40, 'bucket'] = '40-50'
    average_df.loc[average_df['ef__avg_firing_rate'] > 50, 'bucket'] = '50-60'
    average_df.loc[average_df['ef__avg_firing_rate'] > 60, 'bucket'] = '60-70'
    average_df.loc[average_df['ef__avg_firing_rate'] > 70, 'bucket'] = '70-80'
    average_df.loc[average_df['ef__avg_firing_rate'] > 80, 'bucket'] = '80-90'
    average_df.loc[average_df['ef__avg_firing_rate'] > 90, 'bucket'] = '90-100'
    average_df.loc[average_df['ef__avg_firing_rate'] > 100, 'bucket'] = '100-110'
    average_df.loc[average_df['ef__avg_firing_rate'] > 110, 'bucket'] = '110-120'
    average_df.loc[average_df['ef__avg_firing_rate'] > 120, 'bucket'] = '120-130'
    average_df.loc[average_df['ef__avg_firing_rate'] > 130, 'bucket'] = '130-140'
    average_df.loc[average_df['ef__avg_firing_rate'] > 140, 'bucket'] = '140-150'
    average_df.loc[average_df['ef__avg_firing_rate'] > 150, 'bucket'] = '150-160'

    #create heatmap
    heatmap_data = pd.pivot_table(average_df, values='ef__avg_firing_rate', index=['structure_parent__acronym'],columns=['bucket'])
    column_order = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100','100-110','110-120','120-130','130-140','140-150','150-160']
    heatmap_data = heatmap_data.reindex(column_order, axis=1)
    g = sns.heatmap(heatmap_data, yticklabels=1, cmap='coolwarm')
    plt.show()

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
    #print(average(new_df, 'tag__dendrite_type_spiny', [ 'ef__avg_firing_rate' ]))
    heatmapgen(new_df)


if __name__ == '__main__':
    main()

