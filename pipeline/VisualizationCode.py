import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import squarify
import geopandas as gpd
from shapely import wkt
import warnings
warnings.filterwarnings('ignore')

import Make_By_Officer_DF as make_officer_df
import Feat_Engineering as feat_engineering_helpers
import Run_Model as run_model


def create_viz(t1,t2):

    complaints_t1, complaints_t2 = make_officer_df.get_relevant_complaints(t1, t2)
    additional_cont_feat, final_df = make_officer_df.make_df(t1, t2)
    complaints = pd.read_csv("../data/complaints-complaints.csv.gz", compression="gzip")
    complaints_accused = pd.read_csv("../data/complaints-accused.csv.gz", compression="gzip")
    make_target_pct(final_df)
    make_complaints_map(complaints)
    make_complaints_leveled(complaints_t1)
    make_complaints_by_officer(complaints_t1)

def make_target_pct(final_df):
    desired_targets = ['target_sustained', 'target_use_of_force']
    target_col = "target_sustained_force"
    make_target_col(final_df, desired_targets, target_col)

    desired_targets = ['target_sustained', 'target_use_of_force', 'target_drug']
    target_col = "target_sustained_force_drug"
    make_target_col(final_df, desired_targets, target_col)

    desired_targets = ['target_sustained', 'target_use_of_force', 'target_drug', 'target_racial']
    target_col = "target_sustained_force_drug_racial"
    make_target_col(final_df, desired_targets, target_col)

    desired_targets = ['target_use_of_force', 'target_drug', 'target_racial',
                       'target_sustained', 'target_nonviolent']
    target_col = "any_known_complaint"
    make_target_col(final_df, desired_targets, target_col)

    percentage_target_df = final_df[["target_sustained", "target_sustained_force", 
                                          "target_sustained_force_drug", "target_sustained_force_drug_racial",
                                          "any_known_complaint"]]
    
    percentage_target_df.rename(columns={"target_sustained": "Sustanined complaints", 
        "target_sustained_force": "Sustained and force complaints", 
        "target_sustained_force_drug": "Sustained, force and drug complaints", 
        "target_sustained_force_drug_racial": "Sustained, force, drug and racial complaints", 
        "any_known_complaint": "Severe complaints"}, inplace=True)

    percentage_target = pd.DataFrame(columns = ["Target", "True", "False"])
    for col in percentage_target_df:
        count = percentage_target_df[col].value_counts()
        percentage_target = percentage_target.append({"Target": col, "True": count[1], "False":count[0]}, ignore_index = True)

        percentage_target['Percentage True'] = percentage_target['True'] * 100 / (percentage_target['True'] + percentage_target['False'])

    fig = plt.figure(figsize = (10, 5)) 
    plt.bar(percentage_target['Target'], percentage_target['Percentage True'], color ='steelblue',  
            width = 0.4)
    plt.xticks(rotation=90)
    plt.xlabel("Target Configuration") 
    plt.ylabel("Percentage Share") 
    plt.title("SHARE OF TRUE VALUES PER TARGET CONFIGURATION", size =18) 
    plt.savefig('target_pct.png', bbox_inches='tight')


def make_target_col(final_df, desired_targets, col_name):
    '''
    Written by Lily, moved to Feat_Engineering py file by Sasha on June 7th.
    Creates a target column in the target_df that will be true if at least one of the desired targets is true.
    Inputs:
        desired_targets: list of target columns to include
        col_name: name of the new target column
    '''
    final_df[col_name] = final_df[desired_targets].any(axis='columns')
    return None

def make_complaints_map(complaints):
    beat_boundaries = pd.read_csv("../data/PoliceBeatDec2012.csv")
    beat_boundaries.crs = "EPSG:4326"
    beat_boundaries.rename(columns={"BEAT_NUM":"beat"}, inplace=True)
    complaints['beat'] = pd.to_numeric(complaints['beat'], errors='coerce')
    complaints["complaint_date"]= pd.to_datetime(complaints["complaint_date"])
    complaints = complaints[complaints['complaint_date'].dt.year > 2011]
    complaints = complaints[complaints['complaint_date'].dt.year < 2015]
    complaints2 = complaints.groupby('beat').size().reset_index(name='counts').sort_values(['counts'], ascending=False)

    merged_gdf = (gpd.GeoDataFrame(beat_boundaries.merge(complaints2[['beat','counts']], on="beat", how="inner"), crs = beat_boundaries.crs))
    merged_gdf.rename(columns={"the_geom":"geometry"}, inplace=True)
    merged_gdf['geometry'] = merged_gdf['geometry'].apply(wkt.loads)
    my_geo = gpd.GeoDataFrame(merged_gdf, geometry='geometry')
    my_geo = my_geo[my_geo['beat'] != 3100]

    variable = 'counts'
    vmin, vmax = 0, 1709
    fig, ax = plt.subplots(1, figsize= (20, 12))
    my_geo.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')
    ax.set_title('COMPLAINTS LEVELED AGAINST CHICAGO POLICE OFFICERS PER BEAT \n Jan. 1, 2012 to Dec. 31, 2014', fontdict={'fontsize': '18', 'fontweight' : '3'})
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A=[]
    cbar = fig.colorbar(sm)
    plt.axis('off')
    plt.savefig('map_per_beat', bbox_inches='tight' )

def make_complaints_leveled(complaints_t1):
    complaints_binned = feat_engineering_helpers.complaint_bins(complaints_t1)
    complaints_binned_g = complaints_binned.groupby('complaints_binned').size().reset_index(name='counts').sort_values(['counts'], ascending=True)
    complaints_binned_g['complaints_pct'] = complaints_binned_g['counts'] / complaints_binned_g['counts'].sum()

    plt.figure(figsize=(15,10), dpi = 80)

    complaints_binned_g['labels'] = complaints_binned_g['complaints_binned'].str.title() + '\n (' + (complaints_binned_g['complaints_pct']*100).round(2).map('{}%'.format) +')' 
    norm = matplotlib.colors.Normalize(vmin=min(complaints_binned_g['complaints_pct']), vmax=max(complaints_binned_g['complaints_pct']))
    colors = [matplotlib.cm.Blues(norm(value)) for value in complaints_binned_g['complaints_pct']]
    squarify.plot(sizes=complaints_binned_g['complaints_pct'], label= complaints_binned_g['labels'], color = colors, alpha=1)
    plt.title('COMPLAINTS LEVELED AGAINST CHICAGO POLICE OFFICERS PER COMPLAINT TYPE \n Total number of complaints: {:,} \n Jan. 1, 2012 to Dec. 31, 2014'.format(complaints_binned_g['counts'].sum()), fontsize=15)
    plt.axis('off')
    plt.savefig('treemap_per_complaint', bbox_inches='tight' )

def make_complaints_by_officer(complaints_t1):
    complaints_per_officer = complaints_t1.groupby('UID').size().reset_index(name='counts').sort_values(['counts'], ascending=False)
    x = complaints_per_officer.groupby('counts').size().reset_index(name='num_complaints').sort_values(['num_complaints'], ascending=False)
    x['pct'] = x['num_complaints'] / x['num_complaints'].sum()
    x = x[x['counts']<=3]
    x = x.append({'counts': '4+', 'num_complaints':1182, 'pct': 0.174749}, ignore_index=True)

    category_names = list(x['counts'].unique())
    results = {'Percentage \n Complaints' : x['pct'].unique()}

    labels = list(results.keys())
    data = np.array (list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('Blues')(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            labelm = '{}%'.format(round(c*100, 2)) 
            ax.text(x, y, str(labelm), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    plt.title("COMPLAINTS LEVELED AGAINST CHICAGO PER POLICE OFFICER \n Share among officers who had at least one complaint \n Jan. 1, 2012 to Dec. 31, 2014 ") 
    plt.axis('off')
    plt.savefig('stackedbar_per_officer', bbox_inches='tight' )

