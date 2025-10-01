import os
import json
import pandas as pd
import glob
from pathlib import Path
import functools 
import operator

## Constant variables
dataFolder = '/homedata/dveytia/Product_1_data'
bluesky_pred = f'{dataFolder}/outputs/sentiment_predictions/bluesky_sentiments.jsonl'
reddit_pred = f'{dataFolder}/outputs/sentiment_predictions/reddit_sentiments.jsonl'
youtube_pred = f'{dataFolder}/outputs/sentiment_predictions/youtube_sentiments.jsonl'
linkedin_pred = f'{dataFolder}/outputs/sentiment_predictions/linkedin_sentiments.jsonl'
input_dirs = [bluesky_pred,reddit_pred,youtube_pred, linkedin_pred]

# test = pd.read_json(path_or_buf=bluesky_pred, lines=True)
# test = test[["oro_type","source","post_date","up_count","repost_count","post_sentiment","sentiment_score"]]
# print(test.head())


# Generator to stream posts from files without loading all at once
# modify to filter to only the posts that have sentiments

def stream_posts(input_dirs):
    if isinstance(input_dirs, (str, Path)):
        input_dirs = [input_dirs]

    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        for file_path in glob.glob(str(input_dir / "*_sentiments.jsonl")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        post = json.loads(line)
                        if not post.get("skipped"):
                            yield post
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

all_posts = list(stream_posts(f'{dataFolder}/outputs/sentiment_predictions'))
posts_df = pd.DataFrame(all_posts)

# Some cleaning
posts_df.loc[posts_df['source']=='bluesky', 'post_type'] = "bluesky post"
posts_df.loc[posts_df['oro_type']=='Incr', 'oro_type'] = "Incr_eff" 
posts_df.loc[posts_df['oro_type']=='MRE', 'oro_type'] = "MRE_general"
posts_df.loc[posts_df['oro_type']=='CDR', 'oro_type'] = "CDR_general"

def map_values(x):
    value_map = {
        "comment": "reddit comment",
        "submission": "reddit post",
        "bluesky post": "bluesky post",
        "youtube comment": "youtube comment",
        "linkedin post": "linkedin post",
    }
    return value_map.get(x, "NaN")

posts_df['post_type']=posts_df['post_type'].apply(map_values)
posts_df.post_type.value_counts()

intCols = ["up_count", "repost_count", "down_count"]
for col in intCols:
    posts_df[col] = posts_df[col].apply(pd.to_numeric, errors='coerce')
    posts_df[col] = posts_df[col].fillna(0).astype(int)


# Ensure there are no duplicates within ORO type and post type
posts_df = posts_df.drop_duplicates(
  subset = ['oro_type', 'post_type','post_id'],
  keep = 'last').reset_index(drop = True)
    
    
## Calculate weight per post.
"""
weight can be up_count + repost_count. Decided not to subtract down_count because it's only found in reddit so not really comparable....
"""
posts_df['like_weight'] = posts_df["up_count"]+posts_df["repost_count"] #-posts_df["down_count"]
posts_df['like_weight'] = posts_df['like_weight'].apply(lambda x: x if x > 0 else 1) 

## Also add another weight for the number of search keywords used for each oro_type

# Original search query:
qrys = {
    'MRE_located': ["offshore wind", "offshore solar","offshore photovoltaic"],
    'MRE_ocean': ["wave energy","tidal energy","ocean current energy","ocean geothermal energy","thermohaline energy","OTEC","salinity gradient renewable energy", "ocean geothermal energy"],
    'MRE_bio':['seaweed biofuel',"algae biofuel", "kelp biofuel","phytoplankton biofuel"],
    'MRE_general': ['marine renewable energy'],
    'Incr_eff': ["decarbonize shipping","decarbonize maritime industry","increase ship efficiency","reduce ship emissions"],
    'CCS': ['store captured carbon in seabed','seabed carbon storage', 'seabed CCS','deep sea CCS','ocean CCS'],
    
    'CDR_bc': functools.reduce(operator.iconcat, [['blue carbon'], 
                                                  [item + " mitigate climate change" for item in ['mangrove','saltmarsh','seagrass','seaweed','kelp','tidal marsh']], 
                                                  [item + " carbon sequestration" for item in ['mangrove','saltmarsh','seagrass','seaweed','kelp','tidal marsh']]], []),
    'CDR_oae': [item + " carbon removal" for item in ['ocean alkalinity enhancement','ocean liming','ocean weathering']],
    'CDR_biopump':['ocean iron fertilization','artificial upwelling carbon removal', 'ocean biological carbon pump carbon removal'],
    'CDR_cult': functools.reduce(operator.iconcat, [['ocean afforestation', 'seaweed CDR', 'kelp CDR'], [f'sink {item} carbon removal' for item in ['seaweed','kelp','macroalgae']]], []), 
    'CDR_general':['marine CDR','marine carbon removal','ocean CDR','ocean carbon remvoal']
    }

# get the qry length for each oro type, and make 1/length to get the weight
qry_weights = {k: 1/(len(v)) for k, v in qrys.items()}

# Map these values to create a new column in posts_df
posts_df['qry_weight']=posts_df['oro_type'].apply(lambda x: qry_weights.get(x, 0.0))

# Then calculate combined weight
posts_df['weight'] = posts_df['like_weight']*posts_df['qry_weight']



## Calculate summaries per oro type:
## n_posts, n_positive, n_neutral, n_negative, n_positive_weighted, n_negative_weighted, n_neutral_weighted. 

summary_posts = posts_df.groupby('oro_type').agg(
    n_posts=('post_id', 'count'),
    n_posts_qry_weighted = ("qry_weight","sum"),
    n_posts_like_weighted = ("like_weight","sum"),
    n_posts_weighted=('weight', 'sum'),
    positive=('post_sentiment', lambda x: (x == 'positive').sum()),
    neutral=('post_sentiment', lambda x: (x == 'neutral').sum()),
    negative=('post_sentiment', lambda x: (x == 'negative').sum()),
    weighted_positive=('weight', lambda w: w[posts_df['post_sentiment'] == 'positive'].sum()),
    weighted_neutral=('weight', lambda w: w[posts_df['post_sentiment'] == 'neutral'].sum()),
    weighted_negative=('weight', lambda w: w[posts_df['post_sentiment'] == 'negative'].sum())
).reset_index()



## Calculate annual summaries as well
summary_year = posts_df.copy()
summary_year["post_date"] =  pd.to_datetime(summary_year["post_date"], errors='coerce')
summary_year["year"] = summary_year["post_date"].dt.year

# Add binary indicators and weighted sentiment columns

# Create sentiment indicator columns
summary_year["is_positive"] = (summary_year["post_sentiment"] == "positive").astype(int)
summary_year["is_neutral"]  = (summary_year["post_sentiment"] == "neutral").astype(int)
summary_year["is_negative"] = (summary_year["post_sentiment"] == "negative").astype(int)
summary_year["w_positive"] = summary_year["is_positive"] * summary_year["weight"]
summary_year["w_neutral"]  = summary_year["is_neutral"] * summary_year["weight"]
summary_year["w_negative"] = summary_year["is_negative"] * summary_year["weight"]

# Group by oro_type and year
summary_year = (
    summary_year
    .groupby(["oro_type", "year"])
    .agg(
        n_posts=("post_id", "nunique"),
        n_posts_qry_weighted = ("qry_weight","sum"),
        n_posts_like_weighted = ("like_weight","sum"),
        n_posts_weighted=('weight', 'sum'),
        positive=("is_positive", "sum"),
        neutral=("is_neutral", "sum"),
        negative=("is_negative", "sum"),
        weighted_positive=("w_positive", "sum"),
        weighted_neutral=("w_neutral", "sum"),
        weighted_negative=("w_negative", "sum"),
    )
    .reset_index()
)



######## save data frames #############
posts_df.to_pickle(f'{dataFolder}/outputs/sentiment_predictions/sentiment_allCompiled.pickle')
posts_df[['oro_type','source', 'post_id', 'post_body', 'post_date',
       'up_count', 'repost_count', 'post_sentiment', 'sentiment_score',
       'post_type','down_count', 'like_weight', 'qry_weight','weight']].to_csv(f'{dataFolder}/outputs/sentiment_predictions/sentiment_allCompiled_quantCols.csv', index=False)
summary_year.to_csv(f'{dataFolder}/outputs/sentiment_predictions/sentiment_summary_oro_year.csv', index=False)
summary_posts.to_csv(f'{dataFolder}/outputs/sentiment_predictions/sentiment_summary_oro.csv', index=False)





############ Plots #####################

import seaborn as sns
import matplotlib.pyplot as plt


summary_year = pd.read_csv(f'{dataFolder}/outputs/sentiment_predictions/sentiment_summary_oro_year.csv')
summary_posts = pd.read_csv(f'{dataFolder}/outputs/sentiment_predictions/sentiment_summary_oro.csv')



# Ensure we're using a clean style
sns.set(style="whitegrid")

custom_palette = {
    "positive": "#0571b0",
    "neutral": "grey",
    "negative": "#ca0020"
}



## Total number of weighted posts by ORO type
sns.barplot(summary_posts, y="oro_type", x="n_posts_weighted")

## Total number of wegihted posts over time
# Weighted: panel per oro_type
g = sns.FacetGrid(summary_year, col="oro_type", col_wrap=3, height=4, sharey=False)
g.map_dataframe(sns.barplot, x="year", y="n_posts_weighted",native_scale=True)
g.add_legend()
g.set_titles("{col_name}")
g.set_axis_labels("Year", "Weighted Count")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Annual posts by ORO Type (N Weighted)")
plt.savefig('/homedata/dveytia/Product_1_data/figures/tmp_npostsByYear.png')
plt.show()

## Distribution of sentiment per ORO type
df_bar = summary_posts[['oro_type','weighted_positive','weighted_neutral', 'weighted_negative']]
df_bar = df_bar.set_index('oro_type')
total = summary_posts.groupby('oro_type')['n_posts_weighted'].sum().reset_index()
total = total.set_index('oro_type')
df_bar = df_bar.apply(lambda x: [i / j * 100 for i,j in zip(x, total['n_posts_weighted'])])
ax = df_bar.plot.bar(stacked=True, color = {
    "weighted_positive": "#0571b0",
    "weighted_neutral": "grey",
    "weighted_negative": "#ca0020"
})
ax.figure.savefig('/homedata/dveytia/Product_1_data/figures/tmp_percentSentiment.png', bbox_inches='tight')





## By ORO type -----------------------------------

# Melt summary to long format for easy plotting
summary_melted = summary_posts.melt(
    id_vars="oro_type",
    value_vars=["positive", "neutral", "negative"],
    var_name="Sentiment",
    value_name="Count"
)

summary_weighted_melted = summary_posts.melt(
    id_vars="oro_type",
    value_vars=["weighted_positive", "weighted_neutral", "weighted_negative"],
    var_name="Sentiment",
    value_name="Count"
)
# Remove the 'weighted_' prefix for labeling
summary_weighted_melted["Sentiment"] = summary_weighted_melted["Sentiment"].str.replace("weighted_", "")

# Add 'Weighting' column to distinguish panels
summary_melted["Weighting"] = "Unweighted"
summary_weighted_melted["Weighting"] = "Weighted"

# Combine both DataFrames
combined_summary = pd.concat([summary_melted, summary_weighted_melted], ignore_index=True)

# Create FacetGrid with one panel per Weighting (Unweighted vs Weighted)
g = sns.FacetGrid(combined_summary, col="Weighting", height=6, aspect=1.2, sharey=False)
g.map_dataframe(sns.barplot, x="oro_type", y="Count", hue="Sentiment", palette=custom_palette)
g.set_titles("{col_name}")
g.set_axis_labels("ORO Type", "Count")
for ax in g.axes.flatten():
    ax.tick_params(axis='x', rotation=45)
g.add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Number of Posts by Sentiment per ORO Type")
plt.show()




## By ORO type and Year -----------------------------------

# Melt yearly summary for unweighted data
yearly_melted = summary_year.melt(
    id_vars=["oro_type", "year"],
    value_vars=["positive", "neutral", "negative"],
    var_name="Sentiment",
    value_name="Count"
)

# Ensure year is integer for sorting and tick control
yearly_melted["year"] = yearly_melted["year"].astype(int)

# Unweighted: panel per oro_type
g = sns.FacetGrid(yearly_melted, col="oro_type", col_wrap=3, height=4, sharey=False)
g.map_dataframe(sns.barplot, x="year", y="Count", hue="Sentiment", palette=custom_palette)
g.add_legend()
g.set_titles("{col_name}")
g.set_axis_labels("Year", "Count")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Yearly Sentiment Counts by ORO Type (Unweighted)")
plt.show()



# Melt yearly summary for weighted data
yearly_weighted_melted = summary_year.melt(
    id_vars=["oro_type", "year"],
    value_vars=["weighted_positive", "weighted_neutral", "weighted_negative"],
    var_name="Sentiment",
    value_name="Count"
)
yearly_weighted_melted["Sentiment"] = yearly_weighted_melted["Sentiment"].str.replace("weighted_", "")

# Weighted: panel per oro_type
g = sns.FacetGrid(yearly_weighted_melted, col="oro_type", col_wrap=3, height=4, sharey=False)
g.map_dataframe(sns.barplot, x="year", y="Count", hue="Sentiment", palette=custom_palette)
g.add_legend()
g.set_titles("{col_name}")
g.set_axis_labels("Year", "Weighted Count")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Yearly Sentiment Counts by ORO Type (Weighted)")
plt.show()


