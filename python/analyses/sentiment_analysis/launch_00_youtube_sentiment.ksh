#!/bin/ksh -vx
#SBATCH --verbose
#SBATCH --output="%x.out_%j"
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
### on rentre dans le repertoire de run

cd /home/dveytia/IPython_Notebooks/Product_1/analyses/sentiment_analysis

python 00_youtube_sentiment_attribution.py