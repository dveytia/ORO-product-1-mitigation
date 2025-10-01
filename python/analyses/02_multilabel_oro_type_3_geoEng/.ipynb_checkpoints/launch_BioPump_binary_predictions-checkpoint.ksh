#!/bin/ksh -vx
#SBATCH --verbose
#SBATCH --output="%x.out_%j"
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem=120GB
#SBATCH --time=168:00:00
### on rentre dans le repertoire de run


cd /home/dveytia/IPython_Notebooks/Product_1/analyses/02_multilabel_oro_type_3_geoEng

mpiexec -n 5 python 02_binary_BioPump_predictions.py