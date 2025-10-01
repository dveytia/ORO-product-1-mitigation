#!/bin/ksh -vx
#SBATCH --verbose
#SBATCH --output="%x.out_%j"
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=120GB
#SBATCH --time=72:00:00
### on rentre dans le repertoire de run


cd /home/dveytia/IPython_Notebooks/Product_1/analyses/09_multilabel_ecosystem

mpiexec -n 5 python 09_multilabel_ecosystem_predictions.py