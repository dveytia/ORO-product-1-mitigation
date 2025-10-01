#!/bin/ksh -vx
#SBATCH --verbose
#SBATCH --output="%x.out_%j"
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem=150GB
#SBATCH --time=24:00:00
### on rentre dans le repertoire de run


cd /home/dveytia/IPython_Notebooks/Product_1/analyses/02_multilabel_oro_type_1

mpiexec -n 1 python 02_oro_type_1_model_selection.py