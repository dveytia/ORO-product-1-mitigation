#!/bin/ksh -vx
#SBATCH --verbose
#SBATCH --output="%x.out_%j"
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --mem=150GB
#SBATCH --time=48:00:00
### on rentre dans le repertoire de run


cd /home/dveytia/IPython_Notebooks/Product_1/analyses/02_multilabel_oro_type_2_mre

mpiexec -n 3 python 02_oro_type_2_mre_selection.py