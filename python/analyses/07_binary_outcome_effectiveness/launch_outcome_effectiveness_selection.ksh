#!/bin/ksh -vx
#SBATCH --verbose
#SBATCH --output="%x.out_%j"
#SBATCH --nodes=1
#SBATCH --ntasks=15
#SBATCH --mem=150GB
#SBATCH --time=72:00:00
### on rentre dans le repertoire de run


cd /home/dveytia/IPython_Notebooks/Product_1/analyses/07_binary_outcome_effectiveness

mpiexec -n 3 python 07_binary_outcome_effectiveness_selection.py