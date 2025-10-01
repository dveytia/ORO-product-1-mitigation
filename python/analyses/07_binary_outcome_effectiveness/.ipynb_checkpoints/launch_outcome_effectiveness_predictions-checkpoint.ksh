#!/bin/ksh -vx
#SBATCH --verbose
#SBATCH --output="%x.out_%j"
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=100GB
#SBATCH --time=168:00:00
### on rentre dans le repertoire de run


cd /home/dveytia/IPython_Notebooks/Product_1/analyses/07_binary_outcome_effectiveness

mpiexec -n 5 python 07_binary_outcome_effectiveness_prediction.py