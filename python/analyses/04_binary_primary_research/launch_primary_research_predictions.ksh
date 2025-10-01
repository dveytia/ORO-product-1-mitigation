#!/bin/ksh -vx
#SBATCH --verbose
#SBATCH --output="%x.out_%j"
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=120GB
#SBATCH --time=168:00:00
### on rentre dans le repertoire de run


cd /home/dveytia/IPython_Notebooks/Product_1/analyses/04_binary_primary_research

mpiexec -n 5 python 04_binary_primary_research_prediction.py