#!/bin/ksh -vx
#SBATCH --verbose
#SBATCH --output="%x.out_%j"
#SBATCH --nodes=1
#SBATCH --ntasks=15
#SBATCH --mem=150GB
#SBATCH --time=72:00:00
### on rentre dans le repertoire de run


cd /home/dveytia/IPython_Notebooks/Product_1/analyses/08_binary_codis_benefit

mpiexec -n 3 python 08_binary_codis_benefit_selection.py