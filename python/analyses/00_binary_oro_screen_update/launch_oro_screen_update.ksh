#!/bin/ksh -vx
#SBATCH --verbose
#SBATCH --output="%x.out_%j"
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=200GB
#SBATCH --time=168:00:00
### on rentre dans le repertoire de run


cd /home/dveytia/IPython_Notebooks/Product_1/analyses/00_binary_oro_screen_update

mpiexec -n 10 python 00_oro_screen_update_predictions.py