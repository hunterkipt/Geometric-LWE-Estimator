#!/bin/bash -l

# Give the job a name and account designation
#SBATCH --job-name="geo_LWE_SCA_estimate"
#SBATCH --account=mc2
#SBATCH --partition=mc2

# Set the number of nodes (Physical Computers)
#SBATCH --nodes=1

# Set the number of cores needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Set the amount of memory required
#SBATCH --mem-per-cpu=8gb

# Set stdout/stderr
#SBATCH --output="geo_LWE_SCA_estimate.out.%j"
#SBATCH --error="geo_LWE_SCA_estimate.err.%j"

# Set expected wall time for the job (format = hh:mm:ss)
#SBATCH --time=24:00:00

# Set quality of service level (useful for obtaining GPU resources)
#SBATCH --qos=high

# Turn on mail notifications for job failure and completion
#SBATCH --mail-type=END,FAIL

## No more SBATCH commands after this point ##

# Load slurm modules (needed software)
# Source scripts for loading modules in bash
. /usr/share/Modules/init/bash
. /etc/profile.d/ummodules.sh

module add Python/3.7.6
module add sage

# Define and create unique scratch directory for this job
SCRATCH_DIRECTORY=/scratch0/${USER}/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

# Copy code to the scratch directory
REPOSITORY=Geometric-LWE-Estimator
cp -r ${SLURM_SUBMIT_DIR}/${REPOSITORY} ${SCRATCH_DIRECTORY}

# Run code

OUTDIR=out

# cd geometricLWE/validation
cd ${REPOSITORY}/framework
sage load-ntt-data-modified.sage experiments/small_test.json --attack-before-short

# Copy outputs back to home directory
cp -r ${OUTDIR} ${SLURM_SUBMIT_DIR}/${SLURM_JOBID}
cd ${SLURM_SUBMIT_DIR}
mv geo_LWE_SCA_estimate.out.${SLURM_JOBID} ${SLURM_JOBID}
mv geo_LWE_SCA_estimate.err.${SLURM_JOBID} ${SLURM_JOBID}


# Remove code files
rm -rf ${SCRATCH_DIRECTORY}

# Finish the script
exit 0
