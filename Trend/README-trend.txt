### Instructions for estimating the polynomial trend model to the Forstmann et al. (2008) data ###

Before estimating the time-varying models, we recommend exploring code that was provided in previous publications that form components of the estimation approach for time-varying models: 
1. Estimating model parameters via Particle Metropolis within Gibbs (PMwG) algorithm (osf.io/5b4w3).
2. Estimating the marginal likelihood for cognitive models via the IS^2 method (osf.io/xv59c). 

* LBA_PMwG_Markov_deterministic_v1.m : This is the primary file to run the method. Start here. This file loads the data as required:
   -> LBA_realdata_Forstmann_block.mat : Data from Forstmann et al. (2008) in format expected by the algorithm, separated by blocks in the experiment. 

When applying the time-varying models to different data sets, the following scripts will require editing depending on the design of the new data set and the model that is estimated. The main file (above) will also require editing. 
* LBA_CSMC_prior_rw_deterministic.m
* LBA_CSMC_prior_prop_deterministic.m
* reshape_b.m
* reshape_v.m

The remaining files are subsidiary Matlab functions. These don't require editing for application to different data sets.
