# functions_rheology

This code contains all the functions used for the analysis and plotting of rheological data from the manuscript : 
Invasive cancer cells soften collagen networks and disrupt stress-stiffening via volume exclusion, contractility and adhesion (https://www.biorxiv.org/content/10.1101/2025.04.11.648338v1)

In general, the functions described take as an input “data” that corresponds to an array of samples of n x l x c size, with n the number of different samples, l the number of measuring points of the rheological tests and c the number of columns corresponding to the measured variable with its name as the column title (for example “Time”, “Storage Modulus”, “Strain” or “Shear Stress”). 

Most functions included are used for direct plotting of the rheological variables measured. More details related to each function can be found as comments in the code. 
Two custom-built functions were written:

o	The function “onset_time” measures the network polymerization onset time from time sweeps (storage modulus G′ (t) versus time t) acquired during collagen polymerization.
The onset time is defined as the intersection between the tangent to the curve’s inflection point and the time axis, with the inflection point determined as the maximum of the first derivative of the storage modulus with respect to time. Before computing the derivatives, the curves are smoothed using the function gaussian filter1d on a window “sigma”. The function “onset_time” takes as an input the array “data” containing all the time sweeps of individual samples for one given condition and the smoothing window “sigma”. The output is an array of all onset times measured for the different polymerization curves of each sample. 

o	The function “plotDiffMod” plots the differential modulus K as a function of stress 
calculated from the local tangent to the stress-strain curves, K = dσ/dγ, acquired in the stress ramp experiments. The stress-strain curves are first smoothed using the function gaussian filter1d from Python. Plots of the differential modulus as a function of stress are truncated at the rupture point, identified as the point where K reached its maximum value. The function “plotDiffMod” takes as an input “data”, containing all the stress ramps of individual samples for one given condition. The output of the function is a plot of the averaged differential modulus as a function of the stress for all samples contained in the array “data”. 


