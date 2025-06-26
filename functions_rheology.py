# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 11:05:53 2025

@author: inagle
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as pl
import pandas
import io
from scipy.ndimage import gaussian_filter1d

import codecs


#This function calculates the average for n samples. 
# It is first required to construct an array containing all the arrays of each sample.
# The function has as variables :
# 1) data that is an array containing the of n x l x c, with n the number of different samples to average,
# l the number or lines in the array, or number of measuring points, and c the number of columns 
# corresponding to the different variables measured in the rheology test. 
# 2) Title corresponds to the title of the column you would like to plot. 
# For example, Title = Storage Modulus or Title = Loss Modulus

        
def average (data,Title:str):
    # Title = Type + " Modulus"   
    #n number of samples averaged
    n = np.size(data,0)
    S = np.zeros(np.size(data,1))
    for i in range (n) :
        S = S + np.array(data[i][Title], dtype='float64')
    avg = S/n
    return avg 

def standard_deviation (data,Title:str):
    # Title = Type + " Modulus" 
    #n number of samples averaged
    n = np.size(data,0)
    #l number of lines or measuring points per sample
    l = np.size(data,1)
    Q = np.zeros(l)
    for i in range (n) :
        Q = Q + np.square(np.array(data[i][Title], dtype='float64')-average(data,Title))
    SD = np.sqrt(Q/n)
    return SD 


# Plots 
# Plot polymerization curve from data. Mean +/- SD. Data has to be an array of all polymerisation measurements, corresponding to a 12 x 1080 array. (1h30 polymerization and point every 5s)
# G' ( Title = 'Storage Modulus') or G" (Title = 'Loss Modulus') as a function of time.
# Curve and points of given color c, given shape s, and a given label l 
def plotPolymerisation(data,Title:str,l:str,c:str,s:str):
    time_pol = np.array(data[0]['Time'].str.replace(',', ''), dtype='float64') ## adding .str.replace(',', '') removed the commas
    # if np.size(data,0)==1:
    #     pl.plot(time_pol, np.array(data[0][Title],dtype='float64'), ls='-', color = c, label=l,marker=s, ms=1,linewidth=2)
    
    data_avg = average(data,Title)
    data_SD = standard_deviation(data,Title)
    pl.plot(time_pol, data_avg, ls='-', color = c, label=l,marker=s, ms=1, linewidth = 3)
    pl.errorbar(time_pol, data_avg, ls = 'None', yerr = data_SD, ecolor=c,elinewidth=0.05)
    
    pl.xlabel(r't (s)', fontsize=30)
    if Title == 'Storage Modulus':
        pl.ylabel(r'$G^{\prime}$ $(Pa)$', fontsize=30)
        pl.ylim (0,150)
    elif Title == 'Loss Modulus' :
        pl.ylabel(r'$G^{\prime\prime}$ $(Pa)$', fontsize=30) 
        pl.ylim (0,30)
        
    for spine in pl.gca().spines.values():
        spine.set_linewidth(1.5) 
        
    pl.xticks(fontsize=24)
    pl.yticks(fontsize=24)
    pl.tick_params(axis='both', which='major', length=8, width=2)
    pl.legend(loc='upper left',fontsize=20, frameon=False) ##add ncol = 2 for MDAMB
    # pl.tight_layout()
    pl.xlim(0,6000)
    
# Plots 
#Plot polymerization curve from data. Plots all the individual curves.
#Data has to be an array of all polymerisation measurements, corresponding to a 12 x 1080 array. (1h30 polymerization and point every 5s)
#G' ( Title = 'Storage Modulus') or G" (Title = 'Loss Modulus') as a function of time.
#Curve and points of given color c, given shape s, and a given label l 
def plotPolymerisation_all(data,Title:str,l:str,c:str,s:str):
    time_pol = np.array(data[0]['Time'].str.replace(',', ''), dtype='float64') ## adding .str.replace(',', '') removed the commas
    
    for i in range (np.size(data,0)):
        pl.plot(time_pol, np.array(data[i][Title], dtype='float64'), color = c,marker=s, ms=1,linewidth = 1.5)
        
    if Title == 'Storage Modulus':
        pl.xlabel(r'$t$ $(s)$', fontsize=30)
        pl.tight_layout()
        pl.ylabel(r'$G^{\prime}$ $(Pa)$', fontsize=30)
        pl.ylim (0,150)
       
    elif Title == 'Loss Modulus' :
        pl.ylabel(r'$G^{\prime\prime}$ $(Pa)$', fontsize=30) 
        pl.ylim (0,30)
        
    for spine in pl.gca().spines.values():
        spine.set_linewidth(1.5)
        
    pl.xticks(fontsize=24)
    pl.yticks(fontsize=24)
    pl.legend(loc='upper left',fontsize=20, frameon=False)
    pl.tick_params(axis='both', which='major', length=8, width=2)
    pl.xlim(0,6000) 


# Plots 
#Returns the final value of G' of G'' at the end of polymerization. Data has to be an array of all polymerisation measurements, corresponding to a 12 x 1080 array. (1h30 polymerization and point every 5s)
#G' ( Title = 'Storage Modulus') or G" (Title = 'Loss Modulus')
#Curve and points of given color c, given shape s, and a given label l 
def finalPol(data,Title:str):
    time_pol = np.array(data[0]['Time'].str.replace(',', ''), dtype='float64') ## adding .str.replace(',', '') removed the commas
    # if np.size(data,0)==1:
    #     pl.plot(time_pol, np.array(data[0][Title],dtype='float64'), ls='-', color = c, label=l,marker=s, ms=1,linewidth=2)
    
    data_avg = average(data,Title)
    final_value = data_avg[len(data_avg)-1]
    #final value of the Storage of teh Loss modulus 
    return final_value
      
# Plots 
#Plot polymerization normalized curve from data1 (mean and SD). Normalized data by the average final G' or G" value of data2. Data has to be an array of all polymerisation measurements, corresponding to a 12 x 1080 array. (1h30 polymerization and point every 5s)
#G' ( Title = 'Storage Modulus') or G" (Title = 'Loss Modulus') as a function of time.
#Curve and points of given color c, given shape s, and a given label l 
def plotPolymerisation_norm(data1,data2,Title:str,l:str,c:str,s:str):
    time_pol = np.array(data1[0]['Time'].str.replace(',', ''), dtype='float64') ## adding .str.replace(',', '') removed the commas
    # if np.size(data,0)==1:
    #     pl.plot(time_pol, np.array(data[0][Title],dtype='float64'), ls='-', color = c, label=l,marker=s, ms=1,linewidth=2)
    
    data_avg = average(data1,Title)
    data_SD = standard_deviation(data1,Title)
    data_avg_norm = data_avg/finalPol(data2, Title)
    data_SD_norm = data_SD/finalPol(data2, Title)
    
    pl.plot(time_pol, data_avg_norm, ls='-', color = c, label=l,marker=s, ms=1, linewidth = 3)
    pl.errorbar(time_pol, data_avg_norm, ls = 'None', yerr = data_SD_norm, ecolor=c,elinewidth=0.05)
    
    pl.xlabel(r'$t$ $(s)$', fontsize = 30)
    if Title == 'Storage Modulus':
        pl.ylabel(r'$G^{\prime}$ / $G^{\prime}_{final}$ (-)', fontsize=30)
        pl.ylim (0,1.2)
    elif Title == 'Loss Modulus' :
        pl.ylabel(r'$G^{\prime\prime}$/$G^{\prime\prime}_{final}$ (-)', fontsize = 30)
        pl.ylim (0,1.2)

    for spine in pl.gca().spines.values():
            spine.set_linewidth(1.5)
        
        
    pl.tick_params(axis='both', which='major', length=8, width=2)
    pl.xticks(fontsize=24)
    pl.yticks(fontsize=24)
        
    
    pl.legend(loc='lower right',fontsize=20, frameon=False)
    # pl.tight_layout()
    pl.xlim(0,6000)    


def plotPolymerisation_normpeak(data, Title: str, l: str, c: str, s: str):
    # adding .str.replace(',', '') removed the commas
    time_pol = np.array(data[0]['Time'].str.replace(',', ''), dtype='float64')
    # if np.size(data,0)==1:
    #     pl.plot(time_pol, np.array(data[0][Title],dtype='float64'), ls='-', color = c, label=l,marker=s, ms=1,linewidth=2)
    data_norm = []
    for i in range(np.size(data, 0)):
        data_norm.append(np.array(
            data[i][Title], dtype='float64')/max(np.array(data[i][Title], dtype='float64')))
        # Renormalizing each polymerization curve bu the maximal G' or G" value
    data_avg_norm = np.average(data_norm, axis=0)
    data_SD_norm = np.std(data_norm, axis=0)

    pl.plot(time_pol, data_avg_norm, ls='-', color=c,
            label=l, marker=s, ms=1, linewidth=3)
    pl.errorbar(time_pol, data_avg_norm, ls='None',
                yerr=data_SD_norm, ecolor=c, elinewidth=0.05)
    
    pl.xticks(fontsize=24)
    pl.yticks(fontsize=24)
    pl.xlabel(r'$t$ $(s)$',fontsize=30)
    if Title == 'Storage Modulus':
        pl.ylabel(r'$G^{\prime}$/$G^{\prime}_{\text{max}}$(-)',fontsize=30)
        pl.ylim(0, 1.2)
    elif Title == 'Loss Modulus':
        pl.ylabel(r'$Normalized G^{\prime\prime}$',fontsize=30)
        pl.ylim(0, 1.2)
        
    for spine in pl.gca().spines.values():
                spine.set_linewidth(1.5)
                
    pl.legend(loc='lower right',fontsize=20, frameon=False)
    # pl.tight_layout()
    pl.xlim(0, 6000)    
    

def plotFreqSweep(data, Title: str, l: str, c: str, s: str):
    data_avg = average(data, Title)
    data_SD = standard_deviation(data, Title)
    freq = np.array(data[0]['Frequency'], dtype='float64')

    marker_style = 'v' if Title == 'Loss Modulus' else s

    pl.plot(freq[0:6], data_avg[0:6], ls='-', color=c, marker=marker_style, ms=8, linewidth=3)
    pl.errorbar(freq[0:6], data_avg[0:6], ls='None', yerr=data_SD[0:6], ecolor=c, elinewidth=0.8)
    pl.loglog()
    pl.xlabel(r'$f$ $(Hz)$', fontsize=30)

    pl.ylabel(r'$G^{\prime}$, $G^{\prime\prime}$ $(Pa)$', fontsize=30)

    # Show legend with only the colored line for Storage Modulus
    if Title == 'Storage Modulus':
        pl.plot([], [], ls='-', color=c, label=l, linewidth=3)
        pl.legend(loc='lower right', fontsize=20, frameon=False)

    for spine in pl.gca().spines.values():
        spine.set_linewidth(1.5)

    pl.xticks(fontsize=24)
    pl.yticks(fontsize=24)
    pl.tick_params(axis='both', which='major', length=8, width=2)  # Make ticks longer and thicker
    pl.tick_params(axis='both', which='minor', length=6, width=2)  # Adjust minor ticks

    # pl.tight_layout()
    pl.ylim(1, 150)
    
#Plot stress ramp from data. Strain as a function of the applied stress.
#Curve and points of given color c, given shape s, and a given label l    
def plotStressRamp (data,l:str,c:str, s:str):
    data_avg = average(data,'Strain')
    data_SD = standard_deviation(data,'Strain')
    stress = np.array(data[0]['Shear Stress'],dtype='float64')
    pl.plot(data_avg,stress,ls='-', color = c, label=l,marker=s, ms=6, linewidth = 3)
    # pl.errorbar(data_avg,stress, ls = 'None', xerr = data_SD, ecolor=c,elinewidth=0.8) 
    
    # Plot the shaded x-error region
    x_lower = data_avg - data_SD
    x_upper = data_avg + data_SD

    # Construct polygon points
    x_poly = np.concatenate([x_lower, x_upper[::-1]])
    y_poly = np.concatenate([stress, stress[::-1]])

    pl.fill(x_poly, y_poly, color=c, alpha=0.1, edgecolor='none')
   
    # Darker edge lines (left and right sides of the shaded band)
    pl.plot(x_lower, stress, color=c, alpha=0.4, linewidth=1.5)
    pl.plot(x_upper, stress, color=c, alpha=0.4, linewidth=1.5) 
   
    pl.xlabel(r'$\gamma$ (%)', fontsize=30)
    pl.ylabel(r'$\sigma$ $(Pa)$', fontsize=30)
    
    for spine in pl.gca().spines.values():
        spine.set_linewidth(1.5)
    
    pl.xticks(fontsize=24)
    pl.yticks(fontsize=24)
    pl.tick_params(axis='both', which='major', length=8, width=2)
    # pl.legend(loc='lower right', fontsize=20, frameon=False)
    pl.tight_layout()
    pl.xlim(0,40)
    pl.ylim(0,40)
    
    
#Plot differential modulus from data as a function of the stress (or strain). Mean (+/- SD).
#The curve is truncated at the rupture point.
#Curve and points of given color c, given shape s, and a given label l    
def plotDiffMod(data, l: str, c: str, s: str):
    all_diffMod = []
    all_stress = []
    
    # Calculate the differential modulus (K) for each dataset
    for dataset in data:  
        strain = np.array(dataset['Strain'], dtype='float64')
        stress = np.array(dataset['Shear Stress'], dtype='float64')

        # Compute differential modulus (K)
        diffMod = np.gradient(stress, strain / 100.0)
        
        all_diffMod.append(diffMod)
        all_stress.append(stress)

    # Convert lists to arrays for averaging
    all_diffMod = np.array(all_diffMod)
    all_stress = np.array(all_stress)

    # Ensure no NaN values in the datasets
    if np.any(np.isnan(all_diffMod)) or np.any(np.isnan(all_stress)):
        print("Warning: NaN values detected in the data.")
        all_diffMod = np.nan_to_num(all_diffMod)
        all_stress = np.nan_to_num(all_stress)

    # Compute average differential modulus and stress
    avg_diffMod = np.mean(all_diffMod, axis=0)
    avg_stress = np.mean(all_stress, axis=0)
    
    # Compute standard deviation of stress and differential modulus
    std_diffMod = np.std(all_diffMod, axis=0)
    std_stress = np.std(all_stress, axis=0)

    # Search for the peak starting from 1 Pa onwards
    # Find the first index where stress is greater than 1 Pa
    start_index = np.argmax(avg_stress > 1)

    # Ensure the peak index is found in the region of interest
    peak_index_avg = np.argmax(avg_diffMod[start_index:]) + start_index

  
   # Truncate the data at the peak index (corresponding to the rupture point) for the average curve
    avg_diffMod_truncated = avg_diffMod[:peak_index_avg + 1]
    avg_stress_truncated = avg_stress[:peak_index_avg + 1]
    std_diffMod_truncated = std_diffMod[:peak_index_avg + 1]
    std_stress_truncated = std_stress[:peak_index_avg + 1]

    # Plot only the average curve up to the peak
    pl.plot(avg_stress_truncated, avg_diffMod_truncated, ls='-', color=c, label=l, marker=s, ms=6, linewidth=3)
    
    # Plot the standard deviation as a shaded region around the average curve
    pl.fill_between(avg_stress_truncated, avg_diffMod_truncated - std_diffMod_truncated, avg_diffMod_truncated + std_diffMod_truncated, color=c, alpha=0.1)

     
    # Draw the edges with a darker color
    pl.plot(avg_stress_truncated, avg_diffMod_truncated - std_diffMod_truncated, color=c, alpha=0.2, linewidth=1.5)
    pl.plot(avg_stress_truncated, avg_diffMod_truncated + std_diffMod_truncated, color=c, alpha=0.5, linewidth=1.5)
    
    for spine in pl.gca().spines.values():
        spine.set_linewidth(1.5)
        
    pl.xlabel(r'$\sigma$ $(Pa)$', fontsize=30)
    pl.ylabel(r'$K$ $(Pa)$', fontsize=30)
    pl.loglog()
    pl.xticks(fontsize=24)
    pl.yticks(fontsize=24)
    pl.legend(loc='lower right', fontsize=20, frameon=False)
    pl.tick_params(axis='both', which='major', length=8, width=2)
    pl.tick_params(axis='both', which='minor', length=6, width=2)
    pl.xlim(0.1, 100)
    pl.ylim(10, 300)


#Plot differential modulus from data as a function of the stress (or strain) for all individual curves. 
#The curve is truncated at the rupture poin.
#Curve and points of given color c, given shape s, and a given label l    
def plotDiffMod_all(data, l: str, c: str, s: str):
    all_diffMod = []
    all_stress = []
    all_strain = []
    
    for dataset in data:  # Assuming data is a list of dictionaries
        strain = np.array(dataset['Strain'], dtype='float64')
        stress = np.array(dataset['Shear Stress'], dtype='float64')

        # Calculate differential modulus for each dataset
        diffMod = np.gradient(stress, strain / 100.0)
        
        all_diffMod.append(diffMod)
        all_stress.append(stress)
        all_strain.append(strain)

    # Convert lists to arrays for easy averaging
    all_diffMod = np.array(all_diffMod)
    all_stress = np.array(all_stress)
    all_strain = np.array(all_strain)
    
    fig, ax = pl.subplots(figsize=(8, 6)) 
    
    for i in range(np.size(data, 0)):
        # Search for the peak starting from 1 Pa onwards
        start_index = np.argmax(all_stress[i] > 1)
        peak_index = np.argmax(all_diffMod[i][start_index:]) + start_index

        # Truncate the individual dataset at the peak
        truncated_diffMod = all_diffMod[i][:peak_index + 1]
        truncated_stress = all_stress[i][:peak_index + 1]

        # Plot the individual curve up to the peak
        pl.plot(truncated_stress, truncated_diffMod, ls='-', color=c, marker=s, ms=1.5, linewidth=1.5)
    for spine in pl.gca().spines.values():
        spine.set_linewidth(1.5)
        
    ax.set_xlabel(r'$\sigma$ $(Pa)$', fontsize=30)
    ax.set_ylabel(r'$K$ $(Pa)$', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    pl.xticks(fontsize=24)
    pl.yticks(fontsize=24)
    pl.loglog()
    pl.legend(loc='upper left', fontsize=20, frameon=False)
    pl.tick_params(axis='both', which='major', length=8, width=2)
    pl.tick_params(axis='both', which='minor', length=6, width=2)
    pl.xlim(0.1, 100)
    pl.ylim(10, 1000)


# This function returns the average value of the differential modulus of one condition (data containing all the arrays of each sample of the condition)
# at 0.1 Pa stress. (point 11 of the array)   
def DiffMod_init (data):
    all_diffMod = []
    all_stress = []
    all_strain = []
    
    for dataset in data:  # Assuming data is a list of dictionaries
        strain = np.array(dataset['Strain'], dtype='float64')
        stress = np.array(dataset['Shear Stress'], dtype='float64')

        # Calculate differential modulus for each dataset
        diffMod = np.gradient(stress, strain / 100.0)
        
        all_diffMod.append(diffMod)
        all_stress.append(stress)
        all_strain.append(strain)

    # Convert lists to arrays for easy averaging
    all_diffMod = np.array(all_diffMod)
    all_stress = np.array(all_stress)
    all_strain = np.array(all_strain)

    # Calculate average and standard deviation across datasets
    avg_diffMod = np.mean(all_diffMod, axis=0)
    diffMod_init = avg_diffMod[11]
    return diffMod_init 

#Plot differential modulus from data1 as a function of the stress. Differential modulus of data1 normalized by the average value at the initial value 0.1 Pa (point 11) of data 2.
#Curve and points of given color c, given shape s, and a given label l    
def plotDiffMod_norm(data1, data2, l: str, c: str, s: str):
    all_diffMod = []
    all_stress = []
    all_strain = []
    
    for dataset in data1:  # Assuming data is a list of dictionaries
        strain = np.array(dataset['Strain'], dtype='float64')
        stress = np.array(dataset['Shear Stress'], dtype='float64')
        
        # Calculate differential modulus for each dataset
        diffMod = np.gradient(stress, strain / 100.0)
        
        all_diffMod.append(diffMod)
        all_stress.append(stress)
        all_strain.append(strain)

    for spine in pl.gca().spines.values():
        spine.set_linewidth(1.5)

    # Convert lists to arrays for easier computations
    all_diffMod = np.array(all_diffMod)
    all_stress = np.array(all_stress)
    all_strain = np.array(all_strain)

    # Ensure no NaN values in the datasets
    if np.any(np.isnan(all_diffMod)) or np.any(np.isnan(all_stress)):
        print("Warning: NaN values detected in the data.")
        all_diffMod = np.nan_to_num(all_diffMod)
        all_stress = np.nan_to_num(all_stress)

    # Compute average differential modulus and stress
    avg_diffMod = np.mean(all_diffMod, axis=0)
    avg_stress = np.mean(all_stress, axis=0)
    
    # Normalize by K(0.1 Pa) of data2
    avg_diffMod_norm = avg_diffMod / DiffMod_init(data2)
    diffMod_SD_norm = np.std(all_diffMod, axis=0) / DiffMod_init(data2)

    # Search for the peak starting from 1 Pa onwards
    start_index = np.argmax(avg_stress > 1)  # Find the first index where stress > 1 Pa

    # Ensure the peak index is found in the region of interest
    peak_index_avg = np.argmax(avg_diffMod_norm[start_index:]) + start_index

    # Truncate the data at the peak index
    avg_diffMod_norm_truncated = avg_diffMod_norm[:peak_index_avg + 1]
    avg_stress_truncated = avg_stress[:peak_index_avg + 1]
    diffMod_SD_norm_truncated = diffMod_SD_norm[:peak_index_avg + 1]

    # Plot the average differential modulus
    pl.plot(avg_stress_truncated, avg_diffMod_norm_truncated, ls='-', color=c, label=l, marker=s, ms=6, linewidth=3)
    
    # Plot the shaded region for the standard deviation
    pl.fill_between(avg_stress_truncated, avg_diffMod_norm_truncated - diffMod_SD_norm_truncated, avg_diffMod_norm_truncated + diffMod_SD_norm_truncated, color=c, alpha=0.1)
   

    # Draw the edges with a darker color
    pl.plot(avg_stress_truncated, avg_diffMod_norm_truncated - diffMod_SD_norm_truncated, color=c, alpha=0.2, linewidth=1.5)
    pl.plot(avg_stress_truncated, avg_diffMod_norm_truncated + diffMod_SD_norm_truncated, color=c, alpha=0.5, linewidth=1.5)


    pl.xlabel(r'$\sigma$ $(Pa)$', fontsize=30) 
    pl.ylabel(r'$K$ / $K_{0}$ (-)', fontsize=30) 
    pl.tick_params(axis='both', which='major', length=8, width=2)
    pl.tick_params(axis='both', which='minor', length=6, width=2)
    pl.xticks(fontsize=24)
    pl.yticks(fontsize=24)
    
    pl.legend(loc='lower right', fontsize=20, frameon=False) ##for integrin plot ncol=2 and fontsize =15
    pl.loglog()
  
    pl.tight_layout()
    pl.xlim(0.1, 50)
    pl.ylim(0.1, 10)          
    
# Function that gives as an output the list of onset times for an array containing all the arrays of samples for one condition.
# The onset time is defined by the intersection between the tangent of the curve were the first derivative is maximal and the t axis. 
# The input arguments are "data" the array containing all the arrays of each sample of the condition and sigma the value used for the gaussian filtering.
# The function can plot the storage modulus as a function of time, the first derivative as a function of time, the second derivative as a function of time
# with the onset time point visible to verify the computation. 
def onset_time(data,sigma):
    #Defining the time axis
    time = np.array(data[0][:]['Time'], dtype='float64')
    dt = time[1]-time[0]
    # Initializing the list that will contain all teh calculated onset times
    onset_time = np.zeros(np.size(data,0))
    for i in range (np.size(data,0)) :
        #Load the data
        sample = np.array(data[i][:]['Storage Modulus'], dtype='float64')
        #Smooth the curve
        sample_smooth = gaussian_filter1d(sample, sigma)
        #Compute first derivative
        dy = np.gradient(sample_smooth)
        #Compute second derivative
        d2y = np.gradient(dy)
        # Find the max of the first derivative and the corresponding index ind_tM and time tM
        ind_tM = np.argmax(dy)
        tM = ind_tM *dt
        #Equation of the tangent to the curve where the first derivative is maximal.
        tang = dy[ind_tM]/dt*(time-tM)+sample_smooth[ind_tM]
        #Finding the intersection of that tangent with the abcissa axis (t axis). It gives a value t1 corresponding to the onset time.
        ind_t1=0
        for j in range (1,len(dy)):
            if tang [j-1]*tang[j]< 0:  # Check if the product of consecutive y values is negative
                ind_t1 = j-1
        t1 = ind_t1*dt
        onset_time[i] = t1
    
        # Plotting the curve with the tangent, its intersection to the t axis and with the tangent to the Plateau. 
        pl.figure(figsize=(8, 6))
        pl.plot(time,sample_smooth, lw = 3 )
        pl.plot(time,tang, 'k--', lw = 1)
        pl.title('Sample ' + str(i))
        # pl.plot(time_pol,tang2, 'k--', lw = 1)
        pl.plot(tM,sample_smooth[ind_tM],marker='o', color = 'y', markersize = 12)
        pl.plot(t1,0,marker='o', color = 'g',markersize = 12)
        # pl.plot(t2,tang[ind_t2],marker='o', color = 'r')
        pl.ylim(0,120)
        pl.xlabel(r'$t$ $(s)$',fontsize = 26, labelpad=15)
        pl.ylabel(r'$G^{\prime}$ $(Pa)$',fontsize = 26, labelpad=15)
        for spine in pl.gca().spines.values():
            spine.set_linewidth(1.5) 
        pl.xticks(fontsize=22)
        pl.yticks(fontsize=22)
        pl.tick_params(axis='both', which='major', length=8, width=2)
      
        # # Plotting the first derivative and the max of the first derivative at tM
        pl.figure(figsize=(8, 6))
        pl.plot(time,dy,lw = 3)
        pl.plot(tM,dy[ind_tM],marker='o', color = 'y',markersize = 12)
        pl.xlabel(r'$t$ $(s)$',fontsize = 26, labelpad=15)
        pl.ylabel(r'$\frac{\partial G^{\prime}}{\partial t}$  $(Pa/s)$',fontsize = 26, labelpad=15)
        for spine in pl.gca().spines.values():
            spine.set_linewidth(1.5) 
        pl.xticks(fontsize=22)
        pl.yticks(fontsize=22)
        pl.tick_params(axis='both', which='major', length=8, width=2)

        # # Plotting the second derivative and the max of the first derivative at tM corresponding to where the second derivative is equal to zero
        pl.figure(figsize=(8, 6))
        pl.plot(time,d2y)
        pl.plot(tM,d2y[ind_tM],marker='o', color = 'y')
        
    return onset_time

# Returns the strain and stress rupture values defined as the maximum of the differential modulus from one given condition.
# Data contains all the stress ramps of one condition.
def gamma_rupture(data):
    all_diffMod = []
    all_strain = []
    all_stress = []
    
    all_strain_rupture = []
    all_stress_rupture = []
    i = 0
    
    for dataset in data:  # Assuming each argument is a list of dictionaries
        strain = np.array(dataset['Strain'], dtype='float64')
        stress = np.array(dataset['Shear Stress'], dtype='float64')

        # Calculate differential modulus for each dataset
        diffMod = np.gradient(stress, strain / 100.0)
        diffMod_smooth = gaussian_filter1d(diffMod, 2)
            
        # Finding the maximum of the differential modulus curve and the corresponding index from 0.2 Pa, corresponding to 14th point
        ind = 14
        for j in range(ind, np.size(diffMod_smooth)):
            if diffMod_smooth[j-1] < diffMod_smooth[j] :  # Finding the maximum
                ind = j
           
        strain_rupture = strain[ind]
        stress_rupture = stress[ind]
           
        all_strain_rupture.append(strain_rupture)
        all_stress_rupture.append(stress_rupture)
        
    return (all_strain_rupture, all_stress_rupture)