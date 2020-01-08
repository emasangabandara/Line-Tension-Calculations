#!/usr/bin/env python2.7
#This script calculates the line tension in a phase separated membrane via capillary wave theory (CWT)
#Inputs are the set of lipid and cholesterol atom coordinates exist in a predifined 2D surface
#Utilzed in the line tension calculations published in J. Chem. Phys. 150, 204702 (2019); https://doi.org/10.1063/1.5091450

from __future__ import division
import numpy as np
import pylab as pl
from matplotlib import pyplot as plt
import math
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist,squareform
import scipy.interpolate as inter
from scipy.stats import binned_statistic
from scipy.stats import threshold
from pandas import rolling_median
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.fftpack import fftfreq
import os

import seaborn as sns
sns.set(style="ticks",context="poster", font_scale=1.5, rc={"lines.linewidth": 6})

#Work Directory
WRKDIR             ='.'

#Interface Dimension
interfacealong     = ['x']

#Number of Replicates
NREP               = [1]

#Test Snapshots
PRINT_INTERFACE    = 'NO' 
PRINT_FREQ         = 1

#Selecting low wave number modes based on mode index
miMAX = 4

print WRKDIR

LT_REP    = []
CWT_c_REP = []
low_k_modes_REP = []
power_spectrum_REP = []

for REP in NREP :
    print 'Analyzing Replicate # :' , REP

    REPLICATE          = REP

    
    if PRINT_INTERFACE == 'YES' :
        if not os.path.exists('%s\Interface-Test.%s'%(WRKDIR,str(REPLICATE))):
            os.makedirs('%s\Interface-Test.%s'%(WRKDIR,str(REPLICATE)))
    
    #Interface detection parameters
    #On average ordered and disordered lipids are in 33.3% contact with one another assuming hexagonal packing
    #Interface defined with Order-Disorder contacts (0.2 < mol_frac < 0.8) and Order-Protein contacts (cutoffs are further relaxed) 
    DPPC_mol_cut1, DPPC_mol_cut2  = 0.25, 0.55 #DIPC mol fractions
    DIPC_mol_cut1, DIPC_mol_cut2  = 0.25, 0.55 #DPPC mol fractions
    CHOL_mol_cut1, CHOL_mol_cut2  = 0.33, 0.66 #DPPC mol fractions
    
    PROT_mol_cut1, PROT_mol_cut2  = 0.2, 0.8 #DPPC mol fractions : with the observation that protein will not go in to the DPPC domain
    
    #Neighbour Cutoff in angstroms
    
    cutoff        = 30.0 # for lipids
    PROT_cutoff   = 30.0 # for proteins
    
    lipid1 = 'DPPC'
    lipid2 = 'DIPC'
    lipid3 = 'CHOL'
    #protein is hard coded
    
    ##
    parallelto = interfacealong[REP-1] # Interface parallel to x or y 
    ##
    
    ##
    #DBSCAN clustering parameters
    #May have to tune to the box size or system!
    DBSCAN_threshold= 50 #MEM,PROT : #The maximum distance (Angstroms) between two samples for them to be considered as in the same neighborhood.
    minimum_sample_fraction = 0.05 #The number of samples (or total weight) in a neighborhood for a point to be considered as a core point, as a fraction of total points. This includes the point itsef.
    ##
    #Cubic spline parameters
    splgrid = 100 #Number of grid points to construct the spline curve
    smooth     =0.001 #Continue spline computation with the given smoothing factor s and with the knots found at the last call
    
    #Filter parameters
    flterthreshold = 0.25 # threshold = flterthreshold*median
    
    KBT = 1.380*10**(-23)*295 #In Nm units
    
    def separate_Individual_Interface_DBSCAN_PBC_2d(points_xy) :
        #Set to aproximatly one tenth of total points
        minimum_samples = int(minimum_sample_fraction *len(points_xy))  
        #label = -1 for noise
        #Periodic boundry
        if parallelto == 'x' :
            L = YBOX
        if parallelto == 'y' :
            L = XBOX
            
        X = np.asarray(points_xy)
    
        #find the correct distance matrix
        for d in xrange(X.shape[1]):
            # find all 1-d distances
            pd=pdist(X[:,d].reshape(X.shape[0],1))
            # apply boundary conditions
            pd[pd>L*0.5]-=L
            try:
                # sum
                total+=pd**2
            except Exception, e:
                # or define the sum if not previously defined
                total=pd**2
        # transform the condensed distance matrix...
        total=pl.sqrt(total)
        # ...into a square distance matrix
        square=squareform(total)
        db=DBSCAN(eps=DBSCAN_threshold, min_samples=minimum_samples,metric='precomputed').fit(square)
        
        return db.labels_ #1D Arrey of cluster indices correspoinding to each point
    
    def extract_Individual_Interface(points_xy, labels) :
        interface1 = []
        interface2 = []
        for p in range(0,len(points_xy)) :
            if labels[p] == 0 :
                interface1.append(points_xy[p])
            if labels[p] == 1 :
                interface2.append(points_xy[p])
        return interface1, interface2
        
    def unwrapPBC(xy):
        if parallelto == 'x' :
            BOX      = YBOX
            points_1d   = np.hsplit(np.asarray(xy),2)[1]
            points_1d_sorted = sorted(points_1d)
            median   = np.median(points_1d_sorted, axis=0)
            diff     = np.abs(np.subtract(points_1d, median))
            #Unwrapping PBC
            PBC_unwrapped = []
            #Selecting interfaces closer to the boundry
            if median > 0.7*BOX : #high boundry
                for p in range(0,len(xy)) :
                    if  diff[p] > 0.5*BOX :
                        PBC_unwrapped.append((xy[p][0], xy[p][1] + BOX))  # unwrap low value points by adding a BOX length
                    else :
                        PBC_unwrapped.append((xy[p][0], xy[p][1]))
            elif median < 0.3*BOX : #low boundry
                for p in range(0,len(xy)) :
                    if  diff[p] > 0.5*BOX :
                        PBC_unwrapped.append((xy[p][0], xy[p][1] - BOX))  # unwrap high value points by subtracting a BOX length            
                    else :
                        PBC_unwrapped.append((xy[p][0], xy[p][1]))    
            #No unwrapping needed
            else :
                PBC_unwrapped = xy 
                
        if parallelto == 'y' :
            BOX      = XBOX
            points_1d   = np.hsplit(np.asarray(xy),2)[0]
            points_1d_sorted = sorted(points_1d)
            median   = np.median(points_1d_sorted, axis=0)
            diff     = np.abs(np.subtract(points_1d, median))
            #Unwrapping PBC
            PBC_unwrapped = []
            #Selecting interfaces closer to the boundry
            if median > 0.7*BOX : #high boundry
                for p in range(0,len(xy)) :
                    if  diff[p] > 0.5*BOX :
                        PBC_unwrapped.append((xy[p][0] + BOX, xy[p][1]))  # unwrap low value points by adding a BOX length
                    else :
                        PBC_unwrapped.append((xy[p][0], xy[p][1]))
            elif median < 0.3*BOX : #low boundry
                for p in range(0,len(xy)) :
                    if  diff[p] > 0.5*BOX :
                        PBC_unwrapped.append((xy[p][0] - BOX, xy[p][1]))  # unwrap high value points by subtracting a BOX length            
                    else :
                        PBC_unwrapped.append((xy[p][0], xy[p][1]))    
            #No unwrapping needed
            else :
                PBC_unwrapped = xy 
            
        return PBC_unwrapped
    def binInterfacepoints(arrey2d) :
        #binning interface points to get a smooth spline
        xy = np.asarray(arrey2d)
        if len(xy) > 3 :
            if parallelto == 'x' :
                bin_splmean, xb, binnum = binned_statistic(xy[:,0],xy[:,1],statistic=np.mean,bins=25)
                bin_gridmean, xb, binnum = binned_statistic(xy[:,0],xy[:,0],statistic=np.mean,bins=25)
            if parallelto == 'y' :
                bin_splmean, xb, binnum = binned_statistic(xy[:,1],xy[:,0],statistic=np.mean,bins=25)
                bin_gridmean, xb, binnum = binned_statistic(xy[:,1],xy[:,1],statistic=np.mean,bins=25)
            #removing nan values
            bin_gridmean     = bin_gridmean[~np.isnan(bin_gridmean)]
            bin_splmean      = bin_splmean[~np.isnan(bin_splmean)]
            #Impose PBC
            avg_fl           = 0.5*(bin_splmean[0] + bin_splmean[-1])
            bin_splmean[0]   = avg_fl
            bin_splmean[-1]  = avg_fl
        else :
            bin_gridmean     = []
            bin_splmean      = []
            
        return bin_gridmean, bin_splmean
    
    def employ_Cubic_Spline_and_get_Length(bin_gridmean, bin_splmean) :
    
        gridpoints = np.linspace(bin_gridmean.min(), bin_gridmean.max(), splgrid)
    
        #UnivariateSpline
        spl = inter.InterpolatedUnivariateSpline(bin_gridmean,bin_splmean)
        spl.set_smoothing_factor(smooth)
        spline = spl(gridpoints)
    
        distance = 0
        for count in range(1,len(spline)):
            distance += math.sqrt(math.pow(gridpoints[count]-gridpoints[count-1],2) + math.pow(spline[count]-spline[count-1],2))
        return spline, gridpoints, distance    	
    
    def filterOutliersWithMedian(data1,data2) :
        #Filtering is applied on data2
        #Filtering is based on data1
    
        median = np.median(np.asarray(data1))
        threshold = flterthreshold*median
        diff_pds= np.abs(np.asarray(data1) - median)
        filtered_data = []
        for index in range(0, int(len(data1))) :
            #print 'Spline Distance', data1[index]
            if diff_pds[index] < threshold:
                #print 'Accepted'
                filtered_data.append(data2[index])
                
        return np.asarray(filtered_data)
        
    def getLineDeviations(spline):
        #Calculate the deviations of the instantanious interface from its most stable (straight line with at spline mean) position
        #delatah(xi)
        linedeviations      = []
        splmean             = np.mean(spline)
        for spli in spline :
            if  spli >  splmean : # Positive deviation
                if splmean*spli > 0 :   #Same side of origin
                    hs = np.abs(np.abs(spli) - np.abs(splmean))
                elif splmean*spli < 0 : #Either side of origin     
                    hs = np.abs(spli) + np.abs(splmean)
                linedeviations.append(hs)
            elif  spli <  splmean : # Negative deviation
                if splmean*spli > 0 :   #Same side of origin
                    hs = np.abs(np.abs(spli) - np.abs(splmean))
                elif splmean*spli < 0 : #Either side of origin     
                    hs = np.abs(spli) + np.abs(splmean)
                linedeviations.append(-hs)
        return np.asarray(linedeviations) 
    
    def getComposition(frame_composition):
        nprot = 0
        nlpd1 = 0
        nlpd2 = 0
        nlpd3 = 0
        
        for componenti in frame_composition :
            if componenti == 'PROT' :
                nprot = nprot + 1
            elif componenti == lipid1 : 
                nlpd1 = nlpd1 + 1
            elif componenti == lipid2 : 
                nlpd2 = nlpd2 + 1
            elif componenti == lipid3 : 
                nlpd3 = nlpd3 + 1
                
        ntot        = nprot + nlpd1 + nlpd2 + nlpd3
        comp_frac   = [nprot/ntot , nlpd1/ntot , nlpd2/ntot , nlpd3/ntot, ntot] #Interface composition fraction in the order of PROT, lipid1, lipid2, lipid3 and total 
        return comp_frac
        
    def getColor(frame_composition):
        mol_color = []
        
        for componenti in frame_composition :
            if componenti == 'PROT' :
                mol_color.append('g') # PROT-Green
            elif componenti == lipid1 : 
                mol_color.append('b') # DPPC-Blue
            elif componenti == lipid2 : 
                mol_color.append('r') # DIPC-Red
            elif componenti == lipid3 : 
                mol_color.append('k') # CHOL-Black               

        return mol_color
            
    #Load processed frames
    ##
    
    sourcet = np.load(r'%s\upper_interface_data.R%s.npy'%(WRKDIR,str(REPLICATE)))
    sourceb = np.load(r'%s\lower_interface_data.R%s.npy'%(WRKDIR,str(REPLICATE)))
    
    memplanet = np.load(r'%s\upper_mem_plane_data.R%s.npy'%(WRKDIR,str(REPLICATE)))
    memplaneb = np.load(r'%s\lower_mem_plane_data.R%s.npy'%(WRKDIR,str(REPLICATE)))
    
    
    datat = sourcet[:int(len(sourcet))]
    datab = sourceb[:int(len(sourceb))]
    ##
    #Analysing frames
    frmt = 0
    frmb = 0	
    ens_interface_length_tl = [] #interface length of the top leaflet 
    ens_interface_length_bl = [] #interface length of the bottom leaflet
    
    ens_total_interface_length_tl = [] #total_interface length of the top leaflet 
    ens_total_interface_length_bl = [] #total_interface length of the bottom leaflet  
    
    ens_interface_composition_tl = [] #interface composition of the top leaflet 
    ens_interface_composition_bl = [] #interface composition of the bottom leaflet 
    
    ens_hk_hminusk_tl = [] #h(k)*deltah(-k) top leaflet 
    ens_hk_hminusk_bl = [] #(k)*deltah(-k) bottom leaflet 
    
    ens_FFT_tl = []
    ens_FFT_bl = []
    
    ens_BOX_tl = []
    ens_BOX_bl = []
    
    ens_AREA_tl = []
    ens_AREA_bl = []
    
    print 'Starting Analysis'
    
    for idx in range(0,int(len(datat))) :
        framet                    = datat[idx] #interface data
        memframet                 = memplanet[idx] #membrane plane data
        
        XBOX                      = framet[2]
        YBOX                      = framet[3]
        
        frame_tl                  = framet[0] #xy interface coordinates top leaflet
        frame_composition_tl      = framet[1] #interface compositiontop top leaflet
        
        memframe_tl                  = memframet[0] 
        memframe_composition_tl      = memframet[1] 
        
        comp_frac_tl              = getComposition(frame_composition_tl)
        ens_interface_composition_tl.append(comp_frac_tl)
        
        mol_color_tl              = getColor(memframe_composition_tl)
        
        ##
        if PRINT_INTERFACE == 'YES' and np.mod(idx,PRINT_FREQ) == 0 :
            #Plot detected interface
            figt, axt = pl.subplots()          
            axt.scatter(np.hsplit(np.asarray(memframe_tl),3)[0],np.hsplit(np.asarray(memframe_tl),3)[1],c=mol_color_tl,s=50,edgecolors='none')
        ##   
        
        labels_tl    = separate_Individual_Interface_DBSCAN_PBC_2d(frame_tl)
    
        #Extracting individual interfaces based on clustering
    
        intf1_tl , intf2_tl = extract_Individual_Interface(frame_tl,labels_tl)
    
        intf1_PBC_unwrapped_tl = unwrapPBC(intf1_tl)
        intf2_PBC_unwrapped_tl = unwrapPBC(intf2_tl)
        bin_gridmean1_tl, bin_splmean1_tl = binInterfacepoints(intf1_PBC_unwrapped_tl)
        bin_gridmean2_tl, bin_splmean2_tl = binInterfacepoints(intf2_PBC_unwrapped_tl)
        
        if len(bin_gridmean1_tl) > 3 and len(bin_gridmean2_tl) > 3 : 
            spline1_tl, gridpt1_tl, spldist1_tl = employ_Cubic_Spline_and_get_Length(bin_gridmean1_tl, bin_splmean1_tl)
            spline2_tl, gridpt2_tl, spldist2_tl = employ_Cubic_Spline_and_get_Length(bin_gridmean2_tl, bin_splmean2_tl)
    
            ##
            if PRINT_INTERFACE == 'YES' and np.mod(idx,PRINT_FREQ) == 0 :
                #Plot Interface and spline
                #pl.clf()
                axt.scatter(np.hsplit(np.asarray(intf1_PBC_unwrapped_tl),2)[0],np.hsplit(np.asarray(intf1_PBC_unwrapped_tl),2)[1],s=75,facecolor='none',linewidth='3',edgecolors='grey')
                if parallelto == 'x' :
                    axt.scatter(bin_gridmean1_tl,bin_splmean1_tl,c='violet',s=200,marker=(5, 1))
                    axt.plot(gridpt1_tl, spline1_tl, '-m', lw=4)
                elif parallelto == 'y' :
                    axt.scatter(bin_splmean1_tl,bin_gridmean1_tl,c='violet',s=200,marker=(5, 1))
                    axt.plot(spline1_tl,gridpt1_tl,'-m', lw=4)
                    
                axt.scatter(np.hsplit(np.asarray(intf2_PBC_unwrapped_tl),2)[0],np.hsplit(np.asarray(intf2_PBC_unwrapped_tl),2)[1],s=75,facecolor='none',linewidth='3',edgecolors='grey')
                if parallelto == 'x' :
                    axt.scatter(bin_gridmean2_tl,bin_splmean2_tl,c='violet',s=200,marker=(5, 1))
                    axt.plot(gridpt2_tl, spline2_tl, '-m', lw=4)
                elif parallelto == 'y' :
                    axt.scatter(bin_splmean2_tl,bin_gridmean2_tl,c='violet',s=200,marker=(5, 1))
                    axt.plot(spline2_tl,gridpt2_tl,'-m', lw=4)
                    
                axt.set_xlim(0,XBOX)
                axt.set_ylim(0,YBOX)
                axt.set_yticklabels([])
                axt.set_xticklabels([])
                axt.set_aspect('equal', adjustable='box')
                    
                figt.savefig(r'%s\Interface-Test.%s\interface_top_'%(WRKDIR,str(REPLICATE)) + str('%03d' %idx) + '.png')
            ##
            
            frmt = frmt + 1        
    
            
            #Extract L (Box length) in nm
            if parallelto == 'x' :
                L = 0.1*XBOX
            if parallelto == 'y' :
                L = 0.1*YBOX
            
            ens_BOX_tl.append(L)#Interafce 1
            ens_BOX_tl.append(L)#Interafce 2
            
            ens_AREA_tl.append(XBOX*YBOX)#Lateral Area
            
            #Compute interface length
            ens_interface_length_tl.append(spldist1_tl)
            ens_interface_length_tl.append(spldist2_tl)
            
            ens_total_interface_length_tl.append(spldist1_tl+spldist2_tl)
    
            #Compute line deviations in nm
            linedeviations1_tl = 0.1*getLineDeviations(spline1_tl)
            linedeviations2_tl = 0.1*getLineDeviations(spline2_tl)
            
            #Compute FFT and IFFT of line deviations to get hk and hminusk
    
            FFT1_tl              = fft(linedeviations1_tl)
            FFT2_tl              = fft(linedeviations2_tl)
            ens_FFT_tl.append(abs(FFT1_tl))
            ens_FFT_tl.append(abs(FFT2_tl))
            
            hk1_tl              = fft(linedeviations1_tl)/L
            hminusk1_tl         = ifft(linedeviations1_tl)*len(linedeviations1_tl)/L
            
            hk2_tl              = fft(linedeviations2_tl)/L
            hminusk2_tl         = ifft(linedeviations2_tl)*len(linedeviations2_tl)/L       
    
            hk_hminusk1_tl = np.abs(hk1_tl)*np.abs(hminusk1_tl)
            hk_hminusk2_tl = np.abs(hk2_tl)*np.abs(hminusk2_tl)
            
            ens_hk_hminusk_tl.append(hk_hminusk1_tl)
            ens_hk_hminusk_tl.append(hk_hminusk2_tl)
            
            frmt = frmt + 1
        ##
        
        frameb                    = datab[idx] #interface data
        memframeb                 = memplaneb[idx] #membrane plane data
        
        XBOX                      = frameb[2]
        YBOX                      = frameb[3]
        
        frame_bl                 = frameb[0] #xy interface coordinates top leaflet
        frame_composition_bl     = frameb[1] #interface compositiontop top leaflet
        
        memframe_bl                 = memframeb[0] 
        memframe_composition_bl     = memframeb[1] 
        
        comp_frac_bl             = getComposition(frame_composition_bl)
        ens_interface_composition_bl.append(comp_frac_bl)
        
        mol_color_bl             = getColor(memframe_composition_bl)
        
        ##
        if PRINT_INTERFACE == 'YES' and np.mod(idx,PRINT_FREQ) == 0 :
            #Plot detected interface
            figb, axb = pl.subplots()
            axb.scatter(np.hsplit(np.asarray(memframe_bl),3)[0],np.hsplit(np.asarray(memframe_bl),3)[1],c=mol_color_bl,s=50,edgecolors='none')
        ##   
        
        labels_bl   = separate_Individual_Interface_DBSCAN_PBC_2d(frame_bl)
    
        #Extracting individual interfaces based on clustering
    
        intf1_bl, intf2_bl= extract_Individual_Interface(frame_bl,labels_bl)
    
        intf1_PBC_unwrapped_bl= unwrapPBC(intf1_bl)
        intf2_PBC_unwrapped_bl= unwrapPBC(intf2_bl)
        bin_gridmean1_bl, bin_splmean1_bl= binInterfacepoints(intf1_PBC_unwrapped_bl)
        bin_gridmean2_bl, bin_splmean2_bl= binInterfacepoints(intf2_PBC_unwrapped_bl)
        
        if len(bin_gridmean1_bl) > 3 and len(bin_gridmean2_bl) > 3 : 
            spline1_bl, gridpt1_bl, spldist1_bl= employ_Cubic_Spline_and_get_Length(bin_gridmean1_bl, bin_splmean1_bl)
            spline2_bl, gridpt2_bl, spldist2_bl= employ_Cubic_Spline_and_get_Length(bin_gridmean2_bl, bin_splmean2_bl)
    
            ##
            if PRINT_INTERFACE == 'YES' and np.mod(idx,PRINT_FREQ) == 0 :
                #Plot Interface and spline
                #pl.clf()
                axb.scatter(np.hsplit(np.asarray(intf1_PBC_unwrapped_bl),2)[0],np.hsplit(np.asarray(intf1_PBC_unwrapped_bl),2)[1],s=75,facecolor='none',linewidth='3',edgecolors='grey')
                if parallelto == 'x' :
                    axb.scatter(bin_gridmean1_bl,bin_splmean1_bl,c='violet',s=200,marker=(5, 1))
                    axb.plot(gridpt1_bl, spline1_bl, '-m', lw=4)
                elif parallelto == 'y' :
                    axb.scatter(bin_splmean1_bl,bin_gridmean1_bl,c='violet',s=200,marker=(5, 1))
                    axb.plot(spline1_bl,gridpt1_bl,'-m', lw=4)
                    
                axb.scatter(np.hsplit(np.asarray(intf2_PBC_unwrapped_bl),2)[0],np.hsplit(np.asarray(intf2_PBC_unwrapped_bl),2)[1],s=75,facecolor='none',linewidth='3',edgecolors='grey')
                if parallelto == 'x' :
                    axb.scatter(bin_gridmean2_bl,bin_splmean2_bl,c='violet',s=200,marker=(5, 1))
                    axb.plot(gridpt2_bl, spline2_bl, '-m', lw=4)
                elif parallelto == 'y' :
                    axb.scatter(bin_splmean2_bl,bin_gridmean2_bl,c='violet',s=200,marker=(5, 1))
                    axb.plot(spline2_bl,gridpt2_bl,'-m', lw=4)
                    
                axb.set_xlim(0,XBOX)
                axb.set_ylim(0,YBOX)
                axb.set_yticklabels([])
                axb.set_xticklabels([])
                axb.set_aspect('equal', adjustable='box')
                    
                figb.savefig(r'%s\Interface-Test.%s\interface_bottom_'%(WRKDIR,str(REPLICATE)) + str('%03d' %idx) + '.png')

            frmb = frmb + 1                   
            ##
            
            #Extract L (Box length)
            if parallelto == 'x' :
                L = 0.1*XBOX
            if parallelto == 'y' :
                L = 0.1*YBOX
            
            ens_BOX_bl.append(L)#Interafce 1
            ens_BOX_bl.append(L)#Interafce 2
            
            ens_AREA_bl.append(XBOX*YBOX)##Lateral Area
            
            #Compute interface length
            ens_interface_length_bl.append(spldist1_bl)
            ens_interface_length_bl.append(spldist2_bl)
            
            ens_total_interface_length_bl.append(spldist1_bl+spldist2_bl)
    
            #Compute line deviations in nm
            linedeviations1_bl = 0.1*getLineDeviations(spline1_bl)
            linedeviations2_bl = 0.1*getLineDeviations(spline2_bl)
            
            #Compute FFT and IFFT of line deviations to get hk and hminusk
    
            FFT1_bl              = fft(linedeviations1_bl)
            FFT2_bl              = fft(linedeviations2_bl)
            ens_FFT_bl.append(abs(FFT1_bl))
            ens_FFT_bl.append(abs(FFT2_bl))
            
            hk1_bl             = fft(linedeviations1_bl)/L
            hminusk1_bl        = ifft(linedeviations1_bl)*len(linedeviations1_bl)/L
            
            hk2_bl             = fft(linedeviations2_bl)/L
            hminusk2_bl        = ifft(linedeviations2_bl)*len(linedeviations2_bl)/L
            
            hk_hminusk1_bl = np.abs(hk1_bl)*np.abs(hminusk1_bl)
            hk_hminusk2_bl = np.abs(hk2_bl)*np.abs(hminusk2_bl)
            
            ens_hk_hminusk_bl.append(hk_hminusk1_bl)
            ens_hk_hminusk_bl.append(hk_hminusk2_bl)
            
    #Saving bulk analysis files 
    np.save(r'%s\ens_interface_length_tl.R%s'%(WRKDIR,str(REPLICATE)),ens_interface_length_tl)
    np.save(r'%s\ens_interface_length_bl.R%s'%(WRKDIR,str(REPLICATE)),ens_interface_length_bl)
    
    np.save(r'%s\ens_total_interface_length_tl.R%s'%(WRKDIR,str(REPLICATE)),ens_total_interface_length_tl)
    np.save(r'%s\ens_total_interface_length_bl.R%s'%(WRKDIR,str(REPLICATE)),ens_total_interface_length_bl)
    
    np.save(r'%s\ens_FFT_tl.R%s'%(WRKDIR,str(REPLICATE)),ens_FFT_tl)
    np.save(r'%s\ens_FFT_bl.R%s'%(WRKDIR,str(REPLICATE)),ens_FFT_bl)
    
    np.save(r'%s\ens_hk_hminusk_tl.R%s'%(WRKDIR,str(REPLICATE)),ens_hk_hminusk_tl)
    np.save(r'%s\ens_hk_hminusk_bl.R%s'%(WRKDIR,str(REPLICATE)),ens_hk_hminusk_bl)
    
    np.save(r'%s\ens_interface_composition_tl.R%s'%(WRKDIR,str(REPLICATE)),ens_interface_composition_tl)
    np.save(r'%s\ens_interface_composition_bl.R%s'%(WRKDIR,str(REPLICATE)),ens_interface_composition_bl)
    
    np.save(r'%s\ens_BOX_tl.R%s'%(WRKDIR,str(REPLICATE)),ens_BOX_tl)
    np.save(r'%s\ens_BOX_bl.R%s'%(WRKDIR,str(REPLICATE)),ens_BOX_bl)
    
    np.save(r'%s\ens_AREA_tl.R%s'%(WRKDIR,str(REPLICATE)),ens_AREA_tl)
    np.save(r'%s\ens_AREA_bl.R%s'%(WRKDIR,str(REPLICATE)),ens_AREA_bl)
    
    
    #Calculating Line Tension
    #Filter data based on interface lengths
    
    avg_BOX_tl                  = np.mean(ens_BOX_tl)
    avg_BOX_bl                  = np.mean(ens_BOX_bl)
    L                           = (avg_BOX_tl + avg_BOX_bl)/2
    
    ens_FFT_tl_flt = filterOutliersWithMedian(ens_interface_length_tl,ens_FFT_tl)
    ens_FFT_bl_flt = filterOutliersWithMedian(ens_interface_length_bl,ens_FFT_bl)
    
    avg_FFT_tl  = np.mean(ens_FFT_tl_flt,axis=0)
    avg_FFT_bl  = np.mean(ens_FFT_bl_flt,axis=0)
    
    avg_FFT_tl_pos          = avg_FFT_tl[1:int(len(avg_FFT_tl)/2)]
    avg_FFT_tl_neg          = avg_FFT_tl[int(len(avg_FFT_tl)/2):len(avg_FFT_tl)]
    k_tl_pos            = np.linspace(0, len(avg_FFT_tl_pos)*2*np.pi/(L), len(avg_FFT_tl_pos)+1)[1:]
    k_tl_neg            = -np.linspace(0, len(avg_FFT_tl_neg)*2*np.pi/(L), len(avg_FFT_tl_neg)+1)[1:][::-1]
    
    avg_FFT_bl_pos          = avg_FFT_bl[1:int(len(avg_FFT_bl)/2)]
    avg_FFT_bl_neg          = avg_FFT_bl[int(len(avg_FFT_bl)/2):len(avg_FFT_bl)]
    k_bl_pos            = np.linspace(0, len(avg_FFT_bl_pos)*2*np.pi/(L), len(avg_FFT_bl_pos)+1)[1:]
    k_bl_neg            = -np.linspace(0, len(avg_FFT_bl_neg)*2*np.pi/(L), len(avg_FFT_bl_neg)+1)[1:][::-1]
    
    
    ens_hk_hminusk_tl_flt = filterOutliersWithMedian(ens_interface_length_tl,ens_hk_hminusk_tl)
    ens_hk_hminusk_bl_flt = filterOutliersWithMedian(ens_interface_length_bl,ens_hk_hminusk_bl)
    
    avghk_power_spectrum_tl  = np.mean(ens_hk_hminusk_tl_flt,axis=0) #In nm^2 units
    avghk_power_spectrum_bl  = np.mean(ens_hk_hminusk_bl_flt,axis=0) #In nm^2 units
    
    stdhk_power_spectrum_tl  = np.std(ens_hk_hminusk_tl_flt,axis=0) #In nm^2 units
    stdhk_power_spectrum_bl  = np.std(ens_hk_hminusk_bl_flt,axis=0) #In nm^2 units
    
    #Power spectrum coresponding to the positive frequencies
    
    Nspec_tl = np.size(avghk_power_spectrum_tl)
    if Nspec_tl % 2 == 0 : #Even
        avghk_power_spectrum_tl = avghk_power_spectrum_tl[1:int(int(Nspec_tl/2))-1]
        stdhk_power_spectrum_tl = stdhk_power_spectrum_tl[1:int(Nspec_tl/2)-1]
    else : #Odd
        avghk_power_spectrum_tl = avghk_power_spectrum_tl[1:(Nspec_tl-1)/2]
        stdhk_power_spectrum_tl = stdhk_power_spectrum_tl[1:(Nspec_tl-1)/2]
    
    Nspec_bl = np.size(avghk_power_spectrum_bl)
    if Nspec_bl % 2 == 0 : #Even
        avghk_power_spectrum_bl = avghk_power_spectrum_bl[1:int(Nspec_bl/2)-1]
        stdhk_power_spectrum_bl = stdhk_power_spectrum_bl[1:int(Nspec_bl/2)-1]
    else : #Odd
        avghk_power_spectrum_bl = avghk_power_spectrum_bl[1:(Nspec_bl-1)/2]
        stdhk_power_spectrum_bl = stdhk_power_spectrum_bl[1:(Nspec_bl-1)/2]
    
    avg_k_tl = np.linspace(0.0, len(avghk_power_spectrum_tl)*2*np.pi/(avg_BOX_tl), len(avghk_power_spectrum_tl)+1)[1:] #In nm^-1 units
    avg_k_bl = np.linspace(0.0, len(avghk_power_spectrum_bl)*2*np.pi/(avg_BOX_bl), len(avghk_power_spectrum_bl)+1)[1:] #In nm^-1 units
    
    #Selecting first four wave number mode
    avg_k_tl_trimmed = avg_k_tl[:miMAX]
    avghk_power_spectrum_tl_trimmed = avghk_power_spectrum_tl[:miMAX]
    
    low_k_modes_REP.append(avg_k_tl_trimmed)
    power_spectrum_REP.append(avghk_power_spectrum_tl_trimmed)
    
    #Using 1/h(k)^2 = [L*lambda/(KB*T)] (K^2) relationship
    avg_k_tl_trimmed                    = np.square(avg_k_tl_trimmed)
    avghk_power_spectrum_tl_trimmed = 1.0/avghk_power_spectrum_tl_trimmed   
    
    #Line of best fit for the trimmed data
    m_t,c_t = pl.polyfit(avg_k_tl_trimmed, avghk_power_spectrum_tl_trimmed, 1)
    
    LT_t    = (10**12)*(m_t*KBT/(avg_BOX_tl*(10**(-9)))) # In pN Units
    
    file = open(r'%s\LT_results_k2.R%s.dat'%(WRKDIR,str(REPLICATE)), 'w')
    file.write('gradient and intercept : m_t,c_t ' + str(m_t) + ' , ' + str(c_t))
    file.write('\n')
    file.write('LT_topleaflet (pN) = ' + str(LT_t))
    file.write('\n')
    file.write('\n')
    
    print 'gradient and intercept : m_t,c_t', m_t,c_t
    print 'LT top leaflet (pN) =', LT_t
    
    #Selecting first four wave number mode
    avg_k_bl_trimmed = avg_k_bl[:miMAX]
    avghk_power_spectrum_bl_trimmed = avghk_power_spectrum_bl[:miMAX]
    
    low_k_modes_REP.append(avg_k_bl_trimmed)
    power_spectrum_REP.append(avghk_power_spectrum_bl_trimmed)
    
    #Using 1/h(k)^2 = [L*lambda/(KB*T)] (K^2) relationship
    avg_k_bl_trimmed                    = np.square(avg_k_bl_trimmed)
    avghk_power_spectrum_bl_trimmed = 1.0/avghk_power_spectrum_bl_trimmed   
    
    #Line of best fit for the trimmed data
    m_b,c_b = pl.polyfit(avg_k_bl_trimmed, avghk_power_spectrum_bl_trimmed, 1)
    
    LT_b    = (10**12)*(m_b*KBT/(avg_BOX_bl*(10**(-9)))) # In pN Units
    
    file.write('gradient and intercept : m_b,c_b ' + str(m_b) + ' , ' + str(c_b))
    file.write('\n')
    file.write('LT bottom leaflet (pN) = ' + str(LT_b))
    file.close()
    
    print 'gradient and intercept : m_b,c_b', m_b,c_b
    print 'LT bottom leaflet (pN) =', LT_b
    
    LT_REP.append(LT_t)
    LT_REP.append(LT_b)
    
    CWT_c_REP.append(c_t)
    CWT_c_REP.append(c_b)
    
    print 'Analysis Complete'

np.save(r'%s\low_k_modes_REP.ALL'%WRKDIR,low_k_modes_REP)
np.save(r'%s\power_spectrum_REP.ALL'%WRKDIR,power_spectrum_REP)
    
file = open(r'%s\LT_results_k2.ALL.dat'%WRKDIR, 'w')
file.write('Line Tension ALL REPLICATES :' + str(np.mean(np.asarray(LT_REP))) + ' +/- ' + str(np.std(np.asarray(LT_REP))))
file.write('\n')
file.write('Intercept ALL REPLICATES :' + str(np.mean(np.asarray(CWT_c_REP))) + ' +/- ' + str(np.std(np.asarray(CWT_c_REP))))
file.write('\n')
file.close()