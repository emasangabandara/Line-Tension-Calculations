#!/usr/bin/env python2.7
#This script extracts the set of lipid and cholesterol atoms exist in a predifined 2D surface
#Utilzed in the line tension calculations published in J. Chem. Phys. 150, 204702 (2019); https://doi.org/10.1063/1.5091450
#Example usage python extract_membrane_plane.py 1 up #[replicate trajectory] [top(up) or bottom (down) leaflet to analyze]
#This script utilizes the MDAnalysis API to read the GROMACS simulation trajectoy data
from __future__ import division
import numpy as np
import math
from MDAnalysis import *
import MDAnalysis
import MDAnalysis.lib.distances
import MDAnalysis.lib.NeighborSearch as NS
import multiprocessing as mp
import sys

#PBC introduced
MDAnalysis.core.flags['use_periodic_selections'] = True
MDAnalysis.core.flags['use_KDTree_routines'] = False

runindex = sys.argv[1] # replicates 1, 2, 3
side     = sys.argv[2] # up for upper leaflet, down for lower leaflet

###############INPUTS######################################
print "Loading inputs..."
replicateindex = int(runindex)
psf  = '../../step5_assembly.psf'
trr = '../prod.run.trr'

u        = MDAnalysis.Universe(psf,trr)
nprocs   = 28 # number of stupidly parallel processors to use
start_f  = start_f  = int(3*u.trajectory.n_frames/30)
end_f    = u.trajectory.n_frames
skip     = 1

lipid1 = 'DPPC'
lipid2 = 'DIPC'
lipid3 = 'CHOL'

protein = 'YES'

#Prescision radius of 1.5 nm is used in leaflet determination.
#Prescision radius of 1.0 nm is used in Protein atom selction in lipid tail plane. 

#Selection for the leaflet head group plane in leaflet seperation. 
sel1    = 'resname DPPC and name PO4' # POPC head
sel2    = 'resname DIPC and name PO4' # POPC head
sel3    = 'resname CHOL and name ROH' # CHOL head

#Protein residues in contact with the well packed tail region of the bilayer
selprot       = 'segid PRO*'    

frames   = np.arange(start_f,end_f,skip) # frames from which interface atoms will be extracted

data = []
def extract_data(frame):
#for frame in frames:
    u = MDAnalysis.Universe(psf,trr)
    print 'Frame %i of %i......'%(frame,u.trajectory.n_frames)
    u.trajectory[frame]
    XBOX       = u.dimensions[0]
    YBOX       = u.dimensions[1]
##################################################################################
    ### Determining side of the bilayer CHOL belongs to in this frame
    #Lipid Residue names
    lpd1_atoms     = u.select_atoms(sel1)
    lpd2_atoms     = u.select_atoms(sel2)
    lpd3_atoms     = u.select_atoms(sel3)
    #Protein atom selction
    prot_atoms     = u.select_atoms(selprot) 

    num_lpd1 = lpd1_atoms.n_atoms
    num_lpd2 = lpd2_atoms.n_atoms
    num_lpd3 = lpd3_atoms.n_atoms
    # atoms in the upper leaflet as defined by insane.py or the MARTINI-GUI membrane builders
    # select cholesterol headgroups within 1.5 nm of lipid headgroups in the selected leaflet
    if side == 'up':
        lpd1i = lpd1_atoms[:int((num_lpd1)/2)]
        lpd2i = lpd2_atoms[:int((num_lpd2)/2)]
        phospholipids = lpd1i + lpd2i
        ns_lipids = NS.AtomNeighborSearch(lpd3_atoms) #CHOL
        lpd3i = ns_lipids.search(phospholipids,15.0)
        #proti = prot_atoms.select_atoms(protsel_up)
        
    elif side == 'down':
        lpd1i = lpd1_atoms[int((num_lpd1)/2):]
        lpd2i = lpd2_atoms[int((num_lpd2)/2):]
        phospholipids = lpd1i + lpd2i
        ns_lipids = NS.AtomNeighborSearch(lpd3_atoms) #CHOL
        lpd3i = ns_lipids.search(phospholipids,15.0)
        #proti = prot_atoms.select_atoms(protsel_down)
        
    #Using tail definitions
    groups1 = []
    for i in np.arange(len(lpd1i.resnums)):
        resnum = lpd1i.resnums[i]
        group = u.select_atoms('resname DPPC and resnum %i and (name C2A or name C2B)'%resnum)
        groups1.append(group)
    lpd1i = np.sum(groups1)

    groups2 = []
    for i in np.arange(len(lpd2i.resnums)):
        resnum = lpd2i.resnums[i]
        group = u.select_atoms('resname DIPC and resnum %i and (name D2A or name D2B)'%resnum)
        groups2.append(group)
    lpd2i = np.sum(groups2)
    
    groups3 = []
    for i in np.arange(len(lpd3i.resnums)):
        resnum = lpd3i.resnums[i]
        group = u.select_atoms('resname CHOL and resnum %i and (name R3)'%resnum)
        groups3.append(group)
    lpd3i = np.sum(groups3)

    if protein == 'YES' :    
    #Selecting Protein atoms that are on the tail plane.
        phospholipidtails = lpd1i + lpd2i #Phospholipid tail atoms that defines the plane of interest.
        ns_PROT           = NS.AtomNeighborSearch(prot_atoms) #Protein
        proti             = ns_PROT.search(phospholipidtails,10.0) #Protein atoms that are within 1.0nm of the Phospholipid tail atoms.
    
        u_leaflet     = lpd1i + lpd2i + lpd3i + proti
    else :
        u_leaflet     = lpd1i + lpd2i + lpd3i

##################################################################################

    #Collecting coordinates of the interface atoms
    mem_plane             = []
    mem_plane_composition = []

    if protein == 'YES' :
        for selatm in proti :
			mem_plane.append(selatm.position) 
			mem_plane_composition.append('PROT')

    for selatm in lpd1i :
        mem_plane.append(selatm.position) 
        mem_plane_composition.append(lipid1)
		
    for selatm in lpd2i :
        mem_plane.append(selatm.position) 
        mem_plane_composition.append(lipid2)
		
    for selatm in lpd3i :
        mem_plane.append(selatm.position) 
        mem_plane_composition.append(lipid3)		

    mem_plane_data       = [mem_plane,mem_plane_composition,XBOX,YBOX]

    return mem_plane_data

#Collecting frames
pool = mp.Pool(processes=nprocs)
print 'Initiating multiprocessing with %i processors'%nprocs
data = pool.map(extract_data, frames)

print 'Saving numpy data files'
if side == 'up':
    np.save('upper_mem_plane_data.R%s.npy'%str(runindex),np.asarray(data))
if side == 'down':
    np.save('lower_mem_plane_data.R%s.npy'%str(runindex),np.asarray(data))

print 'Analysis Complete'
