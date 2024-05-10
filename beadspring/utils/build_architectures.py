import sys
import math
import numpy as np


def create_ring(num_beads, filename='ring.data'):
    '''
    Create a ring polymer with a given number of beads.
    The ring is centered at the origin and the beads are placed on the circumference of a circle.


    Parameters
    ----------
    num_beads : int
        Number of beads in the ring polymer.
    filename : str
        File name to save the data file. Default is 'ring.data'.

    Returns
    -------
    None
    '''

    # Calculate the radius for a distance of 1 unit between consecutive beads
    radius = num_beads / (2 * math.pi)

    # Header for LAMMPS data file with adjusted box dimensions to center the ring
    header = f"LAMMPS Description\n\n"
    header += f"{num_beads} atoms\n"
    header += "1 atom types\n"
    header += f"{num_beads} bonds\n"
    header += "1 bond types\n\n"
    header += "0 angles\n"
    header += "0 angle types\n"
    header += "0 dihedrals\n"
    header += "0 dihedral types\n"
    box_dim = radius + 1  # Add extra space to the radius for the box dimensions
    header += f"{-box_dim} {box_dim} xlo xhi\n"
    header += f"{-box_dim} {box_dim} ylo yhi\n"
    header += f"-1.0 1.0 zlo zhi\n\n"  # Z dimensions are minimal as it's a 2D simulation
    header += "Masses\n\n"
    header += "1 1.0\n\n"
    
    # Atoms section
    atoms_section = "Atoms\n\n"
    for i in range(num_beads):
        angle = 2 * math.pi * i / num_beads
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 0.0
        atoms_section += f"{i + 1} 1 {x} {y} {z}\n"

    # Bonds section
    bonds_section = "Bonds\n\n"
    for i in range(num_beads):
        next_bond = i + 1 if i < num_beads - 1 else 0
        bonds_section += f"{i + 1} 1 {i + 1} {next_bond + 1}\n"

    if file_name is None:
        file_name = 'ring.data'

    # Write to file
    with open(file_name, 'w') as file:
        file.write(header)
        file.write(atoms_section)
        file.write(bonds_section)


def create_polymer_matrix(n, m, l=0.97, R=0.5, file_name=None):
    '''
    Create a matrix of polymer chains with a given number of chains and monomers in each chain.
    The core is a sphere with radius R.
    The chains are composed of monomers with a bond length of l.
    The charge is zero for all atoms.
    The atom number starts from 1.
    Atom types are set in a way that when a polymer is embedded in the matrix, the total number
    of atom and bond types is matched.

    Parameters
    ----------
    n : int
        Number of stars per side.
    m : int
        Number of monomers in each arm. (2*m gives the strand length)
    R : float
        Radius of the core. Default is 0.5.
    l : float    
        Bond length. Default is 0.97.
    core_mass : float
        Mass of the core.
    atom_types : int
        Number of atom types.
    bond_types : int
        Number of bond types.
    masses : np.ndarray
        Masses of the atoms.
    file_name : str
        File name to save the data file. Default None will save as 'polymer_matrix.data'.

    Returns
    -------
    None
    '''

    core_mass = (2*R)**3

    if R == 0:
        atom_types = 4
        bond_types = 4
        masses = np.ones((4,2))
        masses[1,0] = 2
    else:
        atom_types = 5
        bond_types = 3
        masses = np.ones((5,2))
        masses[1,0] = 2
        masses[2,1] = core_mass
        
    # make a grid
    xmax = n+2*n*m

    x = np.arange(0, xmax, 1)
    X, Y, Z = np.meshgrid(x, x, x)

    # get planes of gridlines
    i = np.arange(m, (2*n-1)*m + (n-1)+1, 2*m+1)  # locations of gridlines
    maskx = np.isin(X, i)
    masky = np.isin(Y, i)
    maskz = np.isin(Z, i)

    # combine masks
    maskxy = maskx & masky
    maskyz = masky & maskz
    maskxz = maskx & maskz
    mask = maskxy | maskyz | maskxz

    # apply mask
    pos = np.empty((len(X[mask]), 3))
    pos[:,0] = X[mask]
    pos[:,1] = Y[mask]
    pos[:,2] = Z[mask]

    # select midpoints and give atom type 2
    maskxyz = maskx & masky & maskz

    centers = np.empty((len(X[maskxyz]), 3))
    centers[:,0] = X[maskxyz]
    centers[:,1] = Y[maskxyz]
    centers[:,2] = Z[maskxyz]
    mask2 = np.all(np.isin(pos, centers), axis=1)
    type = np.where(mask2==True, 2, 1)

    # neighbouring atoms have a bond
    bonda = []
    bondb = []
    for i in range(len(pos)):
        for j in np.arange(i, len(pos), 1):
            if (pos[i,0] == pos[j,0] - 1 or pos[i,0] == pos[j,0] - xmax+1) and pos[i,1] == pos[j,1] and pos[i,2] == pos[j,2]:
                bonda += [i+1]
                bondb += [j+1]
            elif pos[i,0] == pos[j,0] and (pos[i,1] == pos[j,1]-1 or pos[i,1] == pos[j,1] - xmax+1) and pos[i,2] == pos[j,2]:
                bonda += [i+1]
                bondb += [j+1]
            elif pos[i,0] == pos[j,0] and pos[i,1] == pos[j,1] and (pos[i,2] == pos[j,2]-1 or pos[i,2] == pos[j,2] - xmax+1):
                bonda += [i+1]
                bondb += [j+1]

    # scaling
    pos = pos * l


    if file_name is None:
        file_name = 'star.data'

    #---------------------- write LAMMPS data file ---------------#
    with open(file_name, 'w')as f:
        # First line is a comment 
        f.write('The matrix generated by Python\n\n')
        #----------------Header Line----------------#
        f.write(f'{len(pos)} atoms\n')
        f.write(f'{atom_types} atom types\n')
        f.write(f'{len(bonda)} bonds\n')
        f.write(f'{bond_types} bond types\n')
        f.write('0 angles\n')
        f.write('0 angle types\n')
        f.write('0 dihedrals\n')
        f.write('0 dihedral types\n')
        #--------------Specify box dimensions------------------#
        f.write(f'{min(pos[:,0])- 0.5*l} {max(pos[:,0])+ 0.5*l} xlo xhi\n')
        f.write(f'{min(pos[:,1])- 0.5*l} {max(pos[:,1])+ 0.5*l} ylo yhi\n')
        f.write(f'{min(pos[:,2])- 0.5*l} {max(pos[:,2])+ 0.5*l} zlo zhi\n')
        #--------------Specify masses--------------------------#
        f.write('\nMasses\n\n')
        np.savetxt(f, masses, fmt='%1.0f %2f')
        # Atoms section
        f.write('\nAtoms  #full \n\n')
        # Atom_style: full----atom-Id; molecule-ID; atom-type; q; x; y; z;
        # Write Atoms Section
        for i in range (len(pos)):
            f.write(f'{i+1} 1 {type[i]} 0 {pos[i][0]} {pos[i][1]} {pos[i][2]}\n')   
        # Write Bonds Section
        f.write('\n Bonds   \n\n')
        index = 1
        for i in range(len(bonda)):
            f.write(f'{index} 1 {bonda[i]} {bondb[i]} \n')
            index = index + 1

def create_star(num_arms, arm_length, core_radius=0.5, file_name=None):
    '''
    Create a star polymer with a given number of arms and arm length.
    The core is a sphere with radius R = 0.5.
    The arms are composed of monomers with a bond length of 0.97.
    The charge is zero for all atoms.
    The atom number starts from 1.
    The atom types are 1 for the core and 2 for the rest.

    Parameters
    ----------
    num_arms : int
        Number of arms in the star polymer.
    arm_length : int
        Number of monomers in each arm.
    core_radius : float
        Radius of the core. Default is 0.5.
    save_file : str
        File name to save the data file. Default None will save as 'star.data'.

    Returns
    -------
    None
    '''
    R = core_radius
    D = 2 * R
    coor_core = np.zeros(3)

    # Polymer parameters

    blen1 = 0.97 # bond length
    dmin = 0.8 # the minimum distance between atom 1 and atom 3
    rho_polymer = 1 # density
    monomer_mass = 1
    core_mass = (2 * R)**3
    N_polymer = arm_length * num_arms

    xlo = -10000 # box boundaries
    xhi = 100000
    ylo = -10000
    yhi = 100000
    zlo = -10000
    zhi = 100000

    coor_polymer = np.zeros((num_arms, arm_length, 3))

    for j in range(num_arms):  # chain loop
        coor = np.zeros((arm_length,3)) 
        for i in range(arm_length):  # monomer loop for each chain
            if i==0:
                ph = np.pi * np.random.rand()
                th = 2 * np.pi * np.random.rand()
                xij = (R+0.5)*np.sin(ph)*np.cos(th) # initial position for each chain
                yij = (R+0.5)*np.sin(ph)*np.sin(th)
                zij = (R+0.5)*np.cos(ph)
            else:
                restriction = True  # coordinate of the first B atom for the first A atom
                while restriction:
                    dx = 2.0 * np.random.rand() - 1.0
                    dy = 2.0 * np.random.rand() - 1.0
                    dz = 2.0 * np.random.rand() - 1.0
                    rsq = dx**2 + dy**2 + dz**2
                    r = np.sqrt(rsq)
                    dx = dx / r
                    dy = dy / r
                    dz = dz / r
                    xij = coor[i-1][0] + dx * blen1
                    yij = coor[i-1][1] + dy * blen1
                    zij = coor[i-1][2] + dz * blen1
                    restriction = False
                    # all monomers outside of NP
                    if (np.sqrt(xij**2 + yij**2 + zij**2) < R+0.5):
                        restriction = True
                    if i >= 2:
                        distx = xij - coor[i-2][0]
                        disty = yij - coor[i-2][1]
                        distz = zij - coor[i-2][2]
                        if (np.sqrt(distx*distx+ disty*disty + distz*distz) <= dmin): # the minimum distance between atom 1 and atom 3
                            restriction = True
                        elif ((xij<xlo) or (xij>xhi) or (yij<ylo) or (yij>yhi) or (zij<zlo) or (zij>zhi)):
                            restriction = True
                        #elseif ((xij-5)^2 + (yij-5)^2 + (zij-5)^2) < 25 
                        #   restriction = True
            coor[i][0] = xij
            coor[i][1] = yij
            coor[i][2] = zij
        coor_polymer[ j, : ] = coor[:,:]

    coor_polymer = coor_polymer.reshape(num_arms*arm_length, 3)
    coor_core = coor_core.reshape((1,3))
    coords = np.concatenate((coor_core, coor_polymer))

    massA = core_mass
    massB = 1

    num_bonds_arm = arm_length - 1
    num_atoms = num_arms * arm_length + 1
    bonds = num_bonds_arm * num_arms + num_arms
    atomtypes = 2
    bondtypes = 2
    bxlo = min(coords[:,0]) - 0.1
    bxhi = max(coords[:,0]) + 0.1
    bylo = min(coords[:,1]) - 0.1
    byhi = max(coords[:,1]) + 0.1
    bzlo = min(coords[:,2]) - 0.1
    bzhi = max(coords[:,2]) + 0.1

    # Create molecule tag

    molecule = np.zeros(num_atoms)
    molecule[0] = 1

    for i in range(num_arms):
        for j in range(arm_length):
            if j == 0:
                molecule[j+i*arm_length+1] = 1
            else:
                molecule[j+i*arm_length+1] = 2 # i+2 for different molecules (?)
                    
    # Charge

    charge = np.zeros(num_atoms)

    # Atom number                

    num = np.arange(1,num_atoms+1)

    # Type

    types = np.ones(num_atoms) + 1  # 1 for the core, 2 for the rest
    types[0] = 1

    coordinates = np.zeros((num_atoms,7))

    coordinates[:, 0] = num
    coordinates[:, 1] = molecule
    coordinates[:, 2] = types
    coordinates[:, 3] = charge
    coordinates[:, 4:] = coords                

    # Bonds matrix

    bond = np.zeros((bonds, 4))
    bond[:,0] = np.arange(1,bonds+1)

    for k in range(num_arms):
        
        bond[ (k*(arm_length-1)):(k+1)*(arm_length-1), 1] = 2
        bond[ (k*(arm_length-1)):(k+1)*(arm_length-1), 2] = np.arange( k*arm_length+2, (k+1)*arm_length+1)
        bond[ (k*(arm_length-1)):(k+1)*(arm_length-1), 3] = np.arange( k*arm_length+3, (k+1)*arm_length+2) 
                    

    # Bonds between arms and core

    for k in range(num_arms):
        bond[num_bonds_arm * num_arms + (k+1)-1, 1] = 1
        bond[num_bonds_arm * num_arms + (k+1)-1, 2] = k * arm_length + 2
        bond[num_bonds_arm * num_arms + (k+1)-1, 3] = 1




    # Create LAMMPS data file

    if file_name is None:
        file_name = 'star.data'


    with open(file_name, 'w') as f:
        
        f.write('LAMMPS data file for Coarse Grained GNP\n\n')
        f.write('%1.0f atoms\n' % num_atoms)
        f.write('%1.0f atom types\n' % atomtypes)
        f.write('%1.0f bonds\n' % bonds)            
        f.write('%1.0f bond types\n' % bondtypes)
        f.write('0 angles\n')
        f.write('0 angle types\n')
        f.write('0 dihedrals\n')            
        f.write('0 dihedral types\n')
        f.write('%1.6f %1.6f xlo xhi\n' % (bxlo, bxhi))
        f.write('%1.6f %1.6f ylo yhi\n' % (bylo, byhi))        
        f.write('%1.6f %1.6f zlo zhi\n\n' % (bzlo, bzhi))
        
        f.write('Masses\n\n')
        f.write('%1.0f %1.6f\n' % (1, massA))
        f.write('%1.0f %1.6f\n\n'% (2, massB))
        f.write('Atoms\n\n')
        np.savetxt(f, coordinates, fmt='%4.0f %6.0f %6.0f %6.2f %12.6f %12.6f %12.6f')
        f.write('\n')
        f.write('Bonds\n\n')
        np.savetxt(f, bond, fmt='%4.0f %6.0f %6.0f %6.0f')


def create_2d_star(num_arms, arm_length, file_name=None):
    '''
    Create a 2D star polymer with a given number of arms and arm length.
    The core is a circle with radius R = 0.5.
    The arms are composed of monomers with a bond length of 0.97.
    The charge is zero for all atoms.
    The atom number starts from 1.
    The atom types are 1 for the core and 2 for the rest.

    Parameters
    ----------
    num_arms : int
        Number of arms in the star polymer.
    arm_length : int
        Number of monomers in each arm.
    save_file : str
        File name to save the data file. Default None will save as '2d_star.data'.

    Returns
    -------
    None
    '''
    bond_length = 0.97
    theta = 2*np.pi / num_arms

    core = np.zeros(3)
    arm_positions = np.zeros((num_arms, arm_length, 3))


    for j in range(num_arms):
        coor = np.zeros((arm_length, 3))
        for i in range(arm_length):
            if i == 0:
                x_new = core[0] + bond_length * np.cos(j * theta)
                y_new = core[1] + bond_length * np.sin(j * theta)
            else:
                x_new = coor[i-1][0] + bond_length * np.cos(j * theta)
                y_new = coor[i-1][1] + bond_length * np.sin(j * theta)
            
            coor[i][0] = x_new
            coor[i][1] = y_new
        arm_positions[j, :] = coor[:, :]
        
    arm_positions = arm_positions.reshape(num_arms * arm_length, 3)
    core = core.reshape((1, 3))

    positions = np.concatenate((core, arm_positions))

    num_bonds_arm = arm_length - 1
    num_atoms = num_arms * arm_length + 1
    bonds = num_bonds_arm * num_arms + num_arms
    atomtypes = 2
    bondtypes = 2

    bxlo = min(positions[:,0]) - 0.1
    bxhi = max(positions[:,0]) + 0.1
    bylo = min(positions[:,1]) - 0.1
    byhi = max(positions[:,1]) + 0.1
    bzlo = -0.1
    bzhi = 0.1

    # Create molecule tag

    molecule = np.zeros(num_atoms)
    molecule[0] = 1

    for i in range(num_arms):
        for j in range(arm_length):
            if j == 0:
                molecule[j+i*arm_length+1] = 1
            else:
                molecule[j+i*arm_length+1] = 2 # i+2 for different molecules (?)
    # Charge

    charge = np.zeros(num_atoms)

    # Atom number                

    num = np.arange(1,num_atoms+1)

    # Type

    types = np.ones(num_atoms) + 1  # 1 for the core, 2 for the rest
    types[0] = 1

    coordinates = np.zeros((num_atoms, 7))

    coordinates[:, 0] = num
    coordinates[:, 1] = molecule
    coordinates[:, 2] = types
    coordinates[:, 3] = charge
    coordinates[:, 4:] = positions                

    # Bonds matrix

    bond = np.zeros((bonds, 4))
    bond[:,0] = np.arange(1,bonds+1)

    for k in range(num_arms):
        
        bond[ (k*(arm_length-1)):(k+1)*(arm_length-1), 1] = 2
        bond[ (k*(arm_length-1)):(k+1)*(arm_length-1), 2] = np.arange( k*arm_length+2, (k+1)*arm_length+1)
        bond[ (k*(arm_length-1)):(k+1)*(arm_length-1), 3] = np.arange( k*arm_length+3, (k+1)*arm_length+2) 


    # Bonds between arms and core

    for k in range(num_arms):
        bond[num_bonds_arm * num_arms + (k+1)-1, 1] = 1
        bond[num_bonds_arm * num_arms + (k+1)-1, 2] = k * arm_length + 2
        bond[num_bonds_arm * num_arms + (k+1)-1, 3] = 1

    # Create LAMMPS data file

    if file_name is None:
        file_name = '2d_star.data'                

    with open(file_name, 'w') as f:
        
        f.write('LAMMPS data file for Coarse Grained GNP\n\n')
        f.write('%1.0f atoms\n' % num_atoms)
        f.write('%1.0f atom types\n' % atomtypes)
        f.write('%1.0f bonds\n' % bonds)            
        f.write('%1.0f bond types\n' % bondtypes)
        f.write('0 angles\n')
        f.write('0 angle types\n')
        f.write('0 dihedrals\n')            
        f.write('0 dihedral types\n')
        f.write('%1.6f %1.6f xlo xhi\n' % (bxlo, bxhi))
        f.write('%1.6f %1.6f ylo yhi\n' % (bylo, byhi))        
        f.write('%1.6f %1.6f zlo zhi\n\n' % (bzlo, bzhi))
        
        f.write('Masses\n\n')
        f.write('%1.0f %1.6f\n' % (1, 1))
        f.write('%1.0f %1.6f\n\n'% (2, 1))
        f.write('Atoms\n\n')
        np.savetxt(f, coordinates, fmt='%4.0f %6.0f %6.0f %6.2f %12.6f %12.6f %12.6f')
        f.write('\n')
        f.write('Bonds\n\n')
        np.savetxt(f, bond, fmt='%4.0f %6.0f %6.0f %6.0f')

def create_polydisperse_star(f1, M1, f2, M2, file_name=None):
    '''
    Create a polydisperse star polymer with a given number of arms and arm length.
    The core is a sphere with radius R = 0.5.
    The arms are composed of monomers with a bond length of 0.97.
    The charge is zero for all atoms.
    The function creates a star polymer with f1 arms of length M1 and f2 arms of length M2.

    Parameters
    ----------
    f1 : int
        Number of arms in the star polymer.
    M1 : int
        Number of monomers in each arm.
    f2 : int
        Number of arms in the star polymer.
    M2 : int
        Number of monomers in each arm.
    save_file : str
        File name to save the data file. Default None will save as 'polydisperse_star.data'.

    Returns
    -------
    None

    '''

    num_arms = f1 + f2
    num_atoms = f1*M1 + f2*M2 + 1
    R = 0.5
    D = 2 * R

    coor_core = np.zeros(3)

    # Polymer parameters

    blen1 = 0.97
    dmin = 0.8
    rho_polymer = 1.
    monomer_mass = 1
    core_mass = D**3
    N_polymer = f1*M1 + f2*M2

    xlo = -10000 # box boundaries
    xhi = 100000
    ylo = -10000
    yhi = 100000
    zlo = -10000
    zhi = 100000

    coor_polymer1 = np.zeros((f1, M1, 3))

    for j in range(f1):  # chain loop
        coor = np.zeros((M1,3)) 
        for i in range(M1):  # monomer loop for each chain
            if i==0:
                ph = np.pi * np.random.rand()
                th = 2 * np.pi * np.random.rand()
                xij = (R+0.5)*np.sin(ph)*np.cos(th) # initial position for each chain
                yij = (R+0.5)*np.sin(ph)*np.sin(th)
                zij = (R+0.5)*np.cos(ph)
            else:
                restriction = True  # coordinate of the first B atom for the first A atom
                while restriction:
                    dx = 2.0 * np.random.rand() - 1.0
                    dy = 2.0 * np.random.rand() - 1.0
                    dz = 2.0 * np.random.rand() - 1.0
                    rsq = dx**2 + dy**2 + dz**2
                    r = np.sqrt(rsq)
                    dx = dx / r
                    dy = dy / r
                    dz = dz / r
                    xij = coor[i-1][0] + dx * blen1
                    yij = coor[i-1][1] + dy * blen1
                    zij = coor[i-1][2] + dz * blen1
                    restriction = False
                    # all monomers outside of NP
                    if (np.sqrt(xij**2 + yij**2 + zij**2) < R+0.5):
                        restriction = True
                    if i >= 2:
                        distx = xij - coor[i-2][0]
                        disty = yij - coor[i-2][1]
                        distz = zij - coor[i-2][2]
                        if (np.sqrt(distx*distx+ disty*disty + distz*distz) <= dmin): # the minimum distance between atom 1 and atom 3
                            restriction = True
                        elif ((xij<xlo) or (xij>xhi) or (yij<ylo) or (yij>yhi) or (zij<zlo) or (zij>zhi)):
                            restriction = True
                        #elif ((xij-5)**2 + (yij-5)**2 + (zij-5)**2) < 25 
                        #   restriction = True
            coor[i][0] = xij
            coor[i][1] = yij
            coor[i][2] = zij
        coor_polymer1[ j, : ] = coor[:,:]

    coor_polymer1 = coor_polymer1.reshape(f1*M1, 3)
    coor_core = coor_core.reshape((1,3))
    coords = np.concatenate((coor_core, coor_polymer1))


    coor_polymer2 = np.zeros((f2, M2, 3))

    for j in range(f2):  # chain loop
        coor = np.zeros((M2,3)) 
        for i in range(M2):  # monomer loop for each chain
            if i==0:
                ph = np.pi * np.random.rand()
                th = 2 * np.pi * np.random.rand()
                xij = (R+0.5)*np.sin(ph)*np.cos(th) # initial position for each chain
                yij = (R+0.5)*np.sin(ph)*np.sin(th)
                zij = (R+0.5)*np.cos(ph)
            else:
                restriction = True  # coordinate of the first B atom for the first A atom
                while restriction:
                    dx = 2.0 * np.random.rand() - 1.0
                    dy = 2.0 * np.random.rand() - 1.0
                    dz = 2.0 * np.random.rand() - 1.0
                    rsq = dx**2 + dy**2 + dz**2
                    r = np.sqrt(rsq)
                    dx = dx / r
                    dy = dy / r
                    dz = dz / r
                    xij = coor[i-1][0] + dx * blen1
                    yij = coor[i-1][1] + dy * blen1
                    zij = coor[i-1][2] + dz * blen1
                    restriction = False
                    # all monomers outside of NP
                    if (np.sqrt(xij**2 + yij**2 + zij**2) < R+0.5):
                        restriction = True
                    if i >= 2:
                        distx = xij - coor[i-2][0]
                        disty = yij - coor[i-2][1]
                        distz = zij - coor[i-2][2]
                        if (np.sqrt(distx*distx+ disty*disty + distz*distz) <= dmin): # the minimum distance between atom 1 and atom 3
                            restriction = True
                        elif ((xij<xlo) or (xij>xhi) or (yij<ylo) or (yij>yhi) or (zij<zlo) or (zij>zhi)):
                            restriction = True
                        #elif ((xij-5)**2 + (yij-5)**2 + (zij-5)**2) < 25 
                        #   restriction = True
            coor[i][0] = xij
            coor[i][1] = yij
            coor[i][2] = zij
        coor_polymer2[ j, : ] = coor[:,:]

    coor_polymer2 = coor_polymer2.reshape(f2*M2, 3)
    coords = np.concatenate((coords, coor_polymer2))


    massA = core_mass
    massB = 1


    atomtypes = 2
    bondtypes = 2
    bxlo = min(coords[:,0]) - 0.1
    bxhi = max(coords[:,0]) + 0.1
    bylo = min(coords[:,1]) - 0.1
    byhi = max(coords[:,1]) + 0.1
    bzlo = min(coords[:,2]) - 0.1
    bzhi = max(coords[:,2]) + 0.1

    # Create molecule tag

    molecule = np.ones(num_atoms)

    # Charge

    charge = np.zeros(num_atoms)

    # Atom number                

    num = np.arange(1,num_atoms+1)

    # Type

    types = np.ones(num_atoms) + 1  # 1 for the core, 2 for the rest
    types[0] = 1

    coordinates = np.zeros((num_atoms,7))

    coordinates[:, 0] = num
    coordinates[:, 1] = molecule
    coordinates[:, 2] = types
    coordinates[:, 3] = charge
    coordinates[:, 4:] = coords


    ### Bond matrix

    # num_bonds_arm = arm_length - 1
    # num_atoms = num_arms * arm_length + 1
    # bonds = num_bonds_arm * num_arms + num_arms

    num_bonds = num_atoms -1
    bond = np.zeros((num_bonds, 4))
    bond[:,0] = np.arange(1,num_bonds+1)

    for k in range(f1):
        
        bond[ (k*(M1-1)):(k+1)*(M1-1), 1] = 2
        bond[ (k*(M1-1)):(k+1)*(M1-1), 2] = np.arange( k*M1+2, (k+1)*M1+1)
        bond[ (k*(M1-1)):(k+1)*(M1-1), 3] = np.arange( k*M1+3, (k+1)*M1+2)


    for k in range(f2):
        
        bond[f1*(M1-1) + k*(M2-1) : f1*(M1-1) + (k+1)*(M2-1), 1 ] = 2
        bond[f1*(M1-1) + k*(M2-1) : f1*(M1-1) + (k+1)*(M2-1), 2 ] = np.arange( f1*M1+2 + k*M2, f1*M1+(k+1)*M2+1)
        bond[f1*(M1-1) + k*(M2-1) : f1*(M1-1) + (k+1)*(M2-1), 3 ] = np.arange( f1*M1+3 + k*M2, f1*M1+(k+1)*M2+2)    


    # Bonds between arms and core

    for k in range(f1):
        bond[ f1*(M1-1)+f2*(M2-1) + (k+1)-1, 1] = 1
        bond[ f1*(M1-1)+f2*(M2-1) + (k+1)-1, 2] = k * M1 + 2
        bond[ f1*(M1-1)+f2*(M2-1) + (k+1)-1, 3] = 1


    for k in range(f2):
        bond[ f1*(M1-1)+f2*(M2-1)+f1 + (k+1)-1, 1] = 1
        bond[ f1*(M1-1)+f2*(M2-1)+f1 + (k+1)-1, 2] = (f1*M1) + k * M2 + 2
        bond[ f1*(M1-1)+f2*(M2-1)+f1 + (k+1)-1, 3] = 1


    # Create LAMMPS data file                

    if file_name is None:
        file_name = 'polydisperse_star.data'
        
    with open(file_name, 'w') as f:
        
        f.write(f'LAMMPS data file for coarse grained star with (f1,M1)=({f1},{M1}) (f2,M2)=({f2},{M2})  \n\n')
        f.write('%1.0f atoms\n' % num_atoms)
        f.write('%1.0f atom types\n' % atomtypes)
        f.write('%1.0f bonds\n' % num_bonds)            
        f.write('%1.0f bond types\n' % bondtypes)
        f.write('0 angles\n')
        f.write('0 angle types\n')
        f.write('0 dihedrals\n')            
        f.write('0 dihedral types\n')
        f.write('%1.6f %1.6f xlo xhi\n' % (bxlo, bxhi))
        f.write('%1.6f %1.6f ylo yhi\n' % (bylo, byhi))        
        f.write('%1.6f %1.6f zlo zhi\n\n' % (bzlo, bzhi))
        
        f.write('Masses\n\n')
        f.write('%1.0f %1.6f\n' % (1, massA))
        f.write('%1.0f %1.6f\n\n'% (2, massB))
        f.write('Atoms\n\n')
        np.savetxt(f, coordinates, fmt='%4.0f %6.0f %6.0f %6.2f %12.6f %12.6f %12.6f')
        f.write('\n')
        f.write('Bonds\n\n')
        np.savetxt(f, bond, fmt='%4.0f %6.0f %6.0f %6.0f')