# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 22:13:15 2020

This module implements FVM on the 1D steady state convection-diffusion equation
with 3 discretisation schemes: 
    
        -Central Differencing 
        -Upwind Differencing 
        -Power-Law Differencing 
        
The resulting system of equations is solved by TDMA. 


@author: Chengming Zhang， Dec.2020 
"""

from numpy import *
import matplotlib.pyplot as plt


class FVM(object): 
    '''
    The main interface of this module. 
    '''
    def __init__(self, L, n, u, gamma, rho):
        '''
        Parameters
        ----------
        L :      The length of the domain 
        n :      Number of the nodes 
        u :      Field velocity 
        gamma :  Diffusion coefficient 
        rho :    Fluid density        
        '''
        self.L = L 
        self.n = n 
        self.u = u 
        self.gamma = gamma 
        self.rho = rho     
        
        # Mesh generation 
        nodes = linspace(0, L, n)
        nodes2 = linspace(0,L,100)
        self.nodes = array(nodes)
        self.nodes2 = array(nodes2)
        
        self.dx = self.nodes[2] - self.nodes[1]
        
        # Generate velocity field 
        self.velocity = self.u*self.nodes
        
        
    @property 
    def CDS(self):
        '''
        Central differece scheme 
        
        Returns
        -------
        matrix : n x n coefficient matrix 
        phi :    n x 1 arrary property that need to be solved  
        const :  n x 1 constant array 

        '''
        
        # Generate coefficient matrices
        ae = self.gamma/self.dx - 0.5 * self.rho * self.u
        aw = self.gamma/self.dx + 0.5 * self.rho * self.u
        ap = ae + aw + (self.rho*self.u - self.rho*self.u)
        
        matrix_dia = ap * eye(self.n)
        matrix_up = -ae * eye(self.n,k=1)
        matrix_low = -aw * eye(self.n,k=-1)
        
        matrix = matrix_dia + matrix_up + matrix_low 
        
        phi = zeros([self.n,1])
        const = zeros([self.n,1]) 
        
        #apply BCs 
        phi[0] = 100 
        phi[-1] = 20
        
        return matrix, phi, const 
    
    @property 
    def UDS(self):
        '''
        Upwind differece scheme 
        
        Returns
        -------
        matrix : n x n coefficient matrix 
        phi :    n x 1 arrary property that need to be solved  
        const :  n x 1 constant array 

        '''
        # Generate coefficient matrices 
        ae = self.gamma/self.dx + max(-self.rho*self.u, 0)
        aw = self.gamma/self.dx + max( self.rho *self.u, 0)
        ap = ap = ae + aw + (self.rho*self.u - self.rho*self.u)
        
        matrix_dia = ap * eye(self.n)
        matrix_up = -ae * eye(self.n,k=1)
        matrix_low = -aw * eye(self.n,k=-1)
        
        matrix = matrix_dia + matrix_up + matrix_low 
        
        phi = zeros([self.n,1])
        const = zeros([self.n,1])
        
        # Apply boundary conditions
        phi[0] = 100 
        phi[-1] = 20

        return matrix, phi, const  
    
    @property
    def PLDS(self): 
        '''
        Power law differece scheme 
        
        Returns
        -------
        matrix : n x n coefficient matrix 
        phi :    n x 1 arrary property that need to be solved  
        const :  n x 1 constant array 

        '''
        
        Pe = self.rho * self.u * self.dx / self.gamma
        D  = self.gamma / self.dx 
        F = self.rho * self.u 
        
        # Generate coefficient matrices 
        ae = D * max((1-0.1*abs(Pe))**5, 0 ) + max( -F, 0 )
        aw = D * max((1-0.1*abs(Pe))**5, 0 ) + max( F, 0 )
        ap = ap = ae + aw + (F - F)
        
        matrix_dia = ap * eye(self.n)
        matrix_up = -ae * eye(self.n,k=1)
        matrix_low = -aw * eye(self.n,k=-1)
        
        matrix = matrix_dia + matrix_up + matrix_low 
        
        phi = zeros([self.n,1])
        const = zeros([self.n,1])
        
        # Apply boundary conditions
        phi[0] = 100 
        phi[-1] = 20

        return matrix, phi, const  
    
    
    def TDMA(self, matrix, phi, const):
        '''
        Tridiagonal matrix algorithm

        Parameters
        ----------
        matrix : n x n coefficient matrix 
        phi :    n x 1 arrary property that need to be solved  
        const :  n x 1 constant array
        
        Returns
        -------
        phi : n x 1 solved array 

        '''
        #set P1 = b1/a1, Q1 = d1/a1
        P = zeros([self.n,1])
        Q = zeros([self.n,1])
        
        P[0] = 0 
        Q[0] = phi[0]
        
        for i in range(1, self.n-1):
            P[i] = -matrix[i, i+1]/(matrix[i,i]+matrix[i, i-1]*P[i-1])
            Q[i] = (const[i] - matrix[i, i-1]*Q[i-1])/(matrix[i,i]+ \
                                                     matrix[i,i-1]*P[i-1])
        
        P[-1] = 0
        Q[-1] = phi[-1]
        
        for i in range(0, self.n-1)[::-1]: 
            phi[i] = P[i]*phi[i+1] + Q[i]
            
        return phi 
    
    @property
    def exactsol(self):

       self.phi_ana = 100 + ((exp(self.rho*self.u*self.nodes/self.gamma)-1)/(exp\
                                (self.rho*self.u*self.L/self.gamma)-1))*(20-100)
        
       return self.phi_ana
    
        
       

def plotsolu(L, n, phi, phi_ana, scheme = '', u = '', dx = ''):
    
    line1, = plt.plot(linspace(0, L, n), phi, label=scheme, marker= 'o' , \
                                                       c='r', linestyle='--')
    
    line2, = plt.plot(linspace(0, 1, n), phi_ana, label="Exact", c='k')
    
    plt.legend(loc='lower left')
    plt.xlim(0,1)
    #plt.ylim(20, 100)
    plt.xlabel('L')
    plt.ylabel('phi')
    plt.title(scheme+', dx='+dx+', u='+u)
    
 
def error_c(phi, phi_ana, node_num):
    
    diff = phi.T-phi_ana 
    error = sum(abs(diff/phi_ana))/(node_num-1)*100

    return error        
    
        
if __name__ == '__main__':
    
    node_num = array([5,9,26,51])
    spacing = 1/(node_num-1)
    
    #--------------------CDS--------------------------------------
    
    for num in node_num:
    
        solver = FVM(L=1, n=num, u=0.1, gamma=0.1, rho=1)
        matrix, phi, const = solver.CDS
        phi= solver.TDMA(matrix, phi, const)
        phi_ana = solver.exactsol
        plt.figure()
        dx = str(1/(num-1))
        plotsolu(1, num, phi, phi_ana, scheme='CDS', u ='0.1',dx = dx)
    
    for num in node_num:
    
        solver = FVM(L=1, n=num, u=2.5, gamma=0.1, rho=1)
        matrix, phi, const = solver.CDS
        phi= solver.TDMA(matrix, phi, const)
        phi_ana = solver.exactsol
        plt.figure()
        dx = str(1/(num-1))
        plotsolu(1, num, phi, phi_ana, scheme='CDS', u ='2.5',dx = dx)
    
    #--------------------UDS--------------------------------------
    
    for num in node_num:
    
        solver = FVM(L=1, n=num, u=0.1, gamma=0.1, rho=1)
        matrix, phi, const = solver.UDS
        phi= solver.TDMA(matrix, phi, const)
        phi_ana = solver.exactsol
        plt.figure()
        dx = str(1/(num-1))
        plotsolu(1, num, phi, phi_ana, scheme='UDS', u ='0.1',dx = dx)
    
    for num in node_num:
    
        solver = FVM(L=1, n=num, u=2.5, gamma=0.1, rho=1)
        matrix, phi, const = solver.UDS
        phi= solver.TDMA(matrix, phi, const)
        phi_ana = solver.exactsol
        plt.figure()
        dx = str(1/(num-1))
        plotsolu(1, num, phi, phi_ana, scheme='UDS', u ='2.5',dx = dx)
    
    #--------------------PLDS--------------------------------------

    for num in node_num:
    
        solver = FVM(L=1, n=num, u=0.1, gamma=0.1, rho=1)
        matrix, phi, const = solver.PLDS
        phi= solver.TDMA(matrix, phi, const)
        phi_ana = solver.exactsol
        plt.figure()
        dx = str(1/(num-1))
        plotsolu(1, num, phi, phi_ana, scheme='PLDS', u ='0.1',dx = dx)
    
    for num in node_num:
    
        solver = FVM(L=1, n=num, u=2.5, gamma=0.1, rho=1)
        matrix, phi, const = solver.PLDS
        phi= solver.TDMA(matrix, phi, const)
        phi_ana = solver.exactsol
        plt.figure()
        dx = str(1/(num-1))
        plotsolu(1, num, phi, phi_ana, scheme='PLDS', u ='2.5',dx = dx)
        
    #--------------------Error--------------------------------------
    
    # u=0.1  
    error_list_CDS = []
    for num in node_num:
        solver = FVM(L=1, n=num, u=0.1, gamma=0.1, rho=1)
        matrix, phi, const = solver.CDS
        phi= solver.TDMA(matrix, phi, const)
        phi_ana = solver.exactsol
        error = error_c(phi, phi_ana, num)
        error_list_CDS.append(error)
        
    error_vec_CDS = array(error_list_CDS)
    plt.figure()
    plt.plot(spacing, error_vec_CDS, label='CDS', marker= 'o' , \
                                                       c='r', linestyle='--')
    
    
    error_list_UDS = []
    for num in node_num:
        solver = FVM(L=1, n=num, u=0.1, gamma=0.1, rho=1)
        matrix, phi, const = solver.UDS
        phi= solver.TDMA(matrix, phi, const)
        phi_ana = solver.exactsol
        error = error_c(phi, phi_ana, num)
        error_list_UDS.append(error)
    
    error_vec_UDS = array(error_list_UDS)  
    print(error_vec_UDS)      
    plt.plot(spacing, error_vec_UDS, label='UDS', marker= 'o' , \
                                                       c='b', linestyle='--')
        
    error_list_PLDS = []
    for num in node_num:
        solver = FVM(L=1, n=num, u=0.1, gamma=0.1, rho=1)
        matrix, phi, const = solver.PLDS
        phi= solver.TDMA(matrix, phi, const)
        phi_ana = solver.exactsol
        error = error_c(phi, phi_ana, num)
        error_list_PLDS.append(error)
    
    error_vec_PLDS = array(error_list_PLDS)        
    plt.plot(spacing, error_vec_PLDS, label='PLDS', marker= 'o' , \
                                                       c='g', linestyle='--')
    plt.title('Error of different schemes vs various dx (u=0.1)')
    plt.legend()
    plt.xlabel('dx')
    plt.ylabel('Error (%)')
    
    # u=2.5
    error_list_CDS = []
    for num in node_num:
        solver = FVM(L=1, n=num, u=2.5, gamma=0.1, rho=1)
        matrix, phi, const = solver.CDS
        phi= solver.TDMA(matrix, phi, const)
        phi_ana = solver.exactsol
        error = error_c(phi, phi_ana, num)
        error_list_CDS.append(error)
        
    error_vec_CDS = array(error_list_CDS)
    plt.figure()
    plt.plot(spacing, error_vec_CDS, label='CDS', marker= 'o' , \
                                                       c='r', linestyle='--')
    
    
    error_list_UDS = []
    for num in node_num:
        solver = FVM(L=1, n=num, u=2.5, gamma=0.1, rho=1)
        matrix, phi, const = solver.UDS
        phi= solver.TDMA(matrix, phi, const)
        phi_ana = solver.exactsol
        error = error_c(phi, phi_ana, num)
        error_list_UDS.append(error)
    
    error_vec_UDS = array(error_list_UDS)        
    plt.plot(spacing, error_vec_UDS, label='UDS', marker= 'o' , \
                                                       c='b', linestyle='--')
        
    error_list_PLDS = []
    for num in node_num:
        solver = FVM(L=1, n=num, u=2.5, gamma=0.1, rho=1)
        matrix, phi, const = solver.PLDS
        phi= solver.TDMA(matrix, phi, const)
        phi_ana = solver.exactsol
        error = error_c(phi, phi_ana, num)
        error_list_PLDS.append(error)
    
    error_vec_PLDS = array(error_list_PLDS)        
    plt.plot(spacing, error_vec_PLDS, label='PLDS', marker= 'o' , \
                                                       c='g', linestyle='--')
    plt.title('Error of different schemes vs various dx (u=2.5)')
    plt.legend()
    plt.xlabel('dx')
    plt.ylabel('Error (%)')