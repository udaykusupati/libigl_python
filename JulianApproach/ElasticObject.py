import numpy as np

class ElasticObject:
    '''
    Attributes:
    - v       : position of the vertices of the mesh (nV, 3)
    - t       : indices of the element's vertices (nE, 4)
    - rho     : mass per unit volume [kg.m-3]
    - young   : Young's modulus [Pa]
    - poisson : Poisson ratio [no unit]

    - v           : position of the deformed vertices of the mesh (nV, 3)
    - initVols    : np array of shape (nE,) containing the initial volume elements
    - vols        : np array of shape (nE,) containing the volume elements
    - initInvDisp : np array of shape (nE, 3, 3) containing the initial inverted undeformed displacement matrices
    - invDisp     : np array of shape (nE, 3, 3) containing the inverted undeformed displacement matrices
    '''

    def __init__(self, v, t, rho, young, poisson):
        self.v = v
        self.t = t

        self.rho = rho         # [kg.m-3]
        self.young = young     # [Pa]
        self.poisson = poisson # [no unit]

        self.lbda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        self.mu   = young / (2 * (1 + poisson))

        self.Precomputation()
        self.initVols       = self.vols.copy()
        self.initInvDisp    = self.invDisp.copy()
        self.initMassMatrix = self.ComputeMassMatrix(self.v)
        self.vDef = v

    def GetVolAndGradPhis(self, tIdx):
        '''
        Input:
        - tIdx : index of the tetrahedron

        Ouput:
        - vol   : volume of the element
        - grads : np array of shape (4, 3) containing the gradients ordered in the same way as in self.t[tIdx] 

              3
              *
             / \`.
            /   \ `* 2
           / _.--\ /
         0*-------* 1
        '''
        Xtet = self.v[self.t(tIdx)]
        gradDirs = np.zeros(shape=(4, 3))

        e31 = Xtet[1, :] - Xtet[3, :]
        e32 = Xtet[2, :] - Xtet[3, :]
        e01 = Xtet[1, :] - Xtet[0, :]
        e02 = Xtet[2, :] - Xtet[0, :]

        gradDirs[0, :] = np.cross(e32, e31)
        gradDirs[1, :] = np.cross(e32, e02)
        gradDirs[2, :] = np.cross(e01, e31)
        gradDirs[3, :] = np.cross(e01, e02)

        vol6  = - gradDirs[0, :] @ e01
        grads = gradDirs / vol6

        return vol6 / 6, grads

    def GetDeformationGradient(self, tIdx, grads):
        '''
        Input:
        - tIdx : index of the tetrahedron

        Ouput:
        - defGrad : np array of shape (3, 3) containing the gradients ordered in the same way as in self.t[tIdx] 
        '''

        defGrad = np.zeros(shape=(3, 3))
        for i in range(grads.shape[0]):
            defGrad += np.outer(grads[i], )





    def ComputeMassMatrix(self, vDef, lumped=True):
        '''
        Input:
        - vDef   : position of the vertices of the mesh (nV, 3)
        - lumped : whether we want a diagonal matrix in the end
        
        Output:
        - massMatrix : np array of shape (nV, nV) containing the volume elements
        '''
        
        nV         = self.v.shape[0]
        massMatrix = np.zeros(shape=(nV, nV))
        self.Precomputation(vDef) # updates self.vols
        
        # Matrices we add have the same structure and can be reused
        submatrix = 1/24 * (np.ones(shape=(4, 4)) + 2*np.eye(4))
        for i, tet in enumerate(self.t):
            # Construct all pairs of indices
            idxColumn, idxRow = np.meshgrid(tet, tet)
            massMatrix[idxRow, idxColumn] += self.vols[i] * submatrix
            
        if lumped:
            massMatrix = np.diag(np.sum(massMatrix, axis=1))
            
        return massMatrix

    def ComputeElasticForces(self, vDef):
        '''
        Input:
        - vDef    : position of the vertices of the deformed mesh (nV, 3)
        
        Output:
        - force : np array of shape (nV, 3) containing the force per vertex [N]
        '''
        
        nV = self.v.shape[0]
        forces = np.zeros(shape=(nV, 3))
        
        disp = np.zeros(shape=(3, 3))
        for i, tet in enumerate(self.t):
            Xtet = vDef[tet]
            disp[:, 0] = Xtet[0, :] - Xtet[3, :]
            disp[:, 1] = Xtet[1, :] - Xtet[3, :]
            disp[:, 2] = Xtet[2, :] - Xtet[3, :]
            
            # Jacobian of the deformation
            F = disp @ self.initInvDisp[i]
            
            # Linear elasticity
            strain = 0.5 * (F.T + F) - np.eye(3)
            
            # Piola-Kirchhoff stress tensor
            P = 2 * self.mu * strain + self.lbda * np.trace(strain) * np.eye(3)
            
            # Forces
            H = - self.vols[i] * P @ self.initInvDisp[i].T
            forces[tet[0], :] += H[:, 0]
            forces[tet[1], :] += H[:, 1]
            forces[tet[2], :] += H[:, 2]
            forces[tet[3], :] += - np.sum(H, axis=1)
            
        return forces

    def ComputeForceDifferentials(self, vDef, dvDef):
        '''
        Input:
        - vDef    : position of the vertices of the deformed mesh (nV, 3)
        - dvDef   : perturbation of the position of the vertices of the deformed mesh (nV, 3)
        
        Output:
        - dForces : np array of shape (nV, 3) containing the force differential per vertex [N]
        '''
        
        nV = vDef.shape[0]
        dForces = np.zeros(shape=(nV, 3))
        
        disp  = np.zeros(shape=(3, 3))
        dDisp = np.zeros(shape=(3, 3))
        for i, tet in enumerate(self.t):
            Xtet = vDef[tet]
            disp[:, 0] = Xtet[0, :] - Xtet[3, :]
            disp[:, 1] = Xtet[1, :] - Xtet[3, :]
            disp[:, 2] = Xtet[2, :] - Xtet[3, :]
            
            dXtet = dvDef[tet, :]
            dDisp[:, 0] = dXtet[0, :] - dXtet[3, :]
            dDisp[:, 1] = dXtet[1, :] - dXtet[3, :]
            dDisp[:, 2] = dXtet[2, :] - dXtet[3, :]
            
            # Jacobian of the deformation, and its differential
            F  = disp @ self.initInvDisp[i]
            dF = dDisp @ self.initInvDisp[i]
            
            # Differential of linear elasticity
            dStrain = 0.5 * (dF.T + dF)
            
            # Differential of Piola-Kirchhoff stress tensor
            dP = self.mu * dStrain + self.lbda * np.trace(dF) * np.eye(3)
            
            # Force differentials
            dH = - self.vols[i] * dP @ self.initInvDisp[i].T
            dForces[tet[0], :] += dH[:, 0]
            dForces[tet[1], :] += dH[:, 1]
            dForces[tet[2], :] += dH[:, 2]
            dForces[tet[3], :] += - np.sum(dH, axis=1)
            
        return dForces
