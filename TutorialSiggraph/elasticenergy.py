import numpy as np
from numpy.core.einsumfunc import einsum
from scipy import sparse
import igl

class ElasticEnergy:
    def __init__(self, young, poisson):
        '''
        Input:
        - young   : Young's modulus [Pa]
        - poisson : Poisson ratio
        '''
        self.young = young
        self.poisson = poisson
        self.lbda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        self.mu = young / (2 * (1 + poisson))

        self.E  = None
        self.dE = None
        self.P  = None
        self.dP = None
    
    def make_strain_tensor(self, jac):
        '''
        This method computes the strain tensor (#t, 3, 3), and stores it in self.E

        Input:
        - jac : jacobian of the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def make_differential_strain_tensor(self, jac, dJac):
        '''
        This method computes the differential of strain tensor (#t, 3, 3), 
        and stores it in self.dE

        Input:
        - jac  : jacobian of the deformation (#t, 3, 3)
        - dJac : differential of the jacobian of the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def make_piola_kirchoff_stress_tensor(self, jac):
        '''
        This method computes the stress tensor (#t, 3, 3), and stores it in self.P

        Input:
        - jac : jacobian of the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def make_differential_piola_kirchoff_stress_tensor(self, jac, dJac):
        '''
        This method computes the stress tensor (#t, 3, 3), and stores it in self.P

        Input:
        - jac  : jacobian of the deformation (#t, 3, 3)
        - dJac : differential of the jacobian of the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

class LinearElasticEnergy(ElasticEnergy):
    def __init__(self, young, poisson):
        super().__init__(young, poisson)

    def make_strain_tensor(self, jac):
        eye = np.zeros((len(jac), 3, 3))
        for i in range(3):
            eye[:, i, i] = 1
        
        # E = 1/2*(F + F^T) - I
        self.E = 0.5*(np.swapaxes(jac, 1, 2) + jac) - eye
        pass

    def make_differential_strain_tensor(self, jac, dJac):
        # dE = 1/2*(dF + dF^T)
        self.dE = 0.5 * (dJac + np.swapaxes(dJac, 1, 2))
        pass

    def make_piola_kirchoff_stress_tensor(self, jac):

        # First, update the strain tensor
        self.make_strain_tensor(jac)

        tr = np.einsum('ijj->i', self.E)
        eye = np.zeros((len(self.E), 3, 3))
        for i in range(3):
            eye[:, i, i] = tr 

        # P = 2*mu*E + lbda*tr(E)*I
        self.P = 2 * self.mu * self.E + self.lbda * eye
        pass

    def make_differential_piola_kirchoff_stress_tensor(self, jac, dJac):

        # First, update the differential of the strain tensor, 
        # and the strain tensor
        self.make_strain_tensor(jac)
        self.make_differential_strain_tensor(jac, dJac)

        # Diagonal matrices
        tr  = np.einsum('ijj->i', self.E)
        I   = np.zeros((len(self.F), 3, 3))
        dtr = np.einsum('ijj->i', self.dE)
        dI  = np.zeros((len(self.F), 3, 3))
        for i in range(3):
            I[:, i, i]  = tr
            dI[:, i, i] = dtr

        # dP = dF.(2*mu*E + lbda*tr(E)*I) + F.(2*mu*dE + lbda*tr(dE)*I)
        self.dP = (np.einsum('lij,ljk->lik', dJac, 2 * self.mu * self.E + self.lbda * I) +
                   np.einsum('lij,ljk->lik', jac, 2 * self.mu * self.dE + self.lbda * dI))
        pass

class KirchhoffElasticEnergy(ElasticEnergy):
    def __init__(self, young, poisson):
        super().__init__(young, poisson)

    def make_strain_tensor(self, jac):
        eye = np.zeros((len(jac), 3, 3))
        for i in range(3):
            eye[:, i, i] = 1

        # E = 1/2*(F^T.F - I)
        self.E = np.einsum('lij,ljk->lik', np.swapaxes(jac, 1, 2), jac) - eye
        pass

    def make_differential_strain_tensor(self, jac, dJac):
        # dE = 1/2*(dF^T.F + dF.F^T)
        self.dE = 0.5 * (np.einsum('lij,ljk->lik', np.swapaxes(dJac, 1, 2), jac) +
                         np.einsum('lij,ljk->lik', np.swapaxes(jac, 1, 2), dJac))
        pass

    def make_piola_kirchoff_stress_tensor(self, jac):

        # First, update the strain tensor
        self.make_strain_tensor(jac)

        tr = np.einsum('ijj->i', self.E)
        eye = np.zeros((len(self.E), 3, 3))
        for i in range(3):
            eye[:, i, i] = tr 

        # P = F.(2*mu*E + lbda*tr(E)*I)
        self.P = np.einsum('lij,ljk->lik', jac, 2 * self.mu * self.E + self.lbda * eye)
        pass

    def make_differential_piola_kirchoff_stress_tensor(self, jac, dJac):

        # First, update the differential of the strain tensor, 
        # and the strain tensor
        self.make_strain_tensor(jac)
        self.make_differential_strain_tensor(jac, dJac)

        # Diagonal matrices
        tr  = np.einsum('ijj->i', self.E)
        I   = np.zeros((len(self.F), 3, 3))
        dtr = np.einsum('ijj->i', self.dE)
        dI  = np.zeros((len(self.F), 3, 3))
        for i in range(3):
            I[:, i, i]  = tr
            dI[:, i, i] = dtr

        # dP = dF.(2*mu*E + lbda*tr(E)*I) + F.(2*mu*dE + lbda*tr(dE)*I)
        self.dP = (np.einsum('lij,ljk->lik', dJac, 2 * self.mu * self.E + self.lbda * I) +
                   np.einsum('lij,ljk->lik', jac, 2 * self.mu * self.dE + self.lbda * dI))
        pass


class NeoHookeanElasticEnergy(ElasticEnergy):
    def __init__(self, young, poisson):
        super().__init__(young, poisson)

    def make_strain_tensor(self, jac):
        eye = np.zeros((len(jac), 3, 3))
        for i in range(3):
            eye[:, i, i] = 1
        
        # E = 1/2*(F^T.F - I)
        self.E = np.einsum('lij,ljk->lik', np.swapaxes(jac, 1, 2), jac) - eye
        pass

    def make_differential_strain_tensor(self, jac, dJac):
        # dE = 1/2*(dF^T.F + dF.F^T)
        self.dE = 0.5 * (np.einsum('lij,ljk->lik', np.swapaxes(dJac, 1, 2), jac) +
                         np.einsum('lij,ljk->lik', np.swapaxes(jac, 1, 2), dJac))
        pass

    def make_piola_kirchoff_stress_tensor(self, jac):
        logJ = np.log(np.linalg.det(jac))
        # First invert, then transpose
        FinvT = np.swapaxes(np.linalg.inv(jac), 1, 2)

        # P = mu*(F - F^{-T}) + lbda*log(J)*F^{-T}
        self.P = (self.mu * (jac - FinvT) + self.lbda * np.einsum('i,ijk->ijk', logJ, FinvT))
        pass

    def make_differential_piola_kirchoff_stress_tensor(self, jac, dJac):

        # To be reused below
        logJ  = np.log(np.linalg.det(jac))
        Finv  = np.linalg.inv(jac)
        FinvT = np.swapaxes(Finv, 1, 2)
        Fprod = np.einsum("mij, mjk, mkl -> mil", FinvT, np.swapaxes(dJac, 1, 2), FinvT)
        trFinvdF = np.einsum("mij, mji -> m", Finv, dJac)

        # dP = mu*dF + (mu-lbda*log(J))*F^{-T}.dF^T.F^{-T} + lbda*tr(F^{-1}.dF)*F^{-T}
        self.dP = (self.mu * dJac + 
                   (self.mu - self.lbda * logJ) * Fprod + 
                   self.lbda * np.einsum("m, mij -> mij", trFinvdF, FinvT)
                  )
        pass