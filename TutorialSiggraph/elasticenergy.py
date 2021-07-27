import numpy as np
from scipy import sparse
import igl

class ElasticEnergy:
    def __init__(self, young, poisson):
        """
        Input:
        - v       : position of the vertices of the mesh (#v, 3)
        - t       : indices of the element's vertices (#t, 4)
        - rho     : mass per unit volume [kg.m-3]
        - young   : Young's modulus [Pa]
        - poisson : Poisson ratio
        - poisson : Damping factor
        - model   : 'linear', 'kirchhoff' or 'neo-hookean'
        """
        self.young = young
        self.poisson = poisson
        self.lbda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        self.mu = young / (2 * (1 + poisson))
    
    def make_strain_tensor(self, jac):
        '''
        Input:
        - jac : jacobian of the deformation (#t, 3, 3)

        Ouptut:
        - strain : strain tensor (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def make_piola_kirchoff_stress_tensor(self, jac, strain):
        '''
        Input:
        - jac    : jacobian of the deformation (#t, 3, 3)
        - strain : strain tensor (#t, 3, 3)

        Ouptut:
        - stress : stress tensor (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

class LinearElasticEnergy(ElasticEnergy):
    def __init__(self, mu, lbda):
        super().__init__(mu, lbda)

    def make_strain_tensor(self, jac):
        eye = np.zeros((len(jac), 3, 3))
        for i in range(3):
            eye[:, i, i] = 1
        
        # E = 1/2*(F + F^T) - I
        strain = 0.5*(np.swapaxes(jac, 1, 2) + jac) - eye
        return strain

    def make_piola_kirchoff_stress_tensor(self, jac, strain):
        tr = np.einsum('ijj->i', strain)
        eye = np.zeros((len(strain), 3, 3))
        for i in range(3):
            eye[:, i, i] = tr 

        # P = 2*mu*E + lbda*tr(E)*I
        stress = 2 * self.mu * strain + self.lbda * eye
        return stress

class KirchhoffElasticEnergy(ElasticEnergy):
    def __init__(self, mu, lbda):
        super().__init__(mu, lbda)

    def make_strain_tensor(self, jac):
        eye = np.zeros((len(jac), 3, 3))
        for i in range(3):
            eye[:, i, i] = 1

        # E = 1/2*(F^T.F - I)
        strain = np.einsum('lij,ljk->lik', np.swapaxes(jac, 1, 2), jac) - eye
        return strain

    def make_piola_kirchoff_stress_tensor(self, jac, strain):
        tr = np.einsum('ijj->i', strain)
        eye = np.zeros((len(strain), 3, 3))
        for i in range(3):
            eye[:, i, i] = tr 

        # P = F.(2*mu*E + lbda*tr(E)*I)
        stress = np.einsum('lij,ljk->lik', jac, 2 * self.mu * strain + self.lbda * eye)
        return stress


class NeoHookeanElasticEnergy(ElasticEnergy):
    def __init__(self, mu, lbda):
        super().__init__(mu, lbda)

    def make_strain_tensor(self, jac):
        eye = np.zeros((len(jac), 3, 3))
        for i in range(3):
            eye[:, i, i] = 1
        
        # E = 1/2*(F^T.F - I)
        strain = np.einsum('lij,ljk->lik', np.swapaxes(jac, 1, 2), jac) - eye
        return strain

    def make_piola_kirchoff_stress_tensor(self, jac, strain):
        I3 = np.log(np.linalg.det(jac)**2)
        # First invert, then transpose
        FinvT = np.swapaxes(np.linalg.inv(jac), 1, 2)

        # P = mu*(F - F^{-T}) + lbda*log(I3)/2*F^{-T}
        stress = (self.mu * (jac - FinvT)
                  + 0.5 * self.lbda * np.einsum('i,ijk->ijk', I3, FinvT))
        return stress