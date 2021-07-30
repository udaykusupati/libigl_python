import numpy as np
from scipy import sparse
import igl
from Utils import conjugate_gradient


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                             ELASTIC SOLID CLASS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class ElasticSolid(object):

    def __init__(self, v, t, ee, rho=1, damping=1e-2, pin_idx=[]):
        '''
        Input:
        - v       : position of the vertices of the mesh (#v, 3)
        - t       : indices of the element's vertices (#t, 4)
        - ee      : elastic energy used
        - rho     : mass per unit volume [kg.m-3]
        - gamma   : Damping factor
        - pin_idx : list of vertex indices to pin
        '''

        self.v     = v
        self.t     = t
        self.ee    = ee
        self.rho   = rho
        self.gamma = damping
        self.pin_idx = pin_idx

        # Mask for pin constraints
        self.pin_mask = np.array([idx not in self.pin_idx 
                                  for idx in range(self.v.shape[0])]).reshape(-1, 1)

        self.W  = None # (#t,)
        self.Ds = None
        self.dv = None
        self.F  = None
        self.f  = None
        self.W0 = None # (#t,)
        self.Bm = None
        self.M  = None

        self.update_shape(v)
        # In case we want to change the initial velocity, we should multiply by the mask
        self.velocity = self.pin_mask * np.zeros((len(self.v), 3))


    def vertex_tet_sum(self, data):
        '''
        Input:
        - data : np array of shape (#t,) or (4*#t,)

        Output:
        - m : np array of shape (#v,)
        '''
        i = self.t.flatten('F')        # (4*#t,)
        j = np.arange(len(self.t))     # (#t,)
        j = np.tile(j, 4)              # (4*#t,)

        if len(data) == len(self.t):
            data = data[j]

        # Has shape (#v, #t)
        m = sparse.coo_matrix((data, (i, j)), (len(self.v), len(self.t)))
        return np.array(m.sum(axis=1)).flatten()

    def displace(self, v_disp):
        '''
        Input:
        - v_disp   : displacement of the vertices of the mesh (#v, 3)
        '''
        self.update_shape(self.v + self.pin_mask * v_disp)

    def update_shape(self, v_def):
        '''
        Updates the displacement, the Jacobian of the displacement, and the 
        resulting elastic forces. If called for the first time, this also 
        computes the initial mass matrix from the volume of the elements.

        Input:
        - v_def : position of the vertices of the mesh (#v, 3)
        '''
        self.v = v_def
        self.make_shape_matrix()
        self.W = abs(np.linalg.det(self.Ds)) / 6
        if self.W0 is None:
            self.W0 = self.W.copy()
            self.make_mass_matrix()
        self.make_jacobian()
        # This also updates the stress tensor for the ee
        self.make_elastic_forces()

    def make_mass_matrix(self):
        '''
        Make a diagonal mass matrix (#v, #v)
        '''
        M = self.vertex_tet_sum(self.W)
        self.M = 1 / 4 * M * self.rho

    def make_shape_matrix(self):
        '''
        Construct Ds that has shape (#t, 3, 3), and its inverse Bm
        '''
        e1 = (self.v[self.t[:, 0]] - self.v[self.t[:, 3]])
        e2 = (self.v[self.t[:, 1]] - self.v[self.t[:, 3]])
        e3 = (self.v[self.t[:, 2]] - self.v[self.t[:, 3]])
        Ds = np.stack((e1, e2, e3))
        Ds = np.swapaxes(Ds, 0, 1)
        Ds = np.swapaxes(Ds, 1, 2)
        self.Ds = Ds
        if self.Bm is None:
            self.Bm = np.linalg.inv(self.Ds)

    def make_jacobian(self):
        '''
        Compute the current Jacobian of the deformation
        '''
        self.F = np.einsum('lij,ljk->lik', self.Ds, self.Bm)

    def make_elastic_forces(self):
        '''
        This method updates the elastic forces stored in self.f (#v, 3, 3) 
        '''

        # First update strain/stress tensor, stored in self.ee
        self.ee.make_piola_kirchhoff_stress_tensor(self.F)

        # H[el] = - W0[el]*P.Bm[el]^T
        H = np.einsum('lij,ljk->lik', self.ee.P, np.swapaxes(self.Bm, 1, 2))
        H = - np.einsum('i,ijk->ijk', self.W0, H)

        # Extract forces from H of shape (#t, 3, 3)
        # We look at each component separately, then stack them in a vector of shape (4*#t,)
        # Then we distribute the contributions to each vertex
        fx = self.vertex_tet_sum(np.hstack((H[:, 0, 0], H[:, 0, 1], H[:, 0, 2],
                                            -H[:, 0, 0] - H[:, 0, 1] - H[:, 0, 2])))
        fy = self.vertex_tet_sum(np.hstack((H[:, 1, 0], H[:, 1, 1], H[:, 1, 2],
                                            -H[:, 1, 0] - H[:, 1, 1] - H[:, 1, 2])))
        fz = self.vertex_tet_sum(np.hstack((H[:, 2, 0], H[:, 2, 1], H[:, 2, 2],
                                            -H[:, 2, 0] - H[:, 2, 1] - H[:, 2, 2])))
        
        # We stack them in a tensor of shape (#v, 3)
        self.f = np.column_stack((fx, fy, fz))

    def von_mises(self):
        '''
        Computes von Mises stress at each vertex, this assumes that the 
        stress tensor is updated in self.ee

        Output:
        - VM : array of shape (#v,)
        '''
        VM = (.5 * (self.ee.P[:, 0, 0] - self.ee.P[:, 1, 1]) ** 2 +
              .5 * (self.ee.P[:, 0, 0] - self.ee.P[:, 2, 2]) ** 2 +
              .5 * (self.ee.P[:, 1, 1] - self.ee.P[:, 2, 2]) ** 2 +
              3 * (self.ee.P[:, 0, 1]**2 + self.ee.P[:, 0, 2]**2 +
                   self.ee.P[:, 1, 2]**2)) ** .5
        return np.abs(self.vertex_tet_sum(VM))

    def stress_norm(self):
        '''
        Computes the total stress norm, this assumes that the 
        stress tensor is updated in self.ee

        Output:
        - VM : scalar value
        '''
        VM = np.linalg.norm(self.ee.P, axis=(1, 2))
        return np.abs(self.vertex_tet_sum(VM))

    def compute_force_differentials(self, v_disp):
        '''
        Input:
        - v_disp   : displacement of the vertices of the mesh (#v, 3)

        Output:
        - df    : force differentials at the vertices of the mesh (#v, 3)
        '''

        # Compute the displacement differentials
        d1 = (v_disp[self.t[:, 0]] - v_disp[self.t[:, 3]])
        d2 = (v_disp[self.t[:, 1]] - v_disp[self.t[:, 3]])
        d3 = (v_disp[self.t[:, 2]] - v_disp[self.t[:, 3]])
        dDs = np.stack((d1, d2, d3))
        dDs = np.swapaxes(dDs, 0, 1)
        dDs = np.swapaxes(dDs, 1, 2)

        # Differential of the Jacobian
        dF = np.einsum('lij,ljk->lik', dDs, self.Bm)

        # Differential of the stress tensor
        self.ee.make_differential_piola_kirchhoff_stress_tensor(self.F, dF)
        
        # Differential of the forces
        dH = np.einsum('lij,ljk->lik', self.ee.dP, np.swapaxes(self.Bm, 1, 2))
        dH = np.einsum('i,ijk->ijk', self.W0, dH)
        # Same as for the elastic forces
        dfx = self.vertex_tet_sum(np.hstack((dH[:, 0, 0], dH[:, 0, 1], dH[:, 0, 2],
                                             -dH[:, 0, 0] - dH[:, 0, 1] - dH[:, 0, 2])))
        dfy = self.vertex_tet_sum(np.hstack((dH[:, 1, 0], dH[:, 1, 1], dH[:, 1, 2],
                                             -dH[:, 1, 0] - dH[:, 1, 1] - dH[:, 1, 2])))
        dfz = self.vertex_tet_sum(np.hstack((dH[:, 2, 0], dH[:, 2, 1], dH[:, 2, 2],
                                             -dH[:, 2, 0] - dH[:, 2, 1] - dH[:, 2, 2])))
        
        # We stack them in a tensor of shape (#v, 3)
        return np.column_stack((dfx, dfy, dfz))

    def explicit_integration_step(self, dt=1e-6):
        '''
        Input:
        - dt : discretization step
        '''

        # Invert the matrix on the LHS (#v, #v)
        D = np.diag((self.M + self.gamma * dt) ** -1)
        self.velocity += dt * self.pin_mask * np.einsum('ij,jk->ik', D, self.f)
        v_disp         = self.velocity * dt
        self.displace(v_disp) # Pinning is taken care of here as well
        # print(np.mean(np.linalg.norm(self.f.reshape(-1, 9), axis=-1)))

    def implicit_integration_step(self, dt=1e-6, steps=2):
        '''
        Input:
        - dt    : discretization step
        - steps : number of steps taken per time step
        '''

        new_velocity = self.velocity.copy()
        print("New time step")
        for step in range(steps):
            fd = -self.gamma * self.compute_force_differentials(new_velocity)
            ft = fd + self.f
            print("Damping: {}".format(np.mean(np.linalg.norm(fd.reshape(-1, 9), axis=-1))))
            print("Elastic: {}".format(np.mean(np.linalg.norm(self.f.reshape(-1, 9), axis=-1))))
            print("Total: {}".format(np.mean(np.linalg.norm(ft.reshape(-1, 9), axis=-1))))
            M = np.diag(self.M)

            def LHS(dv):
                return (1 / dt ** 2 * np.einsum('ij,jk->ik', M, dv) +
                        (1 + self.gamma / dt) * self.compute_force_differentials(dv))

            RHS = 1 / dt * np.einsum('ij,jk->ik', M, self.velocity - new_velocity) + ft
            # New displacement
            dv = self.pin_mask * conjugate_gradient(LHS, RHS)
            print("RHS: {}".format(np.mean(np.linalg.norm(RHS.reshape(-1, 9), axis=-1))))
            print("Velocity: {}".format(np.mean(np.linalg.norm(dv.reshape(-1, 9), axis=-1))))
            self.displace(dv)
            new_velocity += dv / dt
        self.velocity = new_velocity
        print()

# -----------------------------------------------------------------------------
#                                    Test
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    from mayavi import mlab
    from elasticenergy import *

    v, t = igl.read_msh("meshes/ball.msh")

    # Find the lowest vertex and pin it
    pin_idx = [np.argmin(v[:, 2])]
    print(pin_idx)

    # ee = LinearElasticEnergy(1e6, 0.2)
    ee = CorotatedElasticEnergy(1e6, 0.2)
    # ee = NeoHookeanElasticEnergy(1e6, 0.2)
    S  = ElasticSolid(v, t, ee, rho=10, damping=0., pin_idx=pin_idx)

    # initial deformation

    v_def = v.copy()

    v_def[:, 2] *= 1.5

    S.update_shape(v_def)
    
    # # extract the boundary 2D mesh
    # tb = igl.boundary_facets(t)
    # # plot the 2D mesh
    # M = mlab.triangular_mesh(S.v[:, 0], S.v[:, 1], S.v[:, 2], tb)
    # mlab.show()

    if True:

        # iterations of simulation
        iterations = 1000

        # save the images for video
        save_video = False

        @mlab.animate(delay=10)
        def simulate():
            # extract the boundary 2D mesh
            tb = igl.boundary_facets(t)
            # plot the 2D mesh
            M = mlab.triangular_mesh(S.v[:, 0], S.v[:, 1], S.v[:, 2], tb)
            for i in range(iterations):
                S.explicit_integration_step(dt=1e-5)
                # S.implicit_integration_step(dt=1e-3, steps=4)
                M.mlab_source.reset(x=S.v[:, 0], y=S.v[:, 1], z=S.v[:, 2])
                yield
                if save_video:
                    # to be improved for background color
                    prefix = len(str(int(iterations))) - len(str(i)) + 1
                    n_seq = '0' * prefix + str(i)
                    mlab.savefig(filename='img_{}.png'.format(n_seq),
                                 size=None)

        simulate()

        mlab.show()
