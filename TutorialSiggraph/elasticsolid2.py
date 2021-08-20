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

    def __init__(self, v, t, ee, rho=1, damping=1e-2, pin_idx=[], f_ext=None, self_weight=False):
        '''
        Input:
        - v           : position of the vertices of the mesh (#v, 3)
        - t           : indices of the element's vertices (#t, 4)
        - ee          : elastic energy used
        - rho         : mass per unit volume [kg.m-3]
        - gamma       : damping factor
        - pin_idx     : list of vertex indices to pin
        - f_ext       : external forces (#v, 3)
        - self_weight : whether we add self weight or not
        '''

        self.v           = v
        self.t           = t
        self.ee          = ee
        self.rho         = rho
        self.gamma       = damping
        self.pin_idx     = pin_idx
        self.self_weight = self_weight

        # Make sure f_ext has the same shape as v
        if f_ext is None:
            f_ext = np.zeros_like(v)
        else:
            assert f_ext.shape == v.shape
        self.f_ext = f_ext

        # Mask for pin constraints
        self.pin_mask = np.array([idx not in self.pin_idx 
                                  for idx in range(self.v.shape[0])]).reshape(-1, 1)

        self.W  = None # (#t,)
        self.Dm = None
        self.Ds = None
        self.dv = None
        self.F  = None
        self.dF = None
        self.f  = None
        self.W0 = None # (#t,)
        self.Bm = None
        self.M  = None # (#v,)

        self.update_shape(v)
        # In case we want to change the initial velocity, we should multiply by the mask
        self.velocity = self.pin_mask * np.zeros((len(self.v), 3))

        if self.self_weight:
            gravity       = np.zeros_like(v)
            gravity[:, 2] = -9.81 #m.s-2
            self.f_ext   += self.M.reshape(-1, 1) * gravity

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
        Updates the vertex position, the Jacobian of the deformation, and the 
        resulting elastic forces. If called for the first time, this also 
        computes the initial mass matrix from the volume of the elements.

        Input:
        - v_def : position of the vertices of the mesh (#v, 3)
        '''

        # Can only change the unpinned ones
        self.v = (1. - self.pin_mask) * self.v + self.pin_mask * v_def
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
        M = self.vertex_tet_sum(self.W0)
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
            self.Dm = Ds.copy()
            self.Bm = np.linalg.inv(self.Dm)

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
        This computes the differential of the force given a displacement dx,
        where df = df/dx|x . dx = - K(x).dx. The matrix vector product K(x)w
        is then given by the call self.compute_force_differentials(-w).

        Input:
        - v_disp   : displacement of the vertices of the mesh (#v, 3)

        Output:
        - df    : force differentials at the vertices of the mesh (#v, 3)
        '''

        # Compute the displacement differentials
        d1 = (v_disp[self.t[:, 0]] - v_disp[self.t[:, 3]]).reshape(-1, 3, 1)
        d2 = (v_disp[self.t[:, 1]] - v_disp[self.t[:, 3]]).reshape(-1, 3, 1)
        d3 = (v_disp[self.t[:, 2]] - v_disp[self.t[:, 3]]).reshape(-1, 3, 1)
        dDs = np.concatenate((d1, d2, d3), axis=2)
        
        # Differential of the Jacobian
        self.dF = np.einsum('lij,ljk->lik', dDs, self.Bm)

        # Differential of the stress tensor
        self.ee.make_differential_piola_kirchhoff_stress_tensor(self.F, self.dF)
        
        # Differential of the forces
        dH = np.einsum('lij,ljk->lik', self.ee.dP, np.swapaxes(self.Bm, 1, 2))
        dH = - np.einsum('i,ijk->ijk', self.W0, dH)

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
        self.velocity += dt * self.pin_mask * np.einsum('ij,jk->ik', D, self.f + self.f_ext)
        v_disp         = self.velocity * dt
        self.displace(v_disp) # Pinning is taken care of here as well

    '''TODO'''
    def implicit_integration_step(self, dt=1e-6, steps=2):
        '''
        Input:
        - dt    : discretization step
        - steps : number of steps taken per time step
        '''

        # Provide the displacement as the initial guess to the conjugate gradient method
        self.explicit_integration_step(dt=dt)

        prev_pos = self.v.copy()
        print("New time step")
        for step in range(steps):
            # fd = -self.gamma * self.compute_force_differentials(new_velocity)
            ft = self.f #+ self.f_ext
            ft = 0.
            # print("Damping: {}".format(np.mean(np.linalg.norm(fd, axis=-1))))
            # print("Elastic: {}".format(np.mean(np.linalg.norm(self.f axis=-1))))
            # print("Total: {}".format(np.mean(np.linalg.norm(ft, axis=-1))))

            # def LHS(dv):
            #     return (1 / dt ** 2 * np.einsum('i,ij->ij', self.M, dv) +
            #             (1 + self.gamma / dt) * self.compute_force_differentials(dv))

            def LHS(dx):
                return (1 / dt ** 2 * np.einsum('i,ij->ij', self.M, dx) )

            # def LHS(dx):
            #     return (1 / dt ** 2 * np.einsum('i,ij->ij', self.M, dx) + self.compute_force_differentials(-dx))

            # RHS = 1 / dt * np.einsum('i,ij->ij', self.M, self.velocity - new_velocity) + ft
            RHS = 1 / dt * np.einsum('i,ij->ij', self.M, self.velocity) + ft

            # New displacement
            dx = self.pin_mask * conjugate_gradient(LHS, RHS)
            print("Residuals: {}".format(np.linalg.norm(LHS(dx) - RHS)))
            # print("RHS: {}".format(np.mean(np.linalg.norm(RHS, axis=-1))))
            # print("Correction: {}".format(np.mean(np.linalg.norm(dx, axis=-1))))
            # print("Velocity: {}".format(np.mean(np.linalg.norm(new_velocity, axis=-1))))
            
            self.v += dx
            self.make_shape_matrix()
            # self.displace(dx) # Updates the force too
        
        new_velocity = (self.v - prev_pos) / dt

        # print("Velocity mean: {}".format(np.mean(np.linalg.norm(new_velocity, axis=-1))))
        # print("Velocity min: {}".format(np.min(np.linalg.norm(new_velocity, axis=-1))))
        # print("Velocity max: {}".format(np.max(np.linalg.norm(new_velocity, axis=-1))))
        # print("Velocity std: {}".format(np.std(np.linalg.norm(new_velocity, axis=-1))))

        # print("Displacement mean: {}".format(np.mean(np.linalg.norm(self.v - prev_pos, axis=-1))))
        # print("Speed mean: {}".format(np.mean(np.linalg.norm(self.v - prev_pos, axis=-1)) / dt))
        # print("Velocity mean: {}".format(np.mean(np.linalg.norm(new_velocity, axis=-1)) / dt))

        # print("Displacement ok: {}".format(np.allclose(tot_disp, self.v - prev_pos)))
        # print("Velocity ok: {}".format(np.allclose((new_velocity - self.velocity)*dt, self.v - prev_pos)))
        # print("Velocity ok (bis): {}".format(np.allclose(new_velocity*dt, self.v - prev_pos)))

        self.velocity = new_velocity.copy()
        # self.velocity = (self.v - prev_pos) / dt


        print()

    def assemble_stiffness(self):
        K = np.zeros(shape=(self.v.shape[0], self.v.shape[0]))
        for i in range(self.v.shape[0]):
            e_i = np.zeros(shape=(self.v.shape[0], 3))
            e_i[i, 0] = 1.
            K[:, i] = self.compute_force_differentials(-e_i)[:, 0]

        
        return K


    def equilibrium_step(self, tet):

        ft = self.f + self.f_ext
        # ft = self.f_ext

        # Solve [K(x)+alpha 1.1^T]dx = f_tot (Newton step on total energy)
        def LHS(dx):
            dx[tet[0]] = 0.
            dx[tet[1], 1:] = 0.
            dx[tet[2], 2] = 0.
            df = self.compute_force_differentials(-dx)
            df[tet[0]] = 0.
            df[tet[1], 1:] = 0.
            df[tet[2], 2] = 0.
            return df
        
        ft[tet[0]] = 0.
        ft[tet[1], 1:] = 0.
        ft[tet[2], 2] = 0.
        RHS = ft

        K = self.assemble_stiffness()
        K[tet[0]] = 0.
        K[tet[1], 1:] = 0.
        K[tet[2], 2] = 0.
        K[:, tet[0]] = 0.
        K[1:, tet[1]] = 0.
        K[2, tet[2]] = 0.
        # Build pseudo inverse
        eigval, eigvec = np.linalg.eigh(K)
        print(eigval[:15])
        # eigValInv = np.zeros_like(eigval)
        # eigValInv[1:] = 1./eigval[1:]
        # pinvK = eigvec @ np.diag(eigValInv) @ eigvec.T
        # dx = pinvK @ RHS
        # dx = np.linalg.lstsq(K, RHS)[0]

        # New displacement
        dx0 = 1e-5 * np.random.rand(self.v.shape[0], 3)
        dx0[tet[0]] = 0.
        dx0[tet[1], 1:] = 0.
        dx0[tet[2], 2] = 0.
        dx = self.pin_mask * conjugate_gradient(LHS, RHS, x0=dx0)
        # print("Residuals with force diff: {}".format(np.linalg.norm(LHS(dx) - RHS)))
        # print(np.ones(shape=(self.v.shape[0])) @ dx / self.v.shape[0] / np.linalg.norm(dx, axis=0, keepdims=True))
        print("Residuals with K: {}".format(np.linalg.norm(K @ dx - RHS)))
        print()

        self.displace(0.1 * dx) # Updates the force too


# -----------------------------------------------------------------------------
#                                    Test
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    from mayavi import mlab
    from elasticenergy import *

    v, t = igl.read_msh("meshes/ball.msh")

    # Some characteristics
    rho     = 10000  # [kg.m-3] 10000kg.m-3 to see gravity effects
    damping = 0.
    young   = 1e6 # [Pa] 
    poisson = 0.2
    dt      = 1e-3

    # Find some of the lowest vertices and pin them
    pin_idx = []
    # minZ    = np.min(v[:, 2])
    # pin_idx = np.arange(v.shape[0])[v[:, 2] < minZ + 0.1]
    print("Pinned vertices: {}".format(pin_idx))

    # Add forces at the top 
    f_ext     = None 
    maxZ      = np.max(v[:, 2])
    force_idx = np.arange(v.shape[0])[v[:, 2] > maxZ - 0.1]
    # f_ext     = np.zeros_like(v)
    # f_ext[force_idx, 0] = 1e3 # [N]
    print("Vertices on which a force is applied: {}".format(force_idx))

    ee = LinearElasticEnergy(young, poisson)
    # ee = CorotatedElasticEnergy(young, poisson)
    # ee = NeoHookeanElasticEnergy(young, poisson)
    S  = ElasticSolid(v, t, ee, rho=rho, damping=damping, 
                      pin_idx=pin_idx, f_ext=f_ext, self_weight=False)

    # initial deformation
    # This might invert some elements and make Neo Hookean energy 
    # blow up under pinning constraints
    tb = igl.boundary_facets(t)
    v_def = v.copy()
    v_def[:, 2] *= 1.5
    # v_def[tb[0], 2] *= 1.5
    S.update_shape(v_def)

    # K = S.assemble_stiffness()
    # eigval, eigvec = np.linalg.eigh(K)
    # print(K[:5, :5])
    # print(eigvec[:5, 0])
    # print(np.allclose(K, K.T))
    # print(eigval[:15])
    # print()

    if True:

        # iterations of simulation
        iterations = 100

        # save the images for video
        save_video = False

        @mlab.animate(delay=10)
        def simulate():
            # extract the boundary 2D mesh
            tb = igl.boundary_facets(t)
            # plot the 2D mesh
            M = mlab.triangular_mesh(S.v[:, 0], S.v[:, 1], S.v[:, 2], tb)
            for i in range(iterations):
                # S.explicit_integration_step(dt=dt)
                # S.implicit_integration_step(dt=dt, steps=3)
                S.equilibrium_step(tb[0])
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
