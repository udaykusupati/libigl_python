import numpy as np

from scipy import sparse

import igl


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                             ELASTIC SOLID CLASS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class ElasticSolid(object):

    def __init__(self, v, t, rho=1, young=1e6, poisson=0.2, damping=1e-2,
                 model='kirchhoff'):
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
        self.v = v
        self.t = t
        self.rho = rho
        self.young = young
        self.poisson = poisson
        self.gamma = damping
        self.lbda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        self.mu = young / (2 * (1 + poisson))
        self.W = None
        self.Ds = None
        self.dv = None
        self.F = None
        self.E = None
        self.P = None
        self.f = None
        self.W0 = None
        self.Bm = None
        self.M = None
        self.model = model
        self.update_shape(v)
        self.velocity = np.zeros((len(self.v), 3))


    def vertex_tet_sum(self, data):
        i = self.t.flatten('F')
        j = np.arange(len(self.t))
        j = np.hstack((j, j, j, j))
        if len(data) == len(self.t):
            data = data[j]
        m = sparse.coo_matrix((data, (i, j)), (len(self.v), len(self.t)))
        return np.array(m.sum(axis=1)).flatten()

    def displace(self, v_disp):
        """
        Input:
        - v_disp   : displacement of the vertices of the mesh (#v, 3)
        """
        self.update_shape(self.v + v_disp)

    def update_shape(self, v_def):
        """
        Input:
        - v_def   : position of the vertices of the mesh (#v, 3)
        """
        self.v = v_def
        self.make_shape_matrix()
        self.W = abs(np.linalg.det(self.Ds)) / 6
        if self.W0 is None:
            self.W0 = self.W.copy()
            self.make_mass_matrix()
        self.make_jacobian()
        self.make_piola_kirchhoff_stress_tensor()
        self.make_elastic_forces()

    def make_mass_matrix(self):
        M = self.vertex_tet_sum(self.W)
        self.M = 1 / 4 * M * self.rho

    def make_shape_matrix(self):
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
        self.F = np.einsum('lij,ljk->lik', self.Ds, self.Bm)

    def make_strain_tensor(self):
        eye = np.zeros((len(self.F), 3, 3))
        for i in range(3):
            eye[:, i, i] = 1
        if self.model == 'linear':
            self.E = 0.5*(np.swapaxes(self.F, 1, 2) + self.F) - eye
        else:
            self.E = np.einsum('lij,ljk->lik', np.swapaxes(self.F, 1, 2), self.F) - eye

    def make_piola_kirchhoff_stress_tensor(self):
        if self.model == 'neo-hookean':
            I3 = np.log(np.linalg.det(self.F)**2)
            FinvT = np.swapaxes(np.linalg.inv(self.F), 1, 2)
            self.P = (self.mu * (self.F - FinvT) +
                      0.5 * self.lbda * np.einsum('i,ijk->ijk', I3, FinvT))
        else:
            self.make_strain_tensor()
            tr = np.einsum('ijj->i', self.E)
            eye = np.zeros((len(self.t), 3, 3))
            for i in range(3):
                eye[:, i, i] = tr
            if self.model == 'kirchhoff':
                self.P = np.einsum('lij,ljk->lik', self.F, 2 * self.mu * self.E + self.lbda * eye)
            elif self.model == 'linear':
                self.P = 2 * self.mu * self.E + self.lbda * eye

    def make_elastic_forces(self):
        H = np.einsum('lij,ljk->lik', self.P, np.swapaxes(self.Bm, 1, 2))
        H = - np.einsum('i,ijk->ijk', self.W0, H)
        fx = self.vertex_tet_sum(np.hstack((H[:, 0, 0], H[:, 0, 1], H[:, 0, 2],
                                            -H[:, 0, 0] - H[:, 0, 1] - H[:, 0, 2])))
        fy = self.vertex_tet_sum(np.hstack((H[:, 1, 0], H[:, 1, 1], H[:, 1, 2],
                                            -H[:, 1, 0] - H[:, 1, 1] - H[:, 1, 2])))
        fz = self.vertex_tet_sum(np.hstack((H[:, 2, 0], H[:, 2, 1], H[:, 2, 2],
                                            -H[:, 2, 0] - H[:, 2, 1] - H[:, 2, 2])))
        self.f = np.column_stack((fx, fy, fz))

    def von_mises(self):
        VM = (.5 * (self.P[:, 0, 0] - self.P[:, 1, 1]) ** 2 +
              .5 * (self.P[:, 0, 0] - self.P[:, 2, 2]) ** 2 +
              .5 * (self.P[:, 1, 1] - self.P[:, 2, 2]) ** 2 +
              3 * (self.P[:, 0, 1]**2 + self.P[:, 0, 2]**2 +
                   self.P[:, 1, 2]**2)) ** .5
        return np.abs(self.vertex_tet_sum(VM))

    def stress_norm(self):
        VM = np.linalg.norm(self.P, axis=(1, 2))
        return np.abs(self.vertex_tet_sum(VM))

    def compute_force_differentials(self, v_disp):
        """
        Input:
        - v_disp   : displacement of the vertices of the mesh (#v, 3)

        Output:
        - df    : force differentials at the vertices of the mesh (#v, 3)
        """
        d1 = (v_disp[self.t[:, 0]] - v_disp[self.t[:, 3]])
        d2 = (v_disp[self.t[:, 1]] - v_disp[self.t[:, 3]])
        d3 = (v_disp[self.t[:, 2]] - v_disp[self.t[:, 3]])
        dDs = np.stack((d1, d2, d3))
        dDs = np.swapaxes(dDs, 0, 1)
        dDs = np.swapaxes(dDs, 1, 2)
        dF = np.einsum('lij,ljk->lik', dDs, self.Bm)
        dE = 0.5 * (np.einsum('lij,ljk->lik', np.swapaxes(dF, 1, 2), self.F) +
                    np.einsum('lij,ljk->lik', np.swapaxes(self.F, 1, 2), dF))
        tr = np.einsum('ijj->i', self.E)
        I = np.zeros((len(self.F), 3, 3))
        dtr = np.einsum('ijj->i', dE)
        dI = np.zeros((len(self.F), 3, 3))
        for i in range(3):
            I[:, i, i] = tr
            dI[:, i, i] = dtr
        dP = (np.einsum('lij,ljk->lik', dF, 2 * self.mu * self.E + self.lbda * I) +
              np.einsum('lij,ljk->lik', self.F, 2 * self.mu * dE + self.lbda * dI))
        dH = np.einsum('lij,ljk->lik', dP, np.swapaxes(self.Bm, 1, 2))
        dH = np.einsum('i,ijk->ijk', self.W0, dH)
        dfx = self.vertex_tet_sum(np.hstack((dH[:, 0, 0], dH[:, 0, 1], dH[:, 0, 2],
                                             -dH[:, 0, 0] - dH[:, 0, 1] - dH[:, 0, 2])))
        dfy = self.vertex_tet_sum(np.hstack((dH[:, 1, 0], dH[:, 1, 1], dH[:, 1, 2],
                                             -dH[:, 1, 0] - dH[:, 1, 1] - dH[:, 1, 2])))
        dfz = self.vertex_tet_sum(np.hstack((dH[:, 2, 0], dH[:, 2, 1], dH[:, 2, 2],
                                             -dH[:, 2, 0] - dH[:, 2, 1] - dH[:, 2, 2])))
        return np.column_stack((dfx, dfy, dfz))

    def explicit_integration_step(self, dt=1e-6):
        D = np.diag((self.M + self.gamma * dt) ** -1)
        self.velocity += dt * np.einsum('ij,jk->ik', D, self.f)
        v_disp = self.velocity * dt
        self.displace(v_disp)

    def implicit_integration_step(self, dt=1e-6, step=2):
        new_velocity = self.velocity.copy()
        for i in range(step):
            fd = -self.gamma * self.compute_force_differentials(new_velocity)
            ft = fd + self.f
            M = np.diag(self.M)

            def LHS(dv):
                return (1 / dt ** 2 * np.einsum('ij,jk->ik', M, dv) +
                        (1 + self.gamma / dt) * self.compute_force_differentials(dv))

            RHS = 1 / dt * np.einsum('ij,jk->ik', M, self.velocity - new_velocity) + ft
            dv = self.conjugate_gradient(LHS, RHS)
            self.displace(dv)
            new_velocity += dv / dt
        self.velocity = new_velocity

    def conjugate_gradient(self, Ax_method, b, x0=None, eps=1e-3, max_steps=None):
        """
        Solves Ax = b where A is positive definite.
        A has shape (dim, dim), x and b have shape (dim, n).

        Input:
        - Ax_method : a method that takes x and returns the product Ax
        - b         : right hand side of the linear system
        - x0        : initial guess for x0 (defaults to 0s)
        - eps       : tolerance on the residual magnitude
        - max_steps : maximum number of iterations (defaults to dim)

        Output:
        - xStar : np array of shape (dim, n) solving the linear system
        """
        dim, n = b.shape
        if max_steps is None:
            max_steps = dim
        if x0 is None:
            xStar = np.zeros_like(b)
        else:
            xStar = x0
        residuals = b - Ax_method(xStar)
        directions = residuals
        # Reusable quantities
        A_directions = Ax_method(directions)  # A p
        resNormsSq = np.sum(residuals ** 2, axis=0)  # r^T r
        dirNormsSqA = np.zeros(shape=(n,))  # stores p^T A p
        for i in range(n):
            dirNormsSqA[i] = directions[:, i] @ A_directions[:, i]
        for k in range(max_steps):
            alphas = resNormsSq / dirNormsSqA
            xStar = xStar + directions * alphas
            residuals = residuals - A_directions * alphas
            newResNormsSq = np.sum(residuals ** 2, axis=0)
            if max(np.sqrt(newResNormsSq)) < eps:
                break
            betas = newResNormsSq / resNormsSq
            directions = residuals + directions * betas
            A_directions = Ax_method(directions)
            resNormsSq = newResNormsSq
            for i in range(n):
                dirNormsSqA[i] = directions[:, i] @ A_directions[:, i]
        return xStar

# -----------------------------------------------------------------------------
#                                    Test
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    from mayavi import mlab

    v, t = igl.read_msh("meshes/ball.msh")

    S = ElasticSolid(v, t, 10, 1e6, 0.2, 0)

    # initial deformation

    v_def = v.copy()

    v_def[:, 2] *= 1.2

    S.update_shape(v_def)

    if True:

        # iterations of simulation
        iterations = 10

        # save the images for video
        save_video = True

        @mlab.animate(delay=10)
        def simulate():
            # extract the boundary 2D mesh
            tb = igl.boundary_facets(t)
            # plot the 2D mesh
            M = mlab.triangular_mesh(S.v[:, 0], S.v[:, 1], S.v[:, 2], tb)
            for i in range(iterations):
                S.explicit_integration_step(1e-5)
                M.mlab_source.reset(x=S.v[:, 0], y=S.v[:, 1], z=S.v[:, 2])
                yield
                if save_video:
                    # to be improved for background color
                    prefix = len(str(int(iterations))) - len(str(i))
                    n_seq = '0' * prefix + str(i)
                    mlab.savefig(filename='img_{}.png'.format(n_seq),
                                 size=None)

        simulate()

        mlab.show()
