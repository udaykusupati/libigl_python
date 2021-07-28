import numpy as np


def conjugate_gradient(Ax_method, b, x0=None, eps=1e-3, max_steps=None):
        '''
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
        '''

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