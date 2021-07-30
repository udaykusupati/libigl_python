from elasticsolid2 import *
from elasticenergy import *
from mayavi import mlab
import igl
from scipy.linalg import polar

def test_jacobian_translation(solid):
    dv_def = np.zeros_like(solid.v)
    dv_def[:, 0] += 1

    # Updates the Jacobian
    solid.update_shape(solid.v + dv_def)

    # Should be the identity
    target_F = np.tile(np.eye(3).reshape(1, 3, 3), (solid.t.shape[0], 1, 1))

    assert np.allclose(target_F, solid.F)
    pass

def test_jacobian(solid):
    F = np.array([
        [1, 0, 1],
        [0, 2, 0],
        [0, 0, 3]
    ])

    v_def = solid.v @ F.T
    # Updates the Jacobian
    solid.update_shape(v_def)

    target_F = np.tile(F.reshape(1, 3, 3), (solid.t.shape[0], 1, 1))
    assert np.allclose(target_F, solid.F)
    pass

def test_mass_matrix(solid):
    assert np.allclose(np.ones(shape=(solid.t.shape[0],)) / 6, solid.W0)
    pass

def test_strain_tensor(solid):
    F = np.array([
        [1, 0, 1],
        [0, 2, 0],
        [0, 0, 3]
    ])

    v_def = solid.v @ F.T
    # Updates the Jacobian, then compute the strain tensor
    solid.update_shape(v_def)
    solid.ee.make_strain_tensor(solid.F)

    if type(solid.ee) == LinearElasticEnergy:
        E = np.array([
            [0  , 0, 1/2],
            [0  , 1, 0  ],
            [1/2, 0, 2  ]
        ])
    elif type(solid.ee) == CorotatedElasticEnergy:
        _, S = polar(F)
        E    = S - np.eye(3)
    else:
        E = np.array([
            [0  , 0  , 1/2],
            [0  , 3/2, 0  ],
            [1/2, 0  , 9/2]
        ])
    
    target_E = np.tile(E.reshape(1, 3, 3), (solid.t.shape[0], 1, 1))
    assert np.allclose(target_E, solid.ee.E)

    pass

def test_stress_tensor(solid):
    F = np.array([
        [1, 0, 1],
        [0, 2, 0],
        [0, 0, 3]
    ])

    v_def = solid.v @ F.T
    # Updates the Jacobian
    solid.update_shape(v_def)

    if type(solid.ee) == LinearElasticEnergy:
        P = np.array([
            [3, 0, 1],
            [0, 5, 0],
            [1, 0, 7]
        ])
    elif type(solid.ee) == KirchhoffElasticEnergy:
        P = np.array([
            [7, 0 , 16],
            [0, 18, 0 ],
            [3, 0 , 45]
        ])
    elif type(solid.ee) == CorotatedElasticEnergy:
        R, S = polar(F)
        E    = S - np.eye(3)
        P    = R @ (2*E + np.trace(E)*np.eye(3))
    elif type(solid.ee) == NeoHookeanElasticEnergy:
        logJ = np.log(6)
        P = np.array([
            [logJ      , 0         , 1         ],
            [0         , (3+logJ)/2, 0         ],
            [(1-logJ)/3, 0         , (8+logJ)/3]
        ])
    else:
        print("Unknown energy model.")
        raise NotImplementedError
    
    target_P = np.tile(P.reshape(1, 3, 3), (solid.t.shape[0], 1, 1))
    assert np.allclose(target_P, solid.ee.P)

    pass

def test_von_mises():
    pass

if __name__ == '__main__':

    # Some characteristics
    rho     = 1  # [kg.m-3]
    damping = 0.

    # Choose E and nu, so that mu = lambda = 1
    young   = 2.5 
    poisson = 0.25

    t = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4]
    ])
    v = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [2/3, 2/3, 2/3]
    ])

    tb = igl.boundary_facets(t)
    M  = mlab.triangular_mesh(v[:, 0], v[:, 1], v[:, 2], tb)
    # mlab.show()

    # ee = LinearElasticEnergy(young, poisson)
    # ee = KirchhoffElasticEnergy(young, poisson)
    ee = CorotatedElasticEnergy(young, poisson)
    # ee = NeoHookeanElasticEnergy(young, poisson)

    # Test independence to translation
    solid = ElasticSolid(v, t, ee, rho=rho, damping=damping, 
                         pin_idx=[], f_ext=None, self_weight=True)
    test_jacobian_translation(solid)

    # Test Jacobian computation
    solid = ElasticSolid(v, t, ee, rho=rho, damping=damping, 
                         pin_idx=[], f_ext=None, self_weight=True)
    test_jacobian(solid)

    # Test mass matrix computation
    solid = ElasticSolid(v, t, ee, rho=rho, damping=damping, 
                         pin_idx=[], f_ext=None, self_weight=True)
    test_mass_matrix(solid)

    # Test strain tensor computation
    solid = ElasticSolid(v, t, ee, rho=rho, damping=damping, 
                         pin_idx=[], f_ext=None, self_weight=True)
    test_strain_tensor(solid)

    # Test stress tensor computation
    solid = ElasticSolid(v, t, ee, rho=rho, damping=damping, 
                         pin_idx=[], f_ext=None, self_weight=True)
    test_stress_tensor(solid)

    print("All tests are passed!")

    pass