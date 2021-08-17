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

    passed = np.allclose(target_F, solid.F)
    return passed

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
    passed   = np.allclose(target_F, solid.F)
    return passed

def test_diff_jacobian(solid):
    F = np.array([
        [1, 0, 1],
        [0, 2, 0],
        [0, 0, 3]
    ])

    dF = np.array([
        [2, 0, 4],
        [0, 1, 0],
        [0, 0, 3]
    ])

    tiled_dF = np.tile(dF.reshape(1, 3, 3), (solid.t.shape[0], 1, 1))

    # Find the correct dv
    v_def  = solid.v @ F.T
    v_plus_dv_def = solid.v @ (F + dF).T
    dv_def = v_plus_dv_def - v_def

    # Updates the Jacobian
    solid.update_shape(v_def)
    solid.compute_force_differentials(dv_def)

    passed   = np.allclose(tiled_dF, solid.dF)
    return passed

def test_mass_matrix(solid):
    passed = np.allclose(np.ones(shape=(solid.t.shape[0],)) / 6, solid.W0)
    return passed

def test_strain_tensor(solid):
    F = np.array([
        [1, 0, 1],
        [0, 2, 0],
        [0, 0, 3]
    ])

    v_def = solid.v @ F.T
    # Updates the Jacobian, then compute the strain tensor
    solid.update_shape(v_def)
    solid.ee.make_strain_tensor(solid.F) # Needed for NeoHookeanElasticEnergy

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
    passed = np.allclose(target_E, solid.ee.E)
    return passed

def test_diff_strain_tensor(solid):

    F = np.array([
        [1, 0, 1],
        [0, 2, 0],
        [0, 0, 3]
    ])

    dF = np.array([
        [2, 0, 4],
        [0, 1, 0],
        [0, 0, 3]
    ])

    tiled_dF = np.tile(dF.reshape(1, 3, 3), (solid.t.shape[0], 1, 1))

    v_def = solid.v @ F.T
    # Updates the Jacobian, then compute the strain tensor
    solid.update_shape(v_def)
    solid.ee.make_differential_strain_tensor(solid.F, tiled_dF)

    if type(solid.ee) == LinearElasticEnergy:
        dE = np.array([
            [2, 0, 2],
            [0, 1, 0],
            [2, 0, 3]
        ])
    elif type(solid.ee) == CorotatedElasticEnergy:
        _, S = polar(F)
        Sinv = np.linalg.inv(S)
        dE = Sinv @ np.array([
            [2, 0, 3 ],
            [0, 2, 0 ],
            [3, 0, 13]
        ])
    else:
        dE = np.array([
            [2, 0, 3 ],
            [0, 2, 0 ],
            [3, 0, 13]
        ])
    
    target_dE = np.tile(dE.reshape(1, 3, 3), (solid.t.shape[0], 1, 1))
    passed = np.allclose(target_dE, solid.ee.dE)
    return passed

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
    passed = np.allclose(target_P, solid.ee.P)
    return passed

def test_diff_stress_tensor(solid):

    F = np.array([
        [1, 0, 1],
        [0, 2, 0],
        [0, 0, 3]
    ])

    dF = np.array([
        [2, 0, 4],
        [0, 1, 0],
        [0, 0, 3]
    ])

    tiled_dF = np.tile(dF.reshape(1, 3, 3), (solid.t.shape[0], 1, 1))

    v_def = solid.v @ F.T
    # Updates the Jacobian, then compute the strain tensor
    solid.update_shape(v_def)
    solid.ee.make_differential_piola_kirchhoff_stress_tensor(solid.F, tiled_dF)

    if type(solid.ee) == LinearElasticEnergy:
        dP = np.array([
            [10, 0, 4 ],
            [0 , 8, 0 ],
            [4 , 0, 12]
        ])
    elif type(solid.ee) == KirchhoffElasticEnergy:
        dP = np.array([
            [43, 0 , 111],
            [0 , 51, 0  ],
            [21, 0 , 174]
        ])
    elif type(solid.ee) == CorotatedElasticEnergy:
        R, S = polar(F)
        E    = S - np.eye(3)

        # Compute dS then dR
        Sinv = np.linalg.inv(S)
        dS = Sinv @ np.array([
            [2, 0, 3 ],
            [0, 2, 0 ],
            [3, 0, 13]
        ])
        dR = (dF - R @ dS) @ Sinv

        dP = (dR @ (2 * E + np.trace(E) * np.eye(3)) + 
              R @ (2 * dS + np.trace(dS) * np.eye(3)))

    elif type(solid.ee) == NeoHookeanElasticEnergy:
        logJ = np.log(6)
        dP = np.array([
            [15/2-2*logJ  , 0       , 4            ],
            [0            , 3-logJ/4, 0            ],
            [-(5+2*logJ)/6, 0       , (27-2*logJ)/6]
        ])
    else:
        print("Unknown energy model.")
        raise NotImplementedError
    
    target_dP = np.tile(dP.reshape(1, 3, 3), (solid.t.shape[0], 1, 1))
    passed = np.allclose(target_dP, solid.ee.dP)
    return passed

def test_von_mises():
    pass

if __name__ == '__main__':

    # For some verbosity
    name_tests   = [
        "Translation Jacobian",
        "Jacobian computation",
        "Differential Jacobian computation",
        "Mass computation",
        "Strain computation",
        "Differential strain computation",
        "Stress computation",
        "Differential stress computation"
    ]
    passed_tests = {name:False for name in name_tests}

    # Some characteristics
    rho     = 1. # [kg.m-3]
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
    passed_tests["Translation Jacobian"] = test_jacobian_translation(solid)

    # Test Jacobian computation
    solid = ElasticSolid(v, t, ee, rho=rho, damping=damping, 
                         pin_idx=[], f_ext=None, self_weight=True)
    passed_tests["Jacobian computation"] = test_jacobian(solid)

    # Test differential Jacobian computation
    solid = ElasticSolid(v, t, ee, rho=rho, damping=damping, 
                         pin_idx=[], f_ext=None, self_weight=True)
    passed_tests["Differential Jacobian computation"] = test_diff_jacobian(solid)

    # Test mass matrix computation
    solid = ElasticSolid(v, t, ee, rho=rho, damping=damping, 
                         pin_idx=[], f_ext=None, self_weight=True)
    passed_tests["Mass computation"] = test_mass_matrix(solid)

    # Test strain tensor computation
    solid = ElasticSolid(v, t, ee, rho=rho, damping=damping, 
                         pin_idx=[], f_ext=None, self_weight=True)
    passed_tests["Strain computation"] = test_strain_tensor(solid)

    # Test differential strain tensor computation
    solid = ElasticSolid(v, t, ee, rho=rho, damping=damping, 
                         pin_idx=[], f_ext=None, self_weight=True)
    passed_tests["Differential strain computation"] = test_diff_strain_tensor(solid)

    # Test stress tensor computation
    solid = ElasticSolid(v, t, ee, rho=rho, damping=damping, 
                         pin_idx=[], f_ext=None, self_weight=True)
    passed_tests["Stress computation"] = test_stress_tensor(solid)

    # Test differential stress tensor computation
    solid = ElasticSolid(v, t, ee, rho=rho, damping=damping, 
                         pin_idx=[], f_ext=None, self_weight=True)
    passed_tests["Differential stress computation"] = test_diff_stress_tensor(solid)

    print("==========      Results      ==========\n")
    frac_passed = sum(passed_tests.values()) / max(1, len(passed_tests))
    print("Test passed: {:.0f}%\n".format(100*frac_passed))
    for (name, passed) in passed_tests.items():
        print(name + ": {}".format(passed))
    pass