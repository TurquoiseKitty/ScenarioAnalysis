import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence



@dataclass
class NOI_Args:
    dim: int = field(default= 0)
    num: int = field(default= 1)
    sigma: float = field(default = 0.)
    eig_vals: np.ndarray = field(default = None)


@dataclass
class FE_Args:
    weights: np.ndarray = field(default = None)


def AR_effect_F(
        factors,
        weights,
        **args
):
    num_dep, dim_fac = factors.shape

    assert num_dep == len(weights)

    return np.matmul(factors.T, weights)


def noise_diagLambda_F(
        dim,
        num,
        eig_vals,
        **args
):
    
    assert dim == len(eig_vals)
    
    mean = np.zeros(dim)

    cov = np.diag(eig_vals)

    x = np.random.multivariate_normal(mean,cov,num)

    if num == 1:

        x = x[0]

    return x

def noise_iid_F(
        dim,
        num,
        sigma,
        **args
):
    return noise_diagLambda_F(dim, num, sigma * np.ones(dim))



def risk_factor_generator(
        dep_factors,
        fix_effect_F,
        noise_gen_F,
        fix_args,
        noise_args
):
    
    num_dep, dim_fac = dep_factors.shape

    if noise_args.dim == 0:
        noise_args.dim = dim_fac

    fix_effect = fix_effect_F(dep_factors, **asdict(fix_args))

    noise = noise_gen_F(**asdict(noise_args))

    return fix_effect + noise


def q_generator(
        factor,
        Gamma,          # Gamma m\times K
        noise_gen_F,
        noise_args
):
    m, K = Gamma.shape

    assert len(factor) == K

    fix_effect = np.matmul(Gamma, factor)

    if noise_args.dim == 0:
        noise_args.dim = m

    noise = noise_gen_F(**asdict(noise_args))

    return fix_effect + noise

def Bond_PortVal(
        rates,
        S_mature,
        positions
):
    return np.dot(positions, np.exp(-S_mature * rates))

