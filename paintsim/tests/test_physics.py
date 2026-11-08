"""Equation-level checks of the You et al. 2013 rebuild against analytic results."""
import math

import numpy as np
import pytest

from paintsim import kernels as K
from paintsim.grid import HashGrid
from paintsim.params import BINDER, SOLVENT, Config, Liquid, FLOWER_SOLVENT, POLLOCK_BINDER, capillary_pressure
from paintsim.scene import lattice_box, paper_sheet, wall_box
from paintsim.solver import Simulation


def _static(sim, x):
    """Neighbour lists + density for a frozen particle set (no boundary)."""
    cfg = sim.cfg
    g = HashGrid(x, cfg.h)
    nbr, cnt, _ = g.neighbors(x, cfg.max_neighbors, same_set=True)
    bx = np.array([[1e6, 1e6, 1e6]])
    bn = np.zeros((x.shape[0], 1), np.int32)
    bc = np.zeros(x.shape[0], np.int32)
    rho = np.empty(x.shape[0])
    K.density(x, nbr, cnt, bx, bn, bc, np.ones(1), sim.m, cfg.h, sim.cp6, rho)
    return nbr, cnt, rho


def _interior(x, lo, hi, margin):
    return np.all((x > np.asarray(lo) + margin) & (x < np.asarray(hi) - margin), axis=1)


def test_rest_density_of_lattice():
    cfg = Config()
    sim = Simulation(cfg, [FLOWER_SOLVENT], [[0, 0, 1]])
    lo, hi = (0, 0, 0), (0.06, 0.06, 0.06)
    x = lattice_box(lo, hi, cfg.d0)
    _, _, rho = _static(sim, x)
    inner = _interior(x, lo, hi, cfg.h)
    assert abs(rho[inner].mean() - cfg.rho0) < 1e-6 * cfg.rho0


def test_eq7_constant_diffusivity_matches_erf():
    """Eq. (7) with constant D (curvature='constant'): a step diffuses like erfc with D_eff = D / rho0
    (Eq. 7 carries 1/(rho_i rho_j), i.e. it is (D/rho_i) * lap c)."""
    cfg = Config(curvature="constant")
    liq = Liquid("t", SOLVENT, mu_v=0.0, omega_D=0.05)
    sim = Simulation(cfg, [liq], [[0, 0, 1]])
    lo, hi = (-0.06, 0, 0), (0.06, 0.05, 0.05)
    x = lattice_box(lo, hi, cfg.d0)
    nbr, cnt, rho = _static(sim, x)
    c = (x[:, 0] < 0).astype(float)[:, None]
    lid = np.zeros(len(x), np.int32)
    D = np.zeros_like(c)
    lam = np.zeros(len(x))
    T, dt = 1.0, 1e-3
    out = np.empty_like(c)
    for _ in range(int(T / dt)):
        K.diffusivity(x, rho, lid, c, nbr, cnt, sim.m, cfg.h, sim.csp, sim.clap, sim.omega, 2, cfg.grad_eps,
                      1 / cfg.d0, False, D, lam)
        K.diffuse(x, rho, c, D, nbr, cnt, sim.m, cfg.h, sim.clap, 0, dt, out)
        c, out = out, c
    Deff = 0.05 / cfg.rho0
    inner = _interior(x, lo, hi, cfg.h) & (np.abs(x[:, 0]) < 0.04)
    exact = 0.5 * np.array([math.erfc(v / (2 * math.sqrt(Deff * T))) for v in x[inner, 0]])
    assert np.abs(c[inner, 0] - exact).max() < 0.03


def test_eq8_literal_sign_cannot_spread():
    """Literal Eq. (8): D lap c = -omega (lap c)^2/|grad c| <= 0 -> clean solvent never receives pigment."""
    cfg = Config(curvature="literal")
    liq = Liquid("t", SOLVENT, mu_v=0.0, omega_D=0.05)
    sim = Simulation(cfg, [liq], [[0, 0, 1]])
    lo, hi = (-0.04, 0, 0), (0.04, 0.04, 0.04)
    x = lattice_box(lo, hi, cfg.d0)
    nbr, cnt, rho = _static(sim, x)
    c0 = (x[:, 0] < 0).astype(float)[:, None]
    c = c0.copy()
    lid = np.zeros(len(x), np.int32)
    D = np.zeros_like(c); lam = np.zeros(len(x)); out = np.empty_like(c)
    for _ in range(200):
        K.diffusivity(x, rho, lid, c, nbr, cnt, sim.m, cfg.h, sim.csp, sim.clap, sim.omega, 1, cfg.grad_eps,
                      1 / cfg.d0, False, D, lam)
        K.diffuse(x, rho, c, D, nbr, cnt, sim.m, cfg.h, sim.clap, 0, 1e-3, out)
        c, out = out, c
    assert c[x[:, 0] > 0].max() == 0.0          # nothing crossed into the clean half
    assert c.sum() < c0.sum()                    # pigment only disappears


def test_eq8_abs_spreads_and_pair_mode_conserves():
    cfg = Config(curvature="abs", diffusion_pairing="pair")
    liq = Liquid("t", SOLVENT, mu_v=0.0, omega_D=0.05)
    sim = Simulation(cfg, [liq], [[0, 0, 1]])
    lo, hi = (-0.04, 0, 0), (0.04, 0.04, 0.04)
    x = lattice_box(lo, hi, cfg.d0)
    nbr, cnt, rho = _static(sim, x)
    c0 = (x[:, 0] < 0).astype(float)[:, None]
    c = c0.copy()
    lid = np.zeros(len(x), np.int32)
    D = np.zeros_like(c); lam = np.zeros(len(x)); out = np.empty_like(c)
    vol = sim.m / rho
    for _ in range(100):
        K.diffusivity(x, rho, lid, c, nbr, cnt, sim.m, cfg.h, sim.csp, sim.clap, sim.omega, 0, cfg.grad_eps,
                      1 / cfg.d0, False, D, lam)
        dt = min(1e-3, 0.45 / (np.abs(D).max() * lam.max()))
        K.diffuse(x, rho, c, D, nbr, cnt, sim.m, cfg.h, sim.clap, 1, dt, out)
        c, out = out, c
    assert c[x[:, 0] > 0].max() > 0.05
    # pair mode: the weight m lap W / (rho_i rho_j) is symmetric, so sum_i c_i (uniform mass) is conserved exactly
    assert abs(c.sum() - c0.sum()) < 1e-9 * c0.sum()


def test_eq9_lucas_washburn_exact_integration():
    """Eq. (9)-(10): l^2 = 2 K t with K = (P_h + P_c)(r_c^2 + 4 r_c)/(8 mu); absorption at l = absorb_depth."""
    cfg = Config(hydrostatic=False)
    sol = FLOWER_SOLVENT
    sim = Simulation(cfg, [sol], [[0, 0, 1]])
    top, _ = paper_sheet(-0.02, 0.02, -0.02, 0.02, cfg.d0, layers=1)
    sim.add_boundary(top, True)
    sim.finalize_boundary()
    x = np.array([[0.0025, 0.0025, 0.5 * cfg.d0]])
    bn, bc, _ = sim.bgrid.neighbors(x, 64)
    l = np.zeros(1)
    ab = np.zeros(1, np.bool_)
    pc = capillary_pressure(sol.gamma, sol.theta, sol.r_c)
    Kc = pc * (sol.r_c ** 2 + 4 * sol.r_c) / (8 * sol.mu_v)
    t_abs = cfg.absorb_depth ** 2 / (2 * Kc)
    dt = t_abs / 100.5
    steps = 0
    while not ab[0]:
        K.absorption(x, np.zeros(1, np.int32), l, np.zeros(1), bn, bc, sim.bpaper, sim.rc, sim.gamma, sim.theta,
                     sim.muv, cfg.rho0, 9.81, dt, cfg.contact_height, cfg.absorb_depth, False, ab)
        steps += 1
        if steps < 50:
            assert abs(l[0] - math.sqrt(2 * Kc * steps * dt)) < 1e-12
    assert steps == 101


def test_no_absorption_when_rc_zero():
    cfg = Config()
    sim = Simulation(cfg, [POLLOCK_BINDER], [[1, 0, 0]])
    top, _ = paper_sheet(-0.02, 0.02, -0.02, 0.02, cfg.d0, layers=1)
    sim.add_boundary(top, True)
    sim.finalize_boundary()
    x = np.array([[0.0025, 0.0025, 0.5 * cfg.d0]])
    bn, bc, _ = sim.bgrid.neighbors(x, 64)
    ab = np.zeros(1, np.bool_)
    K.absorption(x, np.zeros(1, np.int32), np.zeros(1), np.zeros(1), bn, bc, sim.bpaper, sim.rc, sim.gamma,
                 sim.theta, sim.muv, cfg.rho0, 9.81, 1.0, cfg.contact_height, cfg.absorb_depth, True, ab)
    assert not ab[0]


def test_appendix_a_strain_rate_and_yield():
    """Simple shear u = (s y, 0, 0): elastic eps_xy grows at s/2 until the von Mises yield; then (A.7) settles where
    the plastic flow balances the strain rate: |eps'|_F = yield + |D|_F / alpha."""
    s, alpha, yld = 2.0, 50.0, 0.05
    liq = Liquid("t", BINDER, mu_v=0.0, mu_e=1.0, alpha=alpha, yield_=yld)
    cfg = Config()
    sim = Simulation(cfg, [liq], [[1, 0, 0]])
    lo, hi = (0, 0, 0), (0.06, 0.06, 0.06)
    x = lattice_box(lo, hi, cfg.d0)
    v = np.zeros_like(x)
    v[:, 0] = s * x[:, 1]
    nbr, cnt, rho = _static(sim, x)
    eps = np.zeros((len(x), 6))
    lid = np.zeros(len(x), np.int32)
    inner = _interior(x, lo, hi, cfg.h)
    dt = 1e-4
    for _ in range(50):  # elastic phase: |eps'| = sqrt(2) * eps_xy < yld while eps_xy < 0.0354
        K.update_strain(x, v, rho, lid, eps, nbr, cnt, sim.m, cfg.h, sim.csp, sim.elastic, sim.alpha, sim.yld, dt)
    # the spiky-kernel SPH gradient (A.4) on a cubic lattice is ~5 % low; that is discretisation, not the model
    rate = eps[inner, 3].mean() / (50 * dt)
    assert abs(rate - 0.5 * s) < 0.08 * 0.5 * s
    for _ in range(20000):
        K.update_strain(x, v, rho, lid, eps, nbr, cnt, sim.m, cfg.h, sim.csp, sim.elastic, sim.alpha, sim.yld, dt)
    e = eps[inner].mean(0)
    nf = math.sqrt(2) * abs(e[3])
    Dn = math.sqrt(2) * rate
    assert abs(nf - (yld + Dn / alpha)) < 0.02 * (yld + Dn / alpha)


def test_eq5_forms_on_linear_strain_field():
    """grad . eps for eps_xx = a x: 'difference' and 'symmetric' recover mu_e a / rho0 in the interior;
    'literal' (eps_i only) gives ~0 there - the printed form only acts at free surfaces."""
    a = 3.0
    liq = Liquid("t", BINDER, mu_v=0.0, mu_e=1.0)
    lo, hi = (0, 0, 0), (0.06, 0.06, 0.06)
    res = {}
    for form in ("literal", "difference", "symmetric"):
        cfg = Config(elastic_form=form)
        sim = Simulation(cfg, [liq], [[1, 0, 0]])
        x = lattice_box(lo, hi, cfg.d0)
        nbr, cnt, rho = _static(sim, x)
        eps = np.zeros((len(x), 6))
        eps[:, 0] = a * x[:, 0]
        acc = np.empty_like(x)
        bx = np.array([[1e6, 1e6, 1e6]])
        K.nonpressure_acc(x, np.zeros_like(x), rho, np.zeros(len(x), np.int32), eps, nbr, cnt, bx,
                          np.zeros((len(x), 1), np.int32), np.zeros(len(x), np.int32), np.ones(1), sim.m, cfg.h,
                          sim.csp, sim.clap, sim.muv, sim.mue, sim.elastic, 0.0, 0.0, 0.0, {"literal": 0,
                          "difference": 1, "symmetric": 2}[form], False, cfg.rho0, acc)
        inner = _interior(x, lo, hi, cfg.h)
        res[form] = acc[inner, 0].mean()
    target = a / 1000.0
    assert abs(res["difference"] - target) < 0.1 * target
    assert abs(res["literal"]) < 0.05 * target


def test_hydrostatic_rest_pcisph():
    cfg = Config(d0=0.005, h=0.01, dt=2e-3, adaptive=True)
    sim = Simulation(cfg, [Liquid("t", BINDER, mu_v=10.0)], [[1, 0, 0]])
    top, rest = paper_sheet(-0.04, 0.04, -0.04, 0.04, cfg.d0)
    sim.add_boundary(top, True); sim.add_boundary(rest, False)
    sim.add_boundary(wall_box(-0.025, 0.025, -0.025, 0.025, 0.05, cfg.d0), False)
    sim.add_particles(lattice_box((-0.025, -0.025, 0), (0.025, 0.025, 0.03), cfg.d0), 0, [1.0])
    while sim.t < 0.4:
        sim.step()
    assert sim.stats["last_err"] < cfg.eta
    assert np.abs(sim.v).max() < 0.05
    assert sim.x[:, 2].min() > 0
