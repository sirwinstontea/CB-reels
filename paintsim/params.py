"""Material and solver parameters for the You et al. 2013 rebuild.

Every liquid field maps to a symbol in the paper; see PAPER_MAP.md for the full equation -> code table.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

BINDER = 0   # paint = binder carrying pigment (Sec. 3, 4)
SOLVENT = 1  # diluent; carries pigment only after diffusion (Sec. 4.2)


@dataclass(frozen=True)
class Liquid:
    """One liquid type: a row of Table 1 plus the Appendix A constants."""
    name: str
    kind: int                 # BINDER or SOLVENT
    mu_v: float               # viscosity            (Eq. 2, 4, and mu in Eq. 9)
    mu_e: float = 0.0         # elasticity           (Eq. 2, 5)  - applied to binder particles only
    omega_D: float = 0.0      # diffusivity weight   (Eq. 8)
    r_c: float = 0.0          # capillary radius     (Eq. 9, 10) - 0 means no absorption
    gamma: float = 0.07275    # surface tension      (Eq. 10)
    theta: float = 0.0        # contact angle, rad   (Eq. 10)    - not given in the paper
    alpha: float = 0.0        # elastic decay rate   (Eq. A.6, A.7) - not given in the paper
    yield_: float = 0.0       # yield point          (Eq. A.6, A.7) - not given in the paper
    rho0: float = 1000.0      # rest density (paper does not distinguish liquids)


# ---- Table 1 (exact values) -------------------------------------------------------------
# Jackson Pollock (Fig. 6): one binder row.  Text: "We set mu_v equal to 80.0 and mu_e equal to 1000.0",
# "For the Jackson Pollock painting, we set r_c equal to zero, which means no absorption."
# alpha / yield are not published; values here are ours (see PAPER_MAP.md, "Gaps").
POLLOCK_BINDER = Liquid("pollock-binder", BINDER, mu_v=80.0, mu_e=1000.0, omega_D=0.001, r_c=0.0,
                        gamma=0.028, alpha=25.0, yield_=0.1)  # alpha, yield: Goktekin 2004 Fig. 2 values

# Flower (Fig. 7): binder + solvent rows of Table 1.
# Note: the text says "The diffusivity weight of the solvent is 0.038" and "we set r_c equal to 2.5e-5";
# Table 1 lists binder 0.038 / solvent 0.05 and r_c = 2.5e-4.  We follow Table 1 by default.
FLOWER_BINDER = Liquid("flower-binder", BINDER, mu_v=10.0, mu_e=0.0, omega_D=0.038, r_c=0.0, gamma=0.028)
FLOWER_SOLVENT = Liquid("flower-solvent", SOLVENT, mu_v=5.0, mu_e=0.0, omega_D=0.05, r_c=2.5e-4, gamma=0.07275)

# CASA logo (Fig. 8), from the text: "the paints have only viscous force, mu_v = 40", "The diffusivity weight
# value is set to 0.01 for the solvent and 0.002 for the binder", "Setting r_c equal to 2.5e-4 for the solvent".
# The solvent viscosity is not stated; we reuse the Table 1 solvent value (5.0).
CASA_BINDER = Liquid("casa-binder", BINDER, mu_v=40.0, mu_e=0.0, omega_D=0.002, r_c=0.0, gamma=0.028)
CASA_SOLVENT = Liquid("casa-solvent", SOLVENT, mu_v=5.0, mu_e=0.0, omega_D=0.01, r_c=2.5e-4, gamma=0.07275)

# Fig. 5: "Red paint is affected by viscoelastic force, whereas green paint is moved by viscous force only.
# Yellow paint shows the results of excluding both forces."
FIG5_VISCOELASTIC = POLLOCK_BINDER
FIG5_VISCOUS = replace(POLLOCK_BINDER, name="fig5-viscous", mu_e=0.0)
FIG5_BASIC = replace(POLLOCK_BINDER, name="fig5-basic", mu_v=0.0, mu_e=0.0)


@dataclass
class Config:
    # discretisation (not given in the paper; derived in SPEC.md)
    d0: float = 0.005                 # particle spacing [m]
    h: float = 0.01                   # smoothing radius [m] (= 2 d0)
    rho0: float = 1000.0
    g: tuple = (0.0, 0.0, -9.81)

    # time stepping. The paper used a fixed dt (0.0002 for Pollock).
    dt: float = 2e-4
    adaptive: bool = False            # adaptive dt (CFL, viscous and diffusion limits), capped at dt
    cfl: float = 0.4

    # PCISPH (Solenthaler & Pajarola 2009, ref [7])
    eta: float = 0.01                 # max density error (1 %)
    min_iter: int = 3
    max_iter: int = 40

    # neighbour storage
    max_neighbors: int = 96
    max_boundary_neighbors: int = 64

    # --- interpretation switches for places where the printed paper is ambiguous (PAPER_MAP.md) ---
    # Eq. 5 is printed with eps_i only.  'literal' = as printed; 'difference' = (eps_j - eps_i),
    # 'symmetric' = m (eps_i/rho_i^2 + eps_j/rho_j^2) (momentum-conserving SPH divergence).
    elastic_form: str = "literal"
    # Eq. 8 printed as D = omega_D (-lap c)/|grad c|.  Taken literally, D*lap c = -omega_D (lap c)^2/|grad c| <= 0,
    # so pigment could never spread into clean solvent.  'abs' uses |lap c| (curvature magnitude); 'literal' as
    # printed; 'constant' uses D = omega_D (the "ideal diffusion" the paper contrasts against).
    curvature: str = "abs"
    # Which discrete operators feed Eq. 8.  'paper' = the paper's own: lap c as in Eq. 7 (m/(rho_i rho_j) lap W) and
    # grad c in the form of Eq. A.4 (m/rho_j grad W), so D = omega_D |lap c| / (rho_i |grad c|).
    # 'geometric' = both standard (m/rho_j), so the ratio is a curvature in 1/m.  ~rho0 = 1000x faster diffusion.
    curvature_ops: str = "paper"
    # Eq. 7 puts D_i (particle i's diffusivity) outside the sum, which is not pigment-conserving when D varies.
    # 'particle' = as printed, 'pair' = (D_i + D_j)/2 inside the sum (conserving).
    diffusion_pairing: str = "particle"
    grad_eps: float = 1e-3            # regulariser for |grad c| in Eq. 8 [1/m]
    kappa_max: float | None = None    # resolution limit on |curvature| in Eq. 8 (default 1/d0)
    diffusion_safety: float = 0.45    # explicit diffusion substep bound dt_sub*|D|*lam (numerics, not the paper)
    max_diffusion_substeps: int = 64

    # --- absorption (Sec. 4.3) ---
    absorb_depth: float | None = None  # Lucas-Washburn length l at which a touching particle counts as absorbed (default d0)
    contact_height: float | None = None  # particle centre within this height of the paper surface = touching (default d0)
    deposit_fraction: float = 1.0       # share of a particle's pigment left on the paper ("for the most part")
    hydrostatic: bool = True            # P_h = rho0 g H from the liquid column above the particle

    # --- boundary ---
    boundary_viscosity: bool = True     # no-slip viscous drag from paper / wall particles (Eq. 4 applied to them)
    # Akinci boundaries cannot hold up a one-particle-thick film (its density never reaches rho0, so PCISPH gives
    # it no pressure); particle centres are therefore kept at least contact_floor * d0 above the paper (the radius).
    contact_floor: float = 0.5
    kill_z: float = -0.2                # particles that fall below this are removed (fell off the sheet)

    def __post_init__(self):
        if self.absorb_depth is None:
            self.absorb_depth = self.d0
        if self.contact_height is None:
            self.contact_height = self.d0


def capillary_pressure(gamma: float, theta: float, r_c: float) -> float:
    """Eq. 10: P_c = 2 gamma cos(theta) / r_c."""
    return 2.0 * gamma * math.cos(theta) / r_c if r_c > 0 else 0.0
