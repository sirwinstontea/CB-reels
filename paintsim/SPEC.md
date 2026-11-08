# paintsim: rebuild of You et al. 2013

Rebuild of *Realistic paint simulation based on fluidity, diffusion, and absorption*
(M. You, T. Jang, S. Cha, J. Kim, J. Noh, Comp. Anim. Virtual Worlds 2013; 24:297-306, DOI 10.1002/cav.1500).
Local copy of the paper (not committed): `research/you2013/src/CAVW2013_You.pdf`, text in `raw.txt`.

## Units and world

- SI units: metres, kilograms, seconds. Gravity `g = (0, 0, -9.81)`.
- **z is up.** The paper is a sheet of boundary ("paper") particles whose top surface is at `z = 0`.
- Liquid particles have uniform mass `m` (the paper writes `m` without an index in Eq. 3-5).
- Default spacing `d0 = 0.005 m`, smoothing radius `h = 2 d0 = 0.01 m`, rest density `rho0 = 1000`.
  Why: with `mu_v = 80` the explicit viscous limit `dt < 0.125 h^2 rho0 / mu_v` gives `h ~ 0.011 m` for the
  paper's reported `dt = 0.0002`. That matches the reported particle counts too.
- Kernels (Mueller et al. 2003, ref [25]): poly6 for density, spiky gradient for pressure, elasticity and gradients,
  viscosity-kernel Laplacian for every `nabla^2 W` (Eq. 4, 7).

## Liquid types (Table 1)

| field | meaning | Eq. |
|---|---|---|
| `mu_v` | viscosity | 2, 4, 9 |
| `mu_e` | elasticity | 2, 5 |
| `omega_D` | diffusivity weight | 8 |
| `r_c` | capillary radius (absorption; 0 means none) | 9, 10 |
| `gamma` | surface tension | 10 |
| `theta` | contact angle (not given; default 0) | 10 |
| `alpha`, `yield_` | elastic decay rate, yield point (Appendix A; not given) | A.6, A.7 |

Kinds: `0 = binder` (paint = binder + pigment), `1 = solvent`.

## Frame file contract (`frame_%05d.npz`), written by the solver, read by the renderer

| key | dtype / shape | meaning |
|---|---|---|
| `x` | float32 (N,3) | liquid particle positions |
| `v` | float32 (N,3) | velocities |
| `kind` | int8 (N,) | 0 binder, 1 solvent |
| `c` | float32 (N,K) | pigment concentration per pigment (paint starts at 1, solvent at 0) |
| `absorbed` | bool (N,) | True once absorbed into the paper (excluded from the surface, Sec. 5.1) |
| `paper_x` | float32 (P,3) | paper particle positions (top layer at z = 0) |
| `paper_c` | float32 (P,K) | pigment left on the paper (concentration x particle volume / paper particle area: a surface density, m) |
| `pigment_rgb` | float32 (K,3) | linear-RGB colour of each pure pigment |
| `d0`, `h`, `time` | float scalars | spacing, smoothing radius, sim time |

## Rendering (Sec. 5)

- Liquid surface: Zhu & Bridson (Eq. 11) signed distance on a grid, absorbed particles excluded; marching cubes for meshes.
- Pigment: grid density field, concentration gathered within radius `r` per cell; paper gets pigment density where
  liquid touches it. The paper rendered with mental ray in Maya; we ray-march our own renderer (same inputs).
