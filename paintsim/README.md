# paintsim

A rebuild of **You et al. 2013, "Realistic paint simulation based on fluidity, diffusion, and absorption"** (CAVW 24:297-306), written from the paper. The original code was never released.

Paint is made of particles. Binder particles carry pigment, and solvent particles start clear. They move as an SPH fluid (PCISPH), with viscosity and, for binder, elasticity. Pigment spreads from paint into solvent by Fick's law, with a curvature-based diffusivity. Liquid that touches the paper soaks in following Lucas-Washburn, and its pigment stays on the paper.

- `PAPER_MAP.md`: every equation, mapped to the code and its test. It also covers what the printed paper gets wrong or leaves out, and what the code does in each case.
- `SPEC.md`: units, world and the frame file format.

## Run

```
python -m pytest -q -p no:anyio paintsim/tests          # equation-level tests
python -m paintsim.scenes fig7 --out paintsim/_runs/fig7  # Fig. 5-8: fig5 | fig6 | fig7 | fig8
python -m paintsim.render paintsim/_runs/fig7 -o fig7.mp4 --camera top
```

Switches for the parts of the paper that are ambiguous: `--elastic-form literal|difference|symmetric` (Eq. 5), `--curvature abs|literal|constant` (Eq. 8), `--pairing particle|pair` (Eq. 7), `--absorb-depth` (absorption threshold).

## Files

| file | contents |
|---|---|
| `params.py` | `Liquid` (one Table 1 row), the paper's presets, `Config` |
| `kernels.py` | numba kernels: density, forces, strain, diffusion, absorption, deposition |
| `solver.py` | `Simulation`: one step = Sec. 4 in order, frame output |
| `grid.py` | hashed uniform-grid neighbour search |
| `scene.py` | lattices, paper sheet, walls, pouring nozzles |
| `scenes/figures.py` | Fig. 5 (viscoelastic / viscous / basic), Fig. 6 (Pollock), Fig. 7 (flower), Fig. 8 (CASA) |
| `surface.py`, `render.py` | Sec. 5: Zhu-Bridson surface (Eq. 11), pigment density grid, ray-marched renderer, OBJ export |
| `preview.py` | quick particle scatter for debugging |

Runs on the CPU with numba, the same way the paper used C++ and OpenMP.
