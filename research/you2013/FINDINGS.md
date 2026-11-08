# You et al. 2013: sources that fill the paper's gaps

Research date: 2026-11-08. Downloads are in `downloads/`; they are untrusted and not committed.

## Sources
- **T14**: Mi You, PhD thesis, KAIST GSCT 2014, "The Creation of Realistic Paint Simulation Reflecting the Real Painting Characteristics", 51 pp (bibctrlno 568448).
  - Viewable without login in the KAIST library ezPDF viewer.
  - Figures are in `downloads/thesis2014_figs/`.
- **K14**: Mi You, J. Korea Multimedia Society 17(7):906-914 (2014), doi 10.9717/kmms.2014.17.7.906. It reuses the same simulator.
- **I13**: Kim, You, Cha and Noh, Communications of KIISE, Nov 2013, pp. 29-33. A qualitative review.
- **Video**: the Wiley `PaintSimulation.avi` (1280x720, 24 fps, 67 s) returns 403 to curl. The same video is on YouTube as Xi2iUE7yGhQ; a copy and 12 frames are in `downloads/`.
- **Chang et al. 2009** (ref. [32]) and **Goktekin et al. 2004** (ref. [29]).

## Gap 1: absorption
- **No threshold given.** No source states a threshold, paper thickness, P_h or contact angle theta.
- **Absorbed particles are kept.** Thesis Fig. 5.4 (side view) shows absorbed particles passing below one paper plane y = y_p and staying there, in clumps 1-4 diameters deep.
- **Pigment goes onto a paper grid.** It accumulates on a 2D paper grid "when the liquid particles pass through the paper". That grid is rendered as (mask + darkening) x upsampled pigment x paper texture.
- **mu in Eq. 9** is each liquid's own mu_v.
- **r_c is set per liquid.** The flower binder has r_c = 0.
- **Solvent is meant to absorb very fast** (K14 p. 910). The pools still stay visible for many frames because each frame covers very little simulated time.
  - A pre-visualisation screenshot shows 4542 steps over 13 frames at dt = 3.2e-5, i.e. about 11 ms of simulated time per frame.
- **(r_c^2 + 4 r_c)** is Washburn's (r^2 + 4 eps r) with the slip eps dropped. Every version the authors published prints it this way.

## Gap 2: Eq. 5, alpha and yield
- **The eps_i-only form is deliberate.** It is copied from Chang 2009 Eq. 12 (f_i = sum_j ((mu_s,i + mu_s,j)/2) (m_j/rho_j) S_i . grad W_ij), so it is not a typo.
- **No alpha or yield values** appear in any You source. Goktekin Fig. 2 uses alpha in {25, 500} and yield in {0, 0.1, 0.5}.

## Gap 3: Eq. 8
Nothing beyond the paper. The thesis prints it identically, and no source re-implements it.

## Gap 4: discretisation
- **Spacing, h and rho0** are not stated anywhere.
- **Thesis code screenshot**: PCISPH with a fixed `minIterations = 3`, and uniform-grid neighbour search.
- **Lab convention** (Jang thesis 2013): poly6, spiky and viscosity kernels; delta taken from an interior prototype particle.
- **Paper particles** look like a single layer (for example 23 028 = 228 x 101 for CASA).

### Particle counts from the video

| Scene | Paint | Solvent | Paper |
|---|---|---|---|
| Comparison | 12 141 each | - | - |
| Drop | 3 365 | 3 830 | 9 408 |
| Flower | 7 363 | 26 445 | 61 904 |
| CASA | 9 006 | 37 180 | 23 028 |

## Gap 5: inconsistencies
The thesis repeats both the text and Table 1 values, and there is no erratum. Table 1 is used.

## Video (what the reproductions should look like)
- **Fig. 5**: tubes squeeze paint onto a plane.
  - Red (viscoelastic) stacks into rope coils and keeps its folds.
  - Green (viscous) slumps into a mound.
  - Yellow (basic) spreads into a wide puddle.
- **Pollock**: glossy raised ropes of 4 colours, with no absorption.
- **Flower**: sharp petals at first, then they bleed into the solvent with a grainy, flat, matte texture.
- **CASA**: letters drop one after another into a translucent solvent slab, leaving soft halos.
