# Paper -> code map

You, Jang, Cha, Kim, Noh. *Realistic paint simulation based on fluidity, diffusion, and absorption.*
Comp. Anim. Virtual Worlds 2013; 24:297-306. DOI 10.1002/cav.1500.

Every equation in the paper, where it lives in the code, and how it is tested.

| Paper | What it says | Code | Test |
|---|---|---|---|
| Sec. 3-4 | Four ingredients: pigment, binder, solvent, paper. Paint = binder + pigment. | `params.Liquid(kind=BINDER/SOLVENT)`, per-particle `c` (one column per pigment), paper = boundary particles flagged `bpaper` | - |
| Eq. (1) | `A(r) = sum_j m A_j/rho_j W(r - r_j, h)`; density by summation | `kernels.density` (poly6, Mueller 2003 = ref. [25]) | `test_rest_density_of_lattice` |
| Eq. (2) | `du/dt = -(1/rho) grad p + mu_v lap u + mu_e div eps + f`, convective term dropped (Lagrangian) | `solver.Simulation.step` | - |
| Eq. (3) | `-grad p(r_i) = -sum_j m^2 (p_i/rho_i^2 + p_j/rho_j^2) grad W` | `kernels.pressure_acc` (divided by m_i -> acceleration), pressure from PCISPH (ref. [7]) | `test_hydrostatic_rest_pcisph` |
| Sec. 4.1 | "We utilize the PCISPH method" | `solver.step` PCISPH loop: predict, density error, `p += delta * err`, `delta` from a filled prototype (Solenthaler & Pajarola 2009 Eq. 8), eta = 1 %, >= 3 iterations | same |
| Eq. (4) | `mu_v lap u(r_i) = mu_v sum_j m (u_j - u_i)/(rho_i rho_j) lap W` | `kernels.nonpressure_acc`, viscosity-kernel Laplacian | - |
| Eq. (5) | `mu_e div eps(r_i) = mu_e sum_j m/(rho_i rho_j) eps_i . grad W`, binder particles only | `kernels.nonpressure_acc`, `Config.elastic_form` | `test_eq5_forms_on_linear_strain_field` |
| App. A (A.1-A.7) | elastic strain: total rate `(grad u + grad u^T)/2`, `grad u = sum m/rho_j (u_j-u_i) (x) grad W`, von Mises plastic flow `alpha eps'/|eps'| max(0, |eps'| - gamma)` | `kernels.update_strain` | `test_appendix_a_strain_rate_and_yield` |
| Eq. (6) | Fick's second law `dc/dt = D lap c` | `solver._diffuse` | - |
| Eq. (7) | `D lap c(r_i) = D sum_j m_j (c_j - c_i)/(rho_i rho_j) lap W` | `kernels.diffuse` | `test_eq7_constant_diffusivity_matches_erf` |
| Eq. (8) | curvature diffusivity `D = omega_D (-lap c)/|grad c|` | `kernels.diffusivity`, `Config.curvature` | `test_eq8_literal_sign_cannot_spread`, `test_eq8_abs_spreads_and_pair_mode_conserves` |
| App. B | advection term dropped because pigment moves with the particles | `c` is carried by particles; no advection step | - |
| Eq. (9) | Lucas-Washburn `dl/dt = (P_h + P_c)/(8 mu l) (r_c^2 + 4 r_c)` | `kernels.absorption` (exact integration `l^2 += 2 K dt`) | `test_eq9_lucas_washburn_exact_integration` |
| Eq. (10) | `P_c = 2 gamma cos(theta)/r_c` | `params.capillary_pressure`, inlined in `kernels.absorption` | same |
| Sec. 4.3 | absorption only where liquid touches paper; binder + solvent absorbed, pigment stays; `r_c = 0` means no absorption | `kernels.absorption`, `kernels.deposit`, `kernels.contact_stain` | `test_no_absorption_when_rc_zero` |
| Eq. (11) | Zhu & Bridson surface `phi(x) = |x - x_bar| - r_bar`, absorbed particles excluded | `surface.zhu_bridson_sdf` | `tests/test_render.py` |
| Sec. 5.2 | pigment as grid density gathered within radius r; paper gets pigment where liquid touches it | `surface.pigment_grid`, `surface.paper_pigment_map` | `tests/test_render.py` |
| Sec. 5 | render liquid geometry and pigment density together (mental ray in Maya) | `render.render_frame` (own ray-marcher; meshes exported as OBJ for external renderers) | - |
| Table 1 | Pollock binder 80 / 1000 / 0.001 / 0 / 0.028; flower binder 10 / 0 / 0.038 / 0 / 0.028; flower solvent 5 / 0 / 0.05 / 2.5e-4 / 0.07275 (`mu_v, mu_e, omega_D, r_c, gamma`) | `params.POLLOCK_BINDER`, `FLOWER_BINDER`, `FLOWER_SOLVENT` | - |
| Sec. 6 | Pollock dt = 0.0002; CASA paint mu_v = 40, omega_D solvent 0.01 / binder 0.002, solvent r_c = 2.5e-4 | `params.CASA_*`, `scenes/figures.py` | - |
| Fig. 5-8 | viscoelastic vs viscous vs basic; Pollock; flower; CASA logo | `scenes/figures.py` (`python -m paintsim.scenes fig5|fig6|fig7|fig8 --out DIR`) | runs in `_runs/` |

## Where the printed paper is ambiguous or wrong, and what the code does

1. **Eq. (8) sign.** Taken literally, `D lap c = -omega_D (lap c)^2 / |grad c| <= 0` everywhere: concentration can
   only fall, and pigment can never enter clean solvent. `test_eq8_literal_sign_cannot_spread` shows exactly that.
   Default `curvature='abs'` uses the curvature magnitude `|lap c|/|grad c|`. That keeps the paper's idea (faster
   diffusion where the front is more curved, which gives irregular fronts). `'literal'` and `'constant'` are kept as
   switches.
2. **Eq. (8) operators and the singularity.** `lap c` and `grad c` in Eq. 8 are the standard SPH operators (both with
   `m/rho_j`), so their ratio is a geometric curvature in 1/m. Eq. 7 is applied exactly as printed (with the extra
   `1/rho_i`, so the effective diffusivity is `D/rho_i`; the erf test confirms that). At concentration extrema
   `|grad c| -> 0` makes D blow up, so |curvature| is limited to `1/d0` (a particle cloud cannot represent a
   sharper level set). Explicit diffusion is sub-stepped for stability instead of clipping D.
3. **Eq. (7) and conservation.** With D_i outside the sum, Eq. 7 does not conserve pigment when D varies.
   `diffusion_pairing='particle'` is the printed form (default); `'pair'` puts `(D_i + D_j)/2` inside and conserves
   `sum c` exactly.
4. **Eq. (5) uses only eps_i.** Since `sum_j m/rho_j grad W ~ 0` inside the fluid, the printed form only acts at free
   surfaces (`test_eq5_forms...`). `'difference'` (eps_j - eps_i, the same pattern as Eq. 4 and Eq. 7) and
   `'symmetric'` are the standard SPH divergences. Default: `'literal'`.
5. **Eq. (3) has m^2.** That is the force; the code uses the acceleration (divide by m), consistent with Eq. 2.
6. **Flower parameters.** The text says solvent omega_D = 0.038 and r_c = 2.5e-5; Table 1 says binder 0.038, solvent
   0.05, r_c = 2.5e-4. We follow Table 1. With Eq. 9 the 4 r_c term dominates, so the absorption speed barely depends on
   r_c anyway: K ~ gamma cos(theta) / mu.

## Gaps (not in the paper) and the choices made

| Gap | Choice | Where |
|---|---|---|
| particle spacing, smoothing radius | d0 = 0.01 m, h = 2 d0 for the metre-scale figures (d0 = 0.006 for the flower). With mu_v = 80 the explicit viscous limit then lands on the paper's dt = 0.0002 s | `scenes/figures.py`, `SPEC.md` |
| kernels | Mueller 2003 (the paper cites it as ref. [25]): poly6 / spiky / viscosity | `kernels.py` |
| rest density, mass | rho0 = 1000 for every liquid, uniform particle mass from a filled lattice | `Simulation._prototype` |
| boundary handling for paper | Akinci et al. 2012 boundary particles (psi = rho0 / sum W), 2 layers; paper viscous drag; particle centres kept >= d0/2 above the paper | `kernels.density/pressure_acc/nonpressure_acc`, `Config.contact_floor` |
| when is a particle "absorbed" | when its Lucas-Washburn length l reaches `absorb_depth` (default d0) while it touches the paper | `Config.absorb_depth` |
| P_h | `rho0 g H`, H = liquid column above the particle (d0 x d0 columns) | `kernels.column_heights` |
| contact angle theta | 0 (fully wetting) | `Liquid.theta` |
| mu in Eq. 9 | the liquid's mu_v | `kernels.absorption` |
| pigment left on paper | all of an absorbed particle's pigment (`deposit_fraction = 1`), spread over the touching paper particles; plus a contact stain where liquid touches paper | `kernels.deposit`, `kernels.contact_stain` |
| Appendix A alpha, yield | Pollock alpha = 10 1/s, yield = 0.05 (our choice) | `params.POLLOCK_BINDER` |
| surface tension force | none (the paper has none; gamma only enters Eq. 10) | - |
| renderer | mental ray in Maya -> own ray-marcher, plus OBJ export | `render.py` |
