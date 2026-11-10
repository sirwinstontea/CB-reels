# Alcohol markers: physics research for the ink rebuild

Research done 2026-11-10 (five research passes plus spot checks). Purpose: the reference for rebuilding
`reelmachine/engine/ink.py` (today a flat alpha fill) as a particle-based physical ink model, extending the
You et al. 2013 engine in `paintsim/`.

Evidence tags: **[src]** read in a primary source; **[snip]** only seen in a search snippet; **[est]** our estimate
or a value recalled from the literature, to be calibrated.

## 1. What is in the ink

**Copic SDS** (material H001, v5.0, 2023) [src]: ethanol 65-85 %, propan-1-ol < 10 %, isopropanol < 5 %,
ethoxylated C12-14 alcohol (nonionic surfactant) < 5 %, BHT < 1 %. The dyes are not listed. An older Copic SDS
lists confidential Cr-, Co- and Cu-complex dyes (< 7 %, < 3.5 %, < 7 %) [snip], and one snippet gives a rosin-maleic
resin at 5-12 % [snip]. https://cdn.modulor.de/hazardous/e2c756fcb2e75a3b3b187b2c8d597b17.pdf

**Patents** for ethanol marker ink (US 5,423,907 [src]; US 5,279,653 and US 5,182,324 [snip]):
- ethanol 65-98 %, at least 90 % of the solvent;
- dye 1-20 %: about 3 % for pale colours (pink, sky blue) and about 10-12 % for dark ones;
- ethanol-soluble resin 1-15 % (ketone or alkylphenol), which fixes the dye and helps it level out;
- small co-solvents (n-propanol, IPA, benzyl alcohol) to stop the dye crystallising;
- optional glycol to keep the nib from drying.

**Particle-level picture**
- The colourant is metal-complex azo solvent dye, 400-1000 g/mol and about 1-2 nm across. It is **dissolved
  molecules (a true solution), not pigment particles.**
- The dye is soluble in alcohol and insoluble in water. That is why dried marker re-dissolves under fresh ink or
  blender, and why water does nothing to it.
- Concentration is about 0.07-0.2 mol/L.
- After drying, the dye (plus any resin) sits on and in the cellulose fibres.

**Properties**

| | ethanol | IPA | ink (est.) |
|---|---|---|---|
| viscosity mPa·s | 1.07-1.2 [src] | 2.0-2.4 | 1.5-4 [est] |
| surface tension mN/m | 22 [src] | 21.7 | 22-25 [est] |
| density g/cm3 | 0.787 | 0.785 | 0.8-0.85 [src SDS] |
| refractive index | 1.36 [src] | 1.378 | - |
| vapour pressure kPa, 20 C | 5.8 [src SDS] | 4.4 | - |
| contact angle on cellulose | ~0 (wets fully, even sized paper) | | |

A Copic Sketch holds about 1.8 mL and a Classic about 2.5 mL [src].

## 2. How alcohol moves in coloring-book paper

**Paper**

| Property | Typical value |
|---|---|
| Thickness h | 95-120 um (80-100 g/m2) |
| Porosity phi | 0.45-0.55 |
| Modal pore radius | 1-5 um |
| Fibres | 1-3 mm long, 15-40 um wide, lying in-plane |
| Capacity per area, h·phi | **about 5 uL/cm2** (50 um of liquid) [est] |

Note: one research pass said 40 uL/cm2, which is a unit error.

**Sizing**
- Alcohol (22 mN/m) is below the critical surface tension of sized cellulose (25-30 mN/m), so it wets sized paper
  that repels water. That is why markers bleed through ordinary paper.
- Alcohol barely swells fibres, so there is no buckling or pilling.

**Measured, ethanol in Whatman filter papers** (Murthy & Kumar, Int. J. Therm. Sci. 210, 109619, 2025,
arXiv 2407.20776) [src]:
- permeability K = 0.66-4.9 x 10^-12 m2;
- pore radius 59-75 um (manufacturer value), porosity 0.48-0.76;
- **evaporation flux up to 2.5-2.8 x 10^-4 kg/m2/s**, i.e. about 0.35 um/s of liquid;
- evaporation, not gravity, caps the wetted length (1.2 cm instead of 9.6 cm);
- evaporation is not uniform: it is highest near the wet front.

Coloring-book paper is denser than filter paper. Expect K of about 1e-13 to 5e-13 m2 [est].

**Through the thickness**
- Lucas-Washburn depth is l^2 = gamma r cos(theta) t/(2 mu). For ethanol it penetrates 100 um in about 1-15 ms
  [est].
- **So the vertical soak is effectively instant.** The useful model treats depth as one lumped, depth-averaged
  saturation with a capacity limit. The paper's per-particle Lucas-Washburn timing does not matter at this scale.

**In-plane (the main event): the Richards equation, i.e. Darcy flow driven by capillary suction**
- Flow law: phi dS/dt = div( (K k_r(S)/mu) grad P_c(S) ) - evaporation.
- Diffusion form: dS/dt = div( D(S) grad S ).
- Brooks-Corey closure:
  - P_c = P_e S_e^(-1/lambda), with P_e = 2 gamma/r_entry, about 9-45 kPa for ethanol;
  - k_r = S_e^((2+3 lambda)/lambda);
  - which gives D(S) = (K P_e/(mu phi lambda)) S_e^(2+1/lambda).
- Fei et al. 2018 instead use D linear in S, which avoids the singularity at S -> 0.
- D0 is about 1-6 mm2/s [est]. The front stays sharp on the wet side and soft at the foot. Its radius grows
  roughly as sqrt(t).
- **Volume caps the spread:** r_max = sqrt(V/(pi h phi S)). For example, 5 uL gives about a 6 mm radius.
- Fibre orientation makes the stain slightly elliptical: K_MD/K_CD is about 1.1-1.6 [est].

**Nib to paper**
- The nib's felt pores are 20-100 um (suction 0.5-2 kPa). Paper pores are 1-5 um (9-45 kPa).
- The paper therefore pulls ink out of the nib. The supply behaves as constant-pressure while the nib is touching,
  so slower strokes deposit more ink.
- No measured ink volume per stroke exists. Guess 0.1-1 uL per cm of stroke and calibrate.

**Bleed-through:** happens when the local dose exceeds capacity (S close to 1 through the full thickness).

## 3. How the colour (dye) moves

Closest published model: Venditti, Murali, Darhuber, "Chromatographic effects in inkjet printing", Langmuir 37,
11726 (2021) [src via agent]. It is a 2D, depth-averaged model of Richards flow plus dye advection, dispersion and
adsorption, plus evaporation, on 104 um paper. Use it as the template.

- **Advection-dispersion-adsorption equation**
  - Mobile dye: d(theta c)/dt + div(theta c v) = div(theta D_disp grad c) - adsorption.
  - Kinetic Langmuir adsorption: dC_ad/dt = k_ad c (1 - C_ad/C_max) - k_de c_ref C_ad/C_max.
  - D_disp = D_mol/tau^2 + alpha_L |v|.
    - D_mol for dye in ethanol is about 3-5e-10 m2/s [est].
    - alpha_L is about 10-50 um [est].
  - Molecular diffusion is tiny: about 25 um in 1 s. **Colour moves because the liquid carries it**
    (Peclet number in the thousands). The paper's curvature-diffusion Eq. 8 is the wrong main mechanism here; at most
    it adds a little soft blur.
- **Chromatography**
  - The dye front lags the solvent front: Rf = 1/R, with R = 1 + (rho_b/phi) K_d.
  - Rf depends on concentration (S-shaped): pale doses are held back (low Rf), and doses above sorption capacity run
    freely (Rf near 1).
  - So dark colours bleed and blend easily, while pale tints move less.
  - No Rf data exists for marker dyes in ethanol. Default Rf is about 0.6-0.95, and different dyes in one ink differ
    by 0.1-0.3 [est].
- **Adsorption is slow compared with drying**
  - Adsorption takes minutes (aqueous cellulose data: 5-100 min); ethanol dries in seconds to a minute.
  - **So dye mostly gets fixed by drying** (it is left behind where the liquid vanishes). This is MoXi's fixture law.
- **Dark edges and tide lines**
  - The dye does not evaporate. As the liquid leaves, c rises, and the edges dry fastest (flux about
    (1 - r^2/R^2)^(-1/2), Deegan 1997).
  - Liquid flowing toward the edge to replace what evaporated carries dye there.
  - A front that stalls on dry paper strands its dye, which gives a sharp, pore-scale dark line.
  - Marangoni flow is negligible inside paper: capillary pressure (~1e4 Pa) far exceeds Marangoni stress (~30 Pa).
- **Re-dissolution**
  - Dried dye goes back into solution when alcohol returns, over a few seconds [est].
  - There may be a non-liftable residue, which is unknown; make it a tunable fraction.

**Effect -> mechanism**

| Effect | Cause |
|---|---|
| Flat fill | Region stays connected and wet; Darcy flow evens out saturation and dye before anything dries |
| Streak or line | New stroke meets a half-dry area: double dose, the front stalls at the old edge, and re-dissolved dye piles up there |
| Dark rim / tide line | Edge evaporation plus outward refill flow strands dye at the front |
| Feathering | Capillary spread limited by volume, along a random fibre/pore field |
| Halo | Solvent front runs ahead of the dye front (Rf < 1) |
| Wet-in-wet blend | Two wet regions connect and the flow mixes them |
| Blender lightening | Pure solvent re-dissolves dye and sweeps it outward; it dries as a ring |
| Darker layers | Dye per area adds up and saturates at sorption capacity and the colour's full-strength floor |
| Bleed-through | Dose exceeds sheet capacity; dye reaches the back face |

## 4. How dye amount becomes the visible colour

**Kubelka-Munk**, per wavelength band:
- K = K_paper + sum_i c_i eps_i. Dye only absorbs.
- S = S_paper, about 40-80 m2/kg; an 80 g/m2 sheet is nearly opaque, with R_inf about 0.85-0.9.
- R_inf = 1 + K/S - sqrt((K/S)^2 + 2K/S).
- For a finite sheet over a backing, use the coth form.
- Saunderson surface correction: k1 is about 0.04; k2 is about 0.6 dry and 0.5 wet.

**Subtractive mixing needs spectra, not RGB**
- Use 8-16 bands (400-700 nm). Blue + yellow then gives green naturally.
- Get each marker's eps_i(lambda) by inverting KM on a swatch:
  - eps_i = (K/S of swatch - K/S of paper) x S / c_ref;
  - with only an sRGB value, first upsample it to a spectrum (Jakob & Hanika 2019, "rgb2spec").

**Wet looks darker**
- Alcohol fills the pores, which matches refractive index: per-interface reflectance drops from 0.044 to 0.0035
  (cellulose 1.53 vs ethanol 1.36). So scattering S falls while K stays the same.
- Model: S(sigma) = S_dry [(1 - sigma) + sigma f_w], with f_w about 0.3 (range 0.2-0.5) [est].
- Drying then lightens the colour automatically. Expect a lightness change of about 5-15 [est]; calibrate it on a
  real video.

**Line art:** laser toner and offset ink resist ethanol. Dye-based inkjet line art smears. Treat printed lines as an
opaque top layer.

## 5. Simulation models to build on

| Model | Use it for |
|---|---|
| You et al. 2013 (`paintsim`) | Free ink particles above the paper (PCISPH); keep Lucas-Washburn only as the soak rate law |
| Lenaerts, Adams, Dutré 2008 (SIGGRAPH) | Paper as porous particles holding a saturation scalar; pairwise conservative Darcy exchange; absorption by shrinking the fluid particle |
| Fei et al. 2018 (SIGGRAPH, read in full) | Capillary suction p = 2 sigma cos theta/((1-phi) r); Richards with linear D(S); mass-conserving capture; drip when over capacity |
| Boettcher et al. 2025 (arXiv 2504.07739, read in full) | Overlapping-phase density and saturation definitions; capillary adhesion C_cap(S) = C0 (1 - S^a) |
| Curtis et al. 1997 | Edge darkening (lower pressure at the wet-mask edge); deposit/lift rules; backruns |
| Chu & Tai 2005, MoXi | Fibre-derived random permeability/pinning field; pigment fixture that rises as the paper dries |
| Venditti et al. 2021 | The physical template for dye: depth-averaged Richards + ADE + Langmuir + evaporation |

No graphics paper simulates alcohol markers specifically, so this combination would be new.

## 6. Architecture these findings point to (particles throughout)

**Paper particles** sit on a fixed lattice at about the page-pixel scale (~0.1-0.2 mm). Each one carries:
- porosity phi and capacity;
- a noisy pore radius r (the fibre texture), which gives P_e and K, with K anisotropic;
- saturation S;
- per dye: mobile mass, fixed mass, and optionally a back-face share.

Neighbour lists are computed once.

**Ink particles** are the liquid the nib emits: low-viscosity PCISPH, each carrying mass and per-dye mass. They
soak in within milliseconds.

**Per step:**
1. **Soak:** ink particle -> paper particles under it, limited by free capacity phi V (1 - S). Dye moves in
   proportion. The ink particle shrinks and is deleted when empty.
2. **Wick:** a pairwise conservative SPH Laplacian of S (Cleary-Monaghan form, harmonic mean of D(S)). Sub-step or
   solve implicitly, because D is large.
3. **Carry dye:**
   - mobile dye moves upwind with the pair liquid fluxes from step 2;
   - add small dispersion;
   - optional slow Langmuir adsorption to get chromatographic lag.
4. **Evaporate:** dS/dt = -j/(rho h phi), with j about 2.5e-4 kg/m2/s, raised at the wet edge. Dye stays behind.
5. **Fix and re-dissolve:**
   - below S_crit (~0.1-0.2), mobile dye becomes fixed;
   - when re-wetted, fixed dye returns to mobile at rate k_rd (timescale of a few seconds), leaving a residual
     fraction.
6. **Bleed:** dose above capacity goes to the back face.
7. **Render:** splat paper particles to pixels, then spectral KM with wet scattering S(sigma), the line-art layer
   and the Saunderson correction.

**Calibrate on real footage:** D0, evaporation flux, ink per cm of stroke, f_w, Rf, k_rd. Everything else follows
from conservation of liquid volume and dye mass.

## Sources

- Copic SDS: https://cdn.modulor.de/hazardous/e2c756fcb2e75a3b3b187b2c8d597b17.pdf
- US 5,423,907: https://patents.google.com/patent/US5423907
- US 5,698,614: https://patents.google.com/patent/US5698614
- Murthy & Kumar 2025: https://arxiv.org/abs/2407.20776
- Venditti et al. 2021: https://doi.org/10.1021/acs.langmuir.1c01624
- Syms 2017 (dye transport in wicking paper with evaporation): https://pmc.ncbi.nlm.nih.gov/articles/PMC5570596
- Deegan et al. 1997: https://www.nature.com/articles/39827
- Fei et al. 2018: https://www.cs.columbia.edu/cg/wetcloth/main.pdf
- Boettcher et al. 2025: https://arxiv.org/pdf/2504.07739v2
- Lenaerts et al. 2008: ACM TOG 27(3) 49
- Curtis et al. 1997: https://www.cs.princeton.edu/courses/archive/fall00/cs597b/papers/curtis97.pdf
- MoXi (Chu & Tai 2005; equations via US 8,296,668): https://patents.google.com/patent/US8296668
- Li et al. 2015, ink penetration in uncoated paper: https://bioresources.cnr.ncsu.edu/BioRes_10/BioRes_10_4_8135_Ink_Penetration_Uncoated_Inkjet_Paper_7740.pdf
- Kubelka-Munk / Saunderson / Jakob & Hanika 2019 (doi 10.1111/cgf.13626) / Sochorova & Jamriska 2021 (Mixbox,
  doi 10.1145/3478513.3480549); Twomey et al. 1986 (wet darkening, doi 10.1364/AO.25.000431)
