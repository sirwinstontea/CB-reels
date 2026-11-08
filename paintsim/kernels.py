"""Numba compute kernels for the You et al. 2013 rebuild.

Kernel functions are Mueller et al. 2003 (paper ref. [25]):
  poly6      W(r)       = 315/(64 pi h^9) (h^2 - r^2)^3         density, Eq. (1)
  spiky      grad W(r)  = -45/(pi h^6) (h - r)^2 r_hat           pressure Eq. (3), elasticity Eq. (5), grad u Eq. (A.4), grad c
  viscosity  lap W(r)   = 45/(pi h^6) (h - r)                    every nabla^2 W: Eq. (4), Eq. (7)
grad W is always grad_i W(r_i - r_j).

Strain tensors are stored as 6 components: xx, yy, zz, xy, xz, yz.
"""
import math

import numpy as np
from numba import njit, prange

PI = math.pi


def coefs(h):
    return (315.0 / (64.0 * PI * h ** 9),   # poly6
            -45.0 / (PI * h ** 6),           # spiky gradient magnitude factor
            45.0 / (PI * h ** 6))            # viscosity laplacian factor


# --------------------------------------------------------------------------- density, Eq. (1)
@njit(parallel=True, cache=True)
def density(x, nbr, cnt, bx, bnbr, bcnt, bpsi, m, h, cp6, out):
    """rho_i = sum_j m W(r_i - r_j, h) over liquid neighbours (and self) plus Akinci boundary term sum_k psi_k W."""
    h2 = h * h
    n = x.shape[0]
    for i in prange(n):
        s = m * cp6 * h2 * h2 * h2
        for a in range(cnt[i]):
            j = nbr[i, a]
            rx = x[i, 0] - x[j, 0]
            ry = x[i, 1] - x[j, 1]
            rz = x[i, 2] - x[j, 2]
            r2 = rx * rx + ry * ry + rz * rz
            if r2 < h2:
                d = h2 - r2
                s += m * cp6 * d * d * d
        for a in range(bcnt[i]):
            k = bnbr[i, a]
            rx = x[i, 0] - bx[k, 0]
            ry = x[i, 1] - bx[k, 1]
            rz = x[i, 2] - bx[k, 2]
            r2 = rx * rx + ry * ry + rz * rz
            if r2 < h2:
                d = h2 - r2
                s += bpsi[k] * cp6 * d * d * d
        out[i] = s


@njit(cache=True)
def boundary_psi(bx, bnbr, bcnt, h, cp6, rho0):
    """Akinci et al. 2012 boundary volume: psi_k = rho0 / sum_l W_kl (boundary-boundary, incl. self)."""
    nb = bx.shape[0]
    psi = np.empty(nb)
    h2 = h * h
    for k in range(nb):
        s = cp6 * h2 * h2 * h2
        for a in range(bcnt[k]):
            l = bnbr[k, a]
            rx = bx[k, 0] - bx[l, 0]
            ry = bx[k, 1] - bx[l, 1]
            rz = bx[k, 2] - bx[l, 2]
            r2 = rx * rx + ry * ry + rz * rz
            if r2 < h2:
                d = h2 - r2
                s += cp6 * d * d * d
        psi[k] = rho0 / s
    return psi


# --------------------------------------------------------------------------- strain, Appendix A
@njit(parallel=True, cache=True)
def update_strain(x, v, rho, lid, eps, nbr, cnt, m, h, csp, elastic_tab, alpha_tab, yield_tab, dt):
    """Eq. (A.2)-(A.7): eps_{t+dt} = eps_t + ((grad u + grad u^T)/2 - alpha eps'/|eps'|_F max(0, |eps'|_F - gamma)) dt,
    grad u = sum_j m/rho_j (u_j - u_i) (x) grad W (A.4), eps' = eps - tr(eps)/3 I (A.5)."""
    n = x.shape[0]
    for i in prange(n):
        L = lid[i]
        if elastic_tab[L] == 0:
            continue
        g00 = 0.0; g01 = 0.0; g02 = 0.0
        g10 = 0.0; g11 = 0.0; g12 = 0.0
        g20 = 0.0; g21 = 0.0; g22 = 0.0
        for a in range(cnt[i]):
            j = nbr[i, a]
            rx = x[i, 0] - x[j, 0]
            ry = x[i, 1] - x[j, 1]
            rz = x[i, 2] - x[j, 2]
            r = math.sqrt(rx * rx + ry * ry + rz * rz)
            if r >= h or r < 1e-12:
                continue
            f = csp * (h - r) * (h - r) / r
            gx = f * rx
            gy = f * ry
            gz = f * rz
            w = m / rho[j]
            du0 = (v[j, 0] - v[i, 0]) * w
            du1 = (v[j, 1] - v[i, 1]) * w
            du2 = (v[j, 2] - v[i, 2]) * w
            g00 += du0 * gx; g01 += du0 * gy; g02 += du0 * gz
            g10 += du1 * gx; g11 += du1 * gy; g12 += du1 * gz
            g20 += du2 * gx; g21 += du2 * gy; g22 += du2 * gz
        # total strain rate (A.3)
        dxx = g00
        dyy = g11
        dzz = g22
        dxy = 0.5 * (g01 + g10)
        dxz = 0.5 * (g02 + g20)
        dyz = 0.5 * (g12 + g21)
        # von Mises plasticity (A.5, A.6)
        exx = eps[i, 0]; eyy = eps[i, 1]; ezz = eps[i, 2]
        exy = eps[i, 3]; exz = eps[i, 4]; eyz = eps[i, 5]
        tr3 = (exx + eyy + ezz) / 3.0
        pxx = exx - tr3
        pyy = eyy - tr3
        pzz = ezz - tr3
        nf = math.sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (exy * exy + exz * exz + eyz * eyz))
        k = 0.0
        if nf > 0.0:
            k = alpha_tab[L] * max(0.0, nf - yield_tab[L]) / nf
        # (A.7)
        eps[i, 0] = exx + (dxx - k * pxx) * dt
        eps[i, 1] = eyy + (dyy - k * pyy) * dt
        eps[i, 2] = ezz + (dzz - k * pzz) * dt
        eps[i, 3] = exy + (dxy - k * exy) * dt
        eps[i, 4] = exz + (dxz - k * exz) * dt
        eps[i, 5] = eyz + (dyz - k * eyz) * dt


# --------------------------------------------------------------------------- non-pressure forces, Eq. (2), (4), (5)
@njit(parallel=True, cache=True)
def nonpressure_acc(x, v, rho, lid, eps, nbr, cnt, bx, bnbr, bcnt, bpsi, m, h, csp, clap,
                    muv_tab, mue_tab, elastic_tab, gx, gy, gz, elastic_form, bvisc, rho0, out):
    """a_i = g + mu_v sum_j m (u_j - u_i)/(rho_i rho_j) lap W        (Eq. 4)
             + mu_e sum_j m/(rho_i rho_j) eps_i . grad W              (Eq. 5, binder only; see elastic_form)
             + boundary viscous drag (paper/walls at rest, Eq. 4 with psi_k as neighbour mass)."""
    n = x.shape[0]
    for i in prange(n):
        L = lid[i]
        mv = muv_tab[L]
        me = mue_tab[L]
        el = elastic_tab[L] != 0 and me > 0.0
        ax = gx
        ay = gy
        az = gz
        ri = rho[i]
        for a in range(cnt[i]):
            j = nbr[i, a]
            rx = x[i, 0] - x[j, 0]
            ry = x[i, 1] - x[j, 1]
            rz = x[i, 2] - x[j, 2]
            r = math.sqrt(rx * rx + ry * ry + rz * rz)
            if r >= h or r < 1e-12:
                continue
            rj = rho[j]
            if mv > 0.0:
                f = mv * m / (ri * rj) * clap * (h - r)
                ax += f * (v[j, 0] - v[i, 0])
                ay += f * (v[j, 1] - v[i, 1])
                az += f * (v[j, 2] - v[i, 2])
            if el:
                q = csp * (h - r) * (h - r) / r
                wx = q * rx
                wy = q * ry
                wz = q * rz
                if elastic_form == 0:      # literal: eps_i
                    c = me * m / (ri * rj)
                    e0 = eps[i, 0]; e1 = eps[i, 1]; e2 = eps[i, 2]
                    e3 = eps[i, 3]; e4 = eps[i, 4]; e5 = eps[i, 5]
                elif elastic_form == 1:    # difference: eps_j - eps_i
                    c = me * m / (ri * rj)
                    e0 = eps[j, 0] - eps[i, 0]; e1 = eps[j, 1] - eps[i, 1]; e2 = eps[j, 2] - eps[i, 2]
                    e3 = eps[j, 3] - eps[i, 3]; e4 = eps[j, 4] - eps[i, 4]; e5 = eps[j, 5] - eps[i, 5]
                else:                      # symmetric: eps_i/rho_i^2 + eps_j/rho_j^2
                    c = me * m
                    a1 = 1.0 / (ri * ri)
                    a2 = 1.0 / (rj * rj)
                    e0 = eps[i, 0] * a1 + eps[j, 0] * a2; e1 = eps[i, 1] * a1 + eps[j, 1] * a2
                    e2 = eps[i, 2] * a1 + eps[j, 2] * a2; e3 = eps[i, 3] * a1 + eps[j, 3] * a2
                    e4 = eps[i, 4] * a1 + eps[j, 4] * a2; e5 = eps[i, 5] * a1 + eps[j, 5] * a2
                ax += c * (e0 * wx + e3 * wy + e4 * wz)
                ay += c * (e3 * wx + e1 * wy + e5 * wz)
                az += c * (e4 * wx + e5 * wy + e2 * wz)
        if bvisc and mv > 0.0:
            for a in range(bcnt[i]):
                k = bnbr[i, a]
                rx = x[i, 0] - bx[k, 0]
                ry = x[i, 1] - bx[k, 1]
                rz = x[i, 2] - bx[k, 2]
                r = math.sqrt(rx * rx + ry * ry + rz * rz)
                if r >= h:
                    continue
                f = mv * bpsi[k] / (ri * rho0) * clap * (h - r)
                ax -= f * v[i, 0]
                ay -= f * v[i, 1]
                az -= f * v[i, 2]
        out[i, 0] = ax
        out[i, 1] = ay
        out[i, 2] = az


# --------------------------------------------------------------------------- pressure, Eq. (3)
@njit(parallel=True, cache=True)
def pressure_acc(x, rho, p, nbr, cnt, bx, bnbr, bcnt, bpsi, m, h, csp, out):
    """-(1/rho_i) grad p_i = -sum_j m (p_i/rho_i^2 + p_j/rho_j^2) grad W   (Eq. 3 divided by m_i)
    plus Akinci boundary term -sum_k psi_k p_i/rho_i^2 grad W_ik."""
    n = x.shape[0]
    for i in prange(n):
        pi = p[i] / (rho[i] * rho[i])
        ax = 0.0
        ay = 0.0
        az = 0.0
        for a in range(cnt[i]):
            j = nbr[i, a]
            rx = x[i, 0] - x[j, 0]
            ry = x[i, 1] - x[j, 1]
            rz = x[i, 2] - x[j, 2]
            r = math.sqrt(rx * rx + ry * ry + rz * rz)
            if r >= h or r < 1e-12:
                continue
            q = -m * (pi + p[j] / (rho[j] * rho[j])) * csp * (h - r) * (h - r) / r
            ax += q * rx
            ay += q * ry
            az += q * rz
        for a in range(bcnt[i]):
            k = bnbr[i, a]
            rx = x[i, 0] - bx[k, 0]
            ry = x[i, 1] - bx[k, 1]
            rz = x[i, 2] - bx[k, 2]
            r = math.sqrt(rx * rx + ry * ry + rz * rz)
            if r >= h or r < 1e-12:
                continue
            q = -bpsi[k] * pi * csp * (h - r) * (h - r) / r
            ax += q * rx
            ay += q * ry
            az += q * rz
        out[i, 0] = ax
        out[i, 1] = ay
        out[i, 2] = az


# --------------------------------------------------------------------------- diffusion, Eq. (6)-(8)
@njit(parallel=True, cache=True)
def diffusivity(x, rho, lid, c, nbr, cnt, m, h, csp, clap, omega_tab, curv_mode, grad_eps, kappa_max, paper_ops,
                D, lam):
    """Per particle and pigment: D = omega_D * curvature (Eq. 8), curvature = (-lap c)/|grad c| with
    standard SPH operators lap c = sum m/rho_j (c_j - c_i) lap W, grad c = sum m/rho_j (c_j - c_i) grad W.
    curv_mode 0 = |lap c|/|grad c| ('abs'), 1 = literal sign, 2 = constant omega_D.
    |curvature| is limited to kappa_max (the particle resolution cannot represent a sharper level set; it also
    removes the 0/0 singularity at concentration extrema).
    lam_i = sum_j m/(rho_i rho_j) lap W is the explicit-Euler stability sum (used to pick diffusion substeps)."""
    n = x.shape[0]
    K = c.shape[1]
    for i in prange(n):
        om = omega_tab[lid[i]]
        ri = rho[i]
        lm = 0.0
        for a in range(cnt[i]):
            j = nbr[i, a]
            rx = x[i, 0] - x[j, 0]
            ry = x[i, 1] - x[j, 1]
            rz = x[i, 2] - x[j, 2]
            r = math.sqrt(rx * rx + ry * ry + rz * rz)
            if r >= h:
                continue
            lm += m / (ri * rho[j]) * clap * (h - r)
        lam[i] = lm
        cap = om * kappa_max
        if paper_ops:
            cap = cap / ri
        for k in range(K):
            if curv_mode == 2:
                d = om
            else:
                ls = 0.0
                gx = 0.0
                gy = 0.0
                gz = 0.0
                for a in range(cnt[i]):
                    j = nbr[i, a]
                    rx = x[i, 0] - x[j, 0]
                    ry = x[i, 1] - x[j, 1]
                    rz = x[i, 2] - x[j, 2]
                    r = math.sqrt(rx * rx + ry * ry + rz * rz)
                    if r >= h or r < 1e-12:
                        continue
                    w = m / rho[j] * (c[j, k] - c[i, k])
                    ls += w * clap * (h - r)
                    q = w * csp * (h - r) * (h - r) / r
                    gx += q * rx
                    gy += q * ry
                    gz += q * rz
                gn = math.sqrt(gx * gx + gy * gy + gz * gz)
                if paper_ops:
                    # the paper's own SPH Laplacian (Eq. 7) carries 1/(rho_i rho_j); its gradient (A.4 form) m/rho_j
                    ls = ls / ri
                if curv_mode == 0:
                    d = om * abs(ls) / (gn + grad_eps)
                else:
                    d = om * (-ls) / (gn + grad_eps)
            if curv_mode != 2:
                if d > cap:
                    d = cap
                elif d < -cap:
                    d = -cap
            D[i, k] = d


@njit(parallel=True, cache=True)
def diffuse(x, rho, c, D, nbr, cnt, m, h, clap, pair, dt, out):
    """Eq. (7): dc_i/dt = D sum_j m_j (c_j - c_i)/(rho_i rho_j) lap W.  pair=1 uses (D_i + D_j)/2 inside the sum."""
    n = x.shape[0]
    K = c.shape[1]
    for i in prange(n):
        ri = rho[i]
        for k in range(K):
            s = 0.0
            for a in range(cnt[i]):
                j = nbr[i, a]
                rx = x[i, 0] - x[j, 0]
                ry = x[i, 1] - x[j, 1]
                rz = x[i, 2] - x[j, 2]
                r = math.sqrt(rx * rx + ry * ry + rz * rz)
                if r >= h:
                    continue
                w = m / (ri * rho[j]) * clap * (h - r) * (c[j, k] - c[i, k])
                if pair == 1:
                    w *= 0.5 * (D[i, k] + D[j, k])
                s += w
            if pair == 1:
                v = c[i, k] + dt * s
            else:
                v = c[i, k] + dt * D[i, k] * s
            out[i, k] = v if v > 0.0 else 0.0


# --------------------------------------------------------------------------- absorption, Eq. (9)-(10)
@njit(parallel=True, cache=True)
def absorption(x, lid, l, H, bnbr, bcnt, bpaper, rc_tab, gamma_tab, theta_tab, muv_tab,
               rho0, gmag, dt, contact_h, absorb_depth, use_ph, absorbed):
    """Lucas-Washburn (Eq. 9): dl/dt = (P_h + P_c)/(8 mu l) (r_c^2 + 4 r_c), P_c = 2 gamma cos(theta)/r_c (Eq. 10).
    Integrated exactly: l^2 <- l^2 + 2 K dt with K = (P_h + P_c)(r_c^2 + 4 r_c)/(8 mu).
    Only particles touching the paper absorb ('the absorption occurs only in the humidified area').
    A particle whose penetration length reaches absorb_depth is absorbed into the paper."""
    n = x.shape[0]
    for i in prange(n):
        absorbed[i] = False
        L = lid[i]
        rc = rc_tab[L]
        if rc <= 0.0:
            continue
        if x[i, 2] > contact_h:
            continue
        touching = False
        for a in range(bcnt[i]):
            if bpaper[bnbr[i, a]]:
                touching = True
                break
        if not touching:
            continue
        pc = 2.0 * gamma_tab[L] * math.cos(theta_tab[L]) / rc
        ph = rho0 * gmag * H[i] if use_ph else 0.0
        mu = muv_tab[L]
        if mu <= 0.0:
            absorbed[i] = True
            continue
        K = (ph + pc) * (rc * rc + 4.0 * rc) / (8.0 * mu)
        l2 = l[i] * l[i] + 2.0 * K * dt
        if l2 < 0.0:
            l2 = 0.0
        l[i] = math.sqrt(l2)
        if l[i] >= absorb_depth:
            absorbed[i] = True


@njit(cache=True)
def column_heights(x, d0, out):
    """Hydrostatic head H_i = (top of the liquid column above particle i) - z_i + d0 on a d0 x d0 column grid."""
    n = x.shape[0]
    if n == 0:
        return
    xmin = x[:, 0].min()
    ymin = x[:, 1].min()
    nx = int((x[:, 0].max() - xmin) / d0) + 1
    ny = int((x[:, 1].max() - ymin) / d0) + 1
    top = np.full(nx * ny, -1e30)
    ci = np.empty(n, np.int64)
    for i in range(n):
        ix = int((x[i, 0] - xmin) / d0)
        iy = int((x[i, 1] - ymin) / d0)
        c = ix * ny + iy
        ci[i] = c
        if x[i, 2] > top[c]:
            top[c] = x[i, 2]
    for i in range(n):
        out[i] = top[ci[i]] - x[i, 2] + d0


@njit(cache=True)
def deposit(xa, ca, va, bx, bnbr, bcnt, bpaper, h, cp6, area, frac, paper_c):
    """Pigment of absorbed particles stays on the paper (Sec. 4.3): c_i V_i spread over the touching paper particles
    with normalised poly6 weights, stored as a surface density (volume of pure pigment per area)."""
    h2 = h * h
    K = ca.shape[1]
    for t in range(xa.shape[0]):
        wsum = 0.0
        for a in range(bcnt[t]):
            k = bnbr[t, a]
            if not bpaper[k]:
                continue
            rx = xa[t, 0] - bx[k, 0]
            ry = xa[t, 1] - bx[k, 1]
            rz = xa[t, 2] - bx[k, 2]
            r2 = rx * rx + ry * ry + rz * rz
            if r2 < h2:
                d = h2 - r2
                wsum += cp6 * d * d * d
        if wsum <= 0.0:
            continue
        for a in range(bcnt[t]):
            k = bnbr[t, a]
            if not bpaper[k]:
                continue
            rx = xa[t, 0] - bx[k, 0]
            ry = xa[t, 1] - bx[k, 1]
            rz = xa[t, 2] - bx[k, 2]
            r2 = rx * rx + ry * ry + rz * rz
            if r2 < h2:
                d = h2 - r2
                w = cp6 * d * d * d / wsum
                for kk in range(K):
                    paper_c[k, kk] += frac * ca[t, kk] * va * w / area


@njit(parallel=True, cache=True)
def contact_stain(x, c, bx, bnbr, bcnt, bpaper, h, cp6, contact_h, stain, touch):
    """'We allow the paper to have pigment density when the liquid particles are in contact with the paper.'
    touch[k] = 1 marks humidified paper; stain[k] = running max of the contacting liquid concentration."""
    n = x.shape[0]
    h2 = h * h
    K = c.shape[1]
    for i in prange(n):
        if x[i, 2] > contact_h:
            continue
        for a in range(bcnt[i]):
            k = bnbr[i, a]
            if not bpaper[k]:
                continue
            rx = x[i, 0] - bx[k, 0]
            ry = x[i, 1] - bx[k, 1]
            r2 = rx * rx + ry * ry
            if r2 < 0.25 * h2:
                touch[k] = 1
                for kk in range(K):
                    if c[i, kk] > stain[k, kk]:
                        stain[k, kk] = c[i, kk]   # benign race: max of equal-ish values
