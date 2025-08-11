import numpy as np
import glob
import xarray as xr
import matplotlib.pyplot as plt
import RC_utils as RC

def read_od_files(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern}")
    od_list = []
    wn_ref = None
    for f in files:
        wn, od = RC.readBinary(f)
        if wn_ref is None:
            wn_ref = wn
        elif not np.allclose(wn, wn_ref):
            raise ValueError(f"Wavenumber grid mismatch in {f}")
        od_list.append(np.asarray(od, dtype=float))
    return np.asarray(wn_ref, dtype=float), np.vstack(od_list)

def read_tape7_layers(filename):
    pressures = []
    temps = []
    with open(filename, 'r') as f:
        lines = f.readlines()[2:202]
    for i, line in enumerate(lines):
        if i % 2 == 0:
            parts = line.split()
            pressures.append(float(parts[0]))
            temps.append(float(parts[1]))
    return np.array(pressures), np.array(temps)

def read_emissivity(filename, wn_target):
    with open(filename, "r") as f:
        header = f.readline().split()
        wn_start = float(header[0])
        wn_end = float(header[1])
        n_points = int(header[3])
        emis_vals = [float(line.strip()) for line in f if line.strip()]
    emis_wn = np.linspace(wn_start, wn_end, n_points)
    emis_interp = np.interp(wn_target, emis_wn, emis_vals)
    return emis_interp

def planck_radiance(nu_cm_inv, T_K):
    nu_m_inv = np.asarray(nu_cm_inv) * 100.0
    h = 6.62607015e-34
    c = 2.99792458e8
    k = 1.380649e-23
    nu_m_inv = np.atleast_1d(nu_m_inv)
    T_K = np.atleast_1d(T_K)
    nom = 2 * h * c**2 * nu_m_inv[:, None]**3
    expo = (h * c * nu_m_inv[:, None]) / (k * T_K[None, :])
    denom = np.exp(expo) - 1.0
    return np.squeeze((nom / denom) * 1e5)

def compute_toa_radiance(wn, od_layers, layer_temps, surf_temp, emissivity, include_reflection=True):
    wn = np.asarray(wn, dtype=float)
    od = np.asarray(od_layers, dtype=float)
    Tlay = np.asarray(layer_temps, dtype=float)
    N, M = od.shape
    assert wn.shape[0] == M, "wn and OD spectral length mismatch"
    assert Tlay.shape[0] == N, "layer_temps and OD layer count mismatch"

    emis = np.asarray(emissivity, dtype=float)
    if emis.ndim == 0:
        emis = np.full(M, float(emis))
    else:
        assert emis.shape[0] == M, "emissivity must be scalar or length M"

    tau = np.exp(-od)
    one_minus_tau = 1.0 - tau

    tau_above = np.ones_like(tau)
    if N > 1:
        tau_above[:-1] = np.flip(np.cumprod(np.flip(tau[1:], axis=0), axis=0), axis=0)
    tau_below = np.ones_like(tau)
    if N > 1:
        tau_below[1:] = np.cumprod(tau[:-1], axis=0)
    T_all = np.cumprod(tau, axis=0)[-1]

    B_layers = planck_radiance(wn, Tlay)
    if B_layers.ndim == 1:
        B_layers = np.tile(B_layers[None, :], (N, 1))
    else:
        B_layers = B_layers.T
    B_surf = planck_radiance(wn, float(surf_temp))

    I_atm_up = np.sum(B_layers * one_minus_tau * tau_above, axis=0)
    I_surf = emis * B_surf * T_all
    I_dn_sfc = np.sum(B_layers * one_minus_tau * tau_below, axis=0)
    I_refl = (1.0 - emis) * I_dn_sfc * T_all if include_reflection else np.zeros_like(I_dn_sfc)

    I_toa = I_surf + I_atm_up + I_refl
    comps = {"surface": I_surf, "atm_up": I_atm_up, "reflected": I_refl, "down_sfc": I_dn_sfc, "T_all": T_all}
    return I_toa, comps

# -----------------------------
# IASI FFT-style spectral convolution (sinc^2 ILS from 1-sided OPD Lcut)
# -----------------------------

def iasi_convolve(wn_hi, rad_hi,
                  v1=605.0, v2=2829.9975,
                  L1=2.0, vlaser=15780.0, vsf=2, Lcut=None,
                  window_half_width=None):
    """Convolve high-res spectrum to IASI channel grid using analytic ILS.

    Implements the IASI interferometer ILS for a single-band, one-sided (0..Lcut)
    interferogram saved length. For a boxcar OPD of length Lcut, the ILS is
        ILS(Δν) = sinc^2(2 * Lcut * Δν),
    where numpy.sinc(x) = sin(pi x)/(pi x).

    Parameters
    ----------
    wn_hi : (M,) array
        High-resolution wavenumber grid (cm^-1), ascending.
    rad_hi : (M,) array
        High-resolution radiance on wn_hi.
    v1, v2 : float
        Band limits (cm^-1).
    L1 : float
        Longest OPD path (cm); IASI channel spacing dvc = 1/(2*L1) = 0.25 cm^-1 for L1=2.
    vlaser : float
        Reference laser wavenumber (cm^-1), used only to align channels to integer multiples if desired.
    vsf : int
        Laser scaling factor (kept for completeness; not used in the simple grid construction below).
    Lcut : float or None
        Saved OPD length (cm). If None, defaults to L1.
    window_half_width : float or None
        Optional local window (cm^-1) around each channel over which to evaluate the ILS.
        If None, uses 6/Lcut (about ±6 mainlobe widths).

    Returns
    -------
    vc : (C,) array
        IASI channel center wavenumbers (cm^-1).
    rad_conv : (C,) array
        Convolved radiance at IASI resolution.
    """
    wn_hi = np.asarray(wn_hi, dtype=float)
    rad_hi = np.asarray(rad_hi, dtype=float)
    assert wn_hi.ndim == 1 and rad_hi.ndim == 1 and wn_hi.size == rad_hi.size

    if Lcut is None:
        Lcut = float(L1)
    dvc = 1.0 / (2.0 * L1)  # 0.25 cm^-1 for L1=2

    # Build channel centers aligned to dvc grid within [v1, v2]
    k_start = int(np.ceil((v1) / dvc))
    k_end = int(np.floor((v2) / dvc))
    vc = (np.arange(k_start, k_end + 1, dtype=float) * dvc)

    # Choose evaluation window for kernel support
    if window_half_width is None:
        window_half_width = 6.0 / Lcut  # ~±6 mainlobes

    # Precompute masks for local neighborhoods per channel
    rad_conv = np.empty_like(vc)

    # For efficiency, keep a running index into wn_hi
    j0 = 0
    for i, vcen in enumerate(vc):
        left = vcen - window_half_width
        right = vcen + window_half_width
        # Move j0 to the first index >= left
        while j0 < wn_hi.size and wn_hi[j0] < left:
            j0 += 1
        j1 = j0
        while j1 < wn_hi.size and wn_hi[j1] <= right:
            j1 += 1
        if j0 >= j1:
            rad_conv[i] = np.nan
            continue
        nu_seg = wn_hi[j0:j1]
        rad_seg = rad_hi[j0:j1]
        dnu = np.diff(nu_seg)
        # Build kernel on the segment; Δν = nu - vcen
        dnu_off = nu_seg - vcen
        ils = np.sinc(2.0 * Lcut * dnu_off)**2
        # Normalize kernel area using trapezoid weights
        w = np.ones_like(nu_seg)
        if w.size > 1:
            w[0] = 0.5
            w[-1] = 0.5
        num = np.sum(rad_seg * ils * w * np.r_[np.diff(nu_seg), 0][0:1] + 0)  # placeholder
        # Use trapezoidal rule explicitly for stability
        num = np.trapz(rad_seg * ils, nu_seg)
        den = np.trapz(ils, nu_seg)
        rad_conv[i] = num / den if den > 0 else np.nan

    return vc, rad_conv

def plot_toa_radiance(wn, toa_radiance, wn_1, toa_radiance_1, namestr = 'output'):
    plt.figure(figsize=(10, 5))
    plt.plot(wn, toa_radiance, label='TOA OD Radiance', linewidth=0.6)
    plt.plot(wn_1, toa_radiance_1, label='LBL Radiance', color='red', linewidth=0.3)
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Radiance (mW m$^{-2}$ sr$^{-1}$ cm)')
    plt.title('Top-of-Atmosphere Radiance Spectrum')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{namestr}.png", dpi=300)

def read_monochrmatic_rad(filename = "TAPE12"):
    import RC_utils as RC
    wn, rad = RC.readBinary(filename)  # layer 1

    return wn, rad*10**7

def main():
    od_pattern = "ODint_*"
    tape7_file = "TAPE7"
    emissivity_file = "EMISSIVITY"
    surf_temp = 285.577

    wn, od_layers = read_od_files(od_pattern)
    pressures, layer_temps = read_tape7_layers(tape7_file)
    if layer_temps.size != od_layers.shape[0]:
        raise ValueError(f"Layer mismatch: OD has {od_layers.shape[0]}, TAPE7 has {layer_temps.size}")

    emis = read_emissivity(emissivity_file, wn)
    toa, comps = compute_toa_radiance(wn, od_layers, layer_temps, surf_temp, emis, include_reflection=True)

    ds = xr.Dataset(
        {
            "TOA_radiance": ("wavenumber", toa),
            "emissivity": ("wavenumber", emis),
            "surface": ("wavenumber", comps["surface"]),
            "atm_up": ("wavenumber", comps["atm_up"]),
            "reflected": ("wavenumber", comps["reflected"]),
            "down_sfc": ("wavenumber", comps["down_sfc"]),
            "T_all": ("wavenumber", comps["T_all"]),
        },
        coords={"wavenumber": wn},
    )
    ds = ds.assign_coords(layer=np.arange(od_layers.shape[0]))
    ds["layer_temperature"] = ("layer", layer_temps)

    ds.to_netcdf("toa_radiance.nc")

    mono_wn, mono_rad = read_monochrmatic_rad()

    namestr = 'toa_radiance_comparison_full_resolution'
    plot_toa_radiance(wn, toa, mono_wn, mono_rad, namestr)

    wn_od_iasi, rad_od_iasi = iasi_convolve(wn, toa, v1=645.0, v2=1210.0, L1=2.0, vlaser=15780.0, vsf=2, Lcut=2.0)
    wn_lbl_iasi, rad_lbl_iasi = iasi_convolve(mono_wn, mono_rad, v1=645.0, v2=1210.0, L1=2.0, vlaser=15780.0, vsf=2, Lcut=2.0)

    namestr = 'toa_radiance_comparison_iasi_resolution'
    plot_toa_radiance(wn_od_iasi, rad_od_iasi, wn_lbl_iasi, rad_lbl_iasi, namestr)



if __name__ == "__main__":
    main()
