import numpy as np
import glob
import xarray as xr
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
    """Planck function in mW/(m²·sr·cm⁻¹)."""
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
    """Vectorized TOA upwelling radiance from OD layers (mW m^-2 sr^-1 (cm^-1)^-1).

    Returns
    -------
    I_toa : (M,) array
    components : dict with keys 'surface','atm_up','reflected','down_sfc','T_all'
    """
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

    tau = np.exp(-od)                          # [N, M]
    one_minus_tau = 1.0 - tau                  # [N, M]

    tau_above = np.ones_like(tau)
    if N > 1:
        tau_above[:-1] = np.flip(np.cumprod(np.flip(tau[1:], axis=0), axis=0), axis=0)
    tau_below = np.ones_like(tau)
    if N > 1:
        tau_below[1:] = np.cumprod(tau[:-1], axis=0)
    T_all = np.cumprod(tau, axis=0)[-1]

    B_layers = planck_radiance(wn, Tlay)      # [M, N] or [M]
    if B_layers.ndim == 1:
        B_layers = np.tile(B_layers[None, :], (N, 1))
    else:
        B_layers = B_layers.T                 # -> [N, M]
    B_surf = planck_radiance(wn, float(surf_temp))  # [M]

    I_atm_up = np.sum(B_layers * one_minus_tau * tau_above, axis=0)
    I_surf = emis * B_surf * T_all
    I_dn_sfc = np.sum(B_layers * one_minus_tau * tau_below, axis=0)
    I_refl = (1.0 - emis) * I_dn_sfc * T_all if include_reflection else np.zeros_like(I_dn_sfc)

    I_toa = I_surf + I_atm_up + I_refl
    comps = {"surface": I_surf, "atm_up": I_atm_up, "reflected": I_refl, "down_sfc": I_dn_sfc, "T_all": T_all}
    return I_toa, comps


def main():
    od_pattern = "ODint_*"
    tape7_file = "TAPE7"
    emissivity_file = "EMISSIVITY"
    surf_temp = 300.0

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

if __name__ == "__main__":
    main()

