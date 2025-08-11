#!/usr/bin/env python3
"""
Batch LBLRTM Wrapper for Ozone Profiles from Extrapolated Sonde Data (with MIRTO emissivity)
Author: Paolo Antonelli <paoloa@adaptivemeteo.com>
"""
import argparse
import datetime
import logging
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset as NC

from lbl_util_lib import lbl_write_tape5, lbl_write_tape5_od, read_lbl_outputs
from lbl_emissivity_lib_mirto import EmissivityModel, create_hsremis_from_sec

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
LOG = logging.getLogger(__name__)

def rh_to_specific_humidity(p_hPa, T_K, RH_pct):
    es = 6.112 * np.exp((17.67 * (T_K - 273.15)) / (T_K - 29.65))  # hPa
    e = RH_pct / 100.0 * es
    w = 0.622 * e / (p_hPa - e)
    return 1000.0 * w  # g/kg

def load_profile(ds: xr.Dataset, index: int, temp_filename: str = None) -> dict:
    """
    Load atmospheric profile from an xarray Dataset and optionally save
    temperature profile (level index, pressure [hPa], temperature [K])
    to an ASCII file.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing atmospheric variables.
    index : int
        Profile index to extract.
    temp_filename : str, optional
        If provided, path to ASCII file where level index, pressure [hPa],
        and temperature [K] will be saved.

    Returns
    -------
    dict
        Dictionary with keys: pressure, tdry, w, oz, alt
    """
    import numpy as np

    # Extract variables from dataset
    p_hPa = ds["pressure_hPa"].isel(profile=index, level=slice(None)).values
    T_K = ds["temperature_K"].isel(profile=index, level=slice(None)).values
    RH = ds["relative_humidity_pct"].isel(profile=index, level=slice(None)).values
    o3_Pa = ds["ozone_partial_pressure_Pa"].isel(profile=index, level=slice(None)).values

    # Convert ozone to ppmv, with clipping
    pressure_Pa = p_hPa * 100.0
    ozone_ppmv = (o3_Pa / pressure_Pa) * 1e6
    ozone_ppmv[ozone_ppmv < 0.001] = 0.001

    # Altitude and specific humidity
    z = ds["height_m"].isel(profile=index, level=slice(None)).values
    w = rh_to_specific_humidity(p_hPa, T_K, RH)

    # Prepare output dictionary
    profile_dict = {"pressure": p_hPa, "tdry": T_K, "w": w, "oz": ozone_ppmv, "alt": z}

    # Optional: write ASCII temperature file
    if temp_filename is not None:
        levels = np.arange(1, len(T_K) + 1)  # Level numbering starts at 1
        ascii_data = np.column_stack((levels, p_hPa, T_K))
        np.savetxt(
            temp_filename,
            ascii_data,
            fmt="%d %.6f %.6f",
            header="level_index pressure_hPa temperature_K",
            comments=""
        )
        print(f"Temperature profile saved to {temp_filename}")

    return profile_dict

"""
def run_lblrtm(path,timeout_seconds):
    cmd = [
        f"{path}/lblrtm_v12.17_linux_gnu_dbl"
    ]
    subprocess.run(cmd, timeout=timeout_seconds, check=True)
"""
from pathlib import Path
import subprocess
import os

def run_lblrtm(work_dir, timeout_seconds, expect=("TAPE27","TAPE12")):
    """Run LBLRTM in work_dir; return stdout/stderr; assert expected outputs exist."""
    wd = Path(work_dir)

    # Sanity: inputs we expect for a radiance run
    must_exist = ["TAPE5", "TAPE3"]
    for m in must_exist:
        p = wd / m
        if not p.exists():
            raise FileNotFoundError(f"[run_lblrtm] Missing required input {m} at {p}")

    exe = wd / "lblrtm_v12.17_linux_gnu_dbl"
    if not exe.exists():
        raise FileNotFoundError(f"[run_lblrtm] LBLRTM exe not found at {exe}")
    # ensure executable bit
    exe.chmod(exe.stat().st_mode | 0o111)

    # Helpful: pin threads to avoid oversubscription weirdness
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")

    try:
        cp = subprocess.run(
            [str(exe)],
            cwd=str(wd),
            timeout=timeout_seconds,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        # Show everything to understand the STOP
        raise RuntimeError(
            f"[run_lblrtm] LBLRTM failed in {wd}\n"
            f"--- STDOUT ---\n{e.stdout}\n"
            f"--- STDERR ---\n{e.stderr}\n"
        ) from e

    # Verify outputs if requested
    missing = []
    for out in expect:
        if out and not (wd / out).exists():
            missing.append(out)

    if missing:
        # Show TAPE5 to understand flags
        tape5 = (wd / "TAPE5").read_text(errors="ignore") if (wd / "TAPE5").exists() else "<missing>"
        raise RuntimeError(
            f"[run_lblrtm] Completed but missing outputs {missing} in {wd}.\n"
            f"Likely TAPE5 does not request them (e.g., PL=1 for TAPE27). "
            f"Here are the first 50 lines of TAPE5:\n"
            f"{os.linesep.join(tape5.splitlines()[:50])}"
        )

    return cp.stdout, cp.stderr

def get_band_limits(instrument, band):
    if instrument == 2:  # IASI
        if band == 1:
            return 645.0, 1200.0
        elif band == 2:
            return 1200.0, 2000.0
        elif band == 3:
            return 2000.0, 2760.0
        else:
            raise ValueError(f"Unsupported IASI band {band}")
    elif instrument == 3:  # CrIS
        if band == 1:
            return 650.0, 1130.0
        elif band == 2:
            return 1210.0, 1750.0
        elif band == 3:
            return 2155.0, 2550.0
        else:
            raise ValueError(f"Unsupported CrIS band {band}")
    else:
        raise ValueError(f"Unsupported instrument {instrument}")

def worker(mode, ds, idx, input_nc, host_in, cont_in, host_out, cont_out,
           docker_image, instrument, docker_timeout, band,
           emissivity_nc, n_eig):

    seed = os.getpid()  # or a combination like os.getpid() + time.time() for more variability
    rng = np.random.default_rng(seed)

    prof_dir = f"profile_{idx:05d}"
    host_in_prof = host_in / prof_dir
    host_out_prof = host_out / prof_dir
    host_in_prof.mkdir(parents=True, exist_ok=True)
    host_out_prof.mkdir(parents=True, exist_ok=True)

    for fname in ["TAPE3", "FSCDXS", "lblrtm_v12.17_linux_gnu_dbl", "absco-ref_wv-mt-ckd.nc"]:
        src = host_in / fname
        if src.exists():
            shutil.copy(src, host_in_prof / fname)
    xs_src = host_in / "xs"
    if xs_src.is_dir():
        shutil.copytree(xs_src, host_in_prof / "xs", dirs_exist_ok=True)

    model = EmissivityModel(emissivity_nc, n_eig)
    #SEC = np.random.normal(0.0, 1.0, n_eig)
    SEC = rng.normal(0.0, np.sqrt(model.eigenvalues), size=n_eig)

    v1_band, v2_band = get_band_limits(instrument, band)
    band_range_input = [v1_band - 30.0, v2_band + 30.0]

    wn_emis_full = np.linspace(band_range_input[0], band_range_input[1], 500)
    #wn_emis_full = np.linspace(600.0, 1250.0, 500)
    coef, emis_arr = create_hsremis_from_sec(SEC, model, wn_emis_full)
    refl_arr = 1.0 - emis_arr

    with open(host_in_prof / "EMISSIVITY", "w") as f:
        step = wn_emis_full[1] - wn_emis_full[0]
        f.write(f"{wn_emis_full[0]:10.3f}{wn_emis_full[-1]:10.3f}{step:10.3f}{len(wn_emis_full):10d}\n")
        for v in emis_arr:
            f.write(f"{v:10.3f}\n")

    with open(host_in_prof / "REFLECTIVITY", "w") as f:
        f.write(f"{wn_emis_full[0]:10.3f}{wn_emis_full[-1]:10.3f}{step:10.3f}{len(wn_emis_full):10d}\n")
        for v in refl_arr:
            f.write(f"{v:10.3f}\n")

    prof = load_profile(ds, idx, temp_filename=f"{host_in_prof}/profile.txt")
    tape5 = host_in_prof / "TAPE5"
    mean_temp = ds["temperature_K"].isel(profile=idx).values[0]
    sfctemp = np.random.normal(loc=mean_temp, scale=3.0)
    #sfctemp = ds["temperature_K"].isel(profile=idx, level=1).item()
    pmin, pmax = prof["pressure"].min(), prof["pressure"].max()

    v1_band, v2_band = get_band_limits(instrument, band)
    band_range_input = [v1_band - 20.0, v2_band + 20.0]

    if mode == 'rad':

        lbl_write_tape5(
            prof,
            f"Profile {idx}",
            band_range_input,
            [sfctemp, 1.0],
            [pmin, pmax, 180.0],
            1,
            np.full_like(prof["pressure"], 420.0),
            1,
            band_range_input,
            0,
            instrument,
            tape5
        )
        run_lblrtm(host_in_prof, timeout_seconds=docker_timeout)
        print(f"run radiance profile: {idx}")

    elif mode == 'od':

        lbl_write_tape5_od(
            prof,
            f"Profile {idx}",
            band_range_input,
            [sfctemp, 1.0],
            [pmin, pmax, 180.0],
            1,
            np.full_like(prof["pressure"], 420.0),
            1,
            band_range_input,
            0,
            instrument,
            tape5
        )
        run_lblrtm(host_in_prof, timeout_seconds=docker_timeout)
        print(f"run od profile: {idx}")

    elif mode == 'all':

        lbl_write_tape5(
            prof,
            f"Profile {idx}",
            band_range_input,
            [sfctemp, 1.0],
            [pmin, pmax, 180.0],
            1,
            np.full_like(prof["pressure"], 420.0),
            1,
            band_range_input,
            0,
            instrument,
            tape5
        )
        run_lblrtm(host_in_prof, timeout_seconds=docker_timeout)
        print(f"run radiance profile: {idx}")

        lbl_write_tape5_od(
            prof,
            f"Profile {idx}",
            band_range_input,
            [sfctemp, 1.0],
            [pmin, pmax, 180.0],
            1,
            np.full_like(prof["pressure"], 420.0),
            1,
            band_range_input,
            0,
            instrument,
            tape5
        )
        run_lblrtm(host_in_prof, timeout_seconds=docker_timeout)
        print(f"run od profile: {idx}")

    wn_rad, rad, inst_str = read_lbl_outputs(instrument, host_out_prof)
    mask = (wn_rad >= v1_band) & (wn_rad <= v2_band)
    wn_rad = wn_rad[mask]
    rad = rad[mask]

    #shutil.rmtree(host_in_prof)
    #shutil.rmtree(host_out_prof)

    return (idx, wn_rad, rad, inst_str, emis_arr, wn_emis_full, coef, sfctemp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_nc", type=Path, required=True)
    parser.add_argument("--output_nc", type=Path, required=True)
    parser.add_argument("--docker_image", type=str, default="aurora_lblrtm:latest")
    parser.add_argument("--instrument", type=int, choices=[2,3], default=2)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--band", type=int, choices=[1,2,3], required=True)
    parser.add_argument("--host_in", type=Path, required=True)
    parser.add_argument("--cont_in", type=str, required=True)
    parser.add_argument("--host_out", type=Path, required=True)
    parser.add_argument("--cont_out", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--docker_timeout", type=int, default=240)
    parser.add_argument("--emissivity_nc", type=str, required=True)
    parser.add_argument("--n_eig", type=int, required=True)
    parser.add_argument("--no_overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_observations", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    args.host_in.mkdir(parents=True, exist_ok=True)
    args.host_out.mkdir(parents=True, exist_ok=True)
    args.output_nc.parent.mkdir(parents=True, exist_ok=True)

    mode = args.mode

    ds_in = xr.open_dataset(args.input_nc)
    total = ds_in.sizes["profile"]
    start = args.start_index
    end = min(start + (args.max_observations or total - start), total)
    LOG.info("Processing profiles %d to %d of %d", start, end, total)

    if args.overwrite and args.output_nc.exists():
        args.output_nc.unlink()

    first_write = True
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                worker, mode, ds_in, idx,
                args.input_nc,
                args.host_in, args.cont_in,
                args.host_out, args.cont_out,
                args.docker_image,
                args.instrument,
                args.docker_timeout,
                args.band,
                args.emissivity_nc,
                args.n_eig
            ): idx for idx in range(start, end)
        }

        for future in as_completed(futures):
            try:
                res = future.result()
            except Exception as e:
                LOG.error("Worker exception: %s", e)
                continue

            idx, wn_rad, rad, inst_str, emis_arr, wn_emis, coef, skt = res

            LOG.info("Profile %d finished (instrument=%s)", idx, inst_str)
            if first_write:
                nc = NC(args.output_nc, "w", format="NETCDF4")
                nc.createDimension("profile", total)
                nc.createDimension("wn", len(wn_rad))
                nc.createDimension("wn_emis", len(wn_emis))
                nc.createDimension("n_eig", len(coef))
                nc.createVariable("wn", "f4", ("wn",))[:] = wn_rad
                nc.createVariable("wn_emis", "f4", ("wn_emis",))[:] = wn_emis
                nc.createVariable("rad", "f4", ("profile", "wn"), zlib=True)
                nc.createVariable("emis", "f4", ("profile", "wn_emis"), zlib=True)
                nc.createVariable("coef", "f4", ("profile", "n_eig"), zlib=True)
                nc.createVariable("profile_number", "i4", ("profile",))
                nc.createVariable("skt", "f4", ("profile",))
                nc.setncattr("creation_date", datetime.datetime.utcnow().isoformat())
                nc.setncattr("author", "Paolo Antonelli <paoloa@adaptivemeteo.com>")
                nc.setncattr("profile_file", args.input_nc.name)
                nc.setncattr("instrument_str", inst_str)
                first_write = False
            else:
                nc = NC(args.output_nc, "a")

            nc.variables["rad"][idx, :] = rad
            nc.variables["emis"][idx, :] = emis_arr
            nc.variables["coef"][idx, :] = coef
            nc.variables["skt"][idx] = skt
            nc.variables["profile_number"][idx] = idx
            nc.close()

    ds_in.close()
    LOG.info("All done. Output NetCDF: %s", args.output_nc)

if __name__ == "__main__":
    main()
