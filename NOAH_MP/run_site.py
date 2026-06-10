#!/usr/bin/env python3
"""
Full NOAH-MP pipeline for one causal discovery site.

Steps:
  1. Generate LDASIN from ERA5 forcing
  2. Create hrldas_setup.nc with site-specific parameters
  3. Run hrldas.exe (2020-2021, ~6 min)
  4. Extract observed + hidden CSVs from LDASOUT + ERA5

Usage:
    python run_site.py --site_id 1
    python run_site.py --site_id 1 --skip_ldasin      # reuse existing LDASIN
    python run_site.py --site_id 1 --skip_noahmp      # skip run, just extract
    python run_site.py --site_id 0                    # Madagascar: reuse existing LDASOUT
"""

import argparse
import os
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import netCDF4 as nc4
import xarray as xr
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ESM_DIR    = Path("/Net/Groups/BGI/scratch/npapan/esm_models")
NOAHMP_RUN = ESM_DIR / "noahmp" / "run"
ERA5_DIR   = Path("/Net/Groups/data_BGC/era5_land/e1/0d10_hourly")
SITES_DIR  = Path("/Net/Groups/BGI/work_1/scratch/npapan/LLM_causal_discovery/NOAH_MP/sites")

YEARS      = [2020, 2021]

# ── Site configurations ────────────────────────────────────────────────────────
# ivgtyp: MODIS IGBP land cover  isltyp: soil texture class
# hgt: elevation (m)  tmn: annual mean deep soil temp (K)
# shdmax/shdmin: max/min vegetation fraction (%)
# lai: initial LAI  smois: soil moisture per layer (m3/m3)
# tslb: soil temperature per layer (K), surface to deep
SITES = {
    0: dict(name="Madagascar",   lat=-17.75, lon=44.25,   ivgtyp=8,  isltyp=6, hgt=82,  tmn=297, shdmax=70, shdmin=20, lai=1.5, smois=[0.28, 0.30, 0.30, 0.29], tslb=[298.5, 298.0, 297.5, 297.0]),
    1: dict(name="Amazon",       lat=-3.0,   lon=-60.0,   ivgtyp=2,  isltyp=8, hgt=100, tmn=300, shdmax=95, shdmin=90, lai=5.0, smois=[0.35, 0.35, 0.35, 0.35], tslb=[299.0, 299.0, 299.0, 300.0]),
    2: dict(name="Congo",        lat=-1.5,   lon=23.5,    ivgtyp=2,  isltyp=8, hgt=400, tmn=297, shdmax=95, shdmin=90, lai=5.0, smois=[0.35, 0.36, 0.36, 0.35], tslb=[298.0, 297.0, 297.0, 297.0]),
    3: dict(name="Sahel",        lat=13.5,   lon=2.5,     ivgtyp=9,  isltyp=3, hgt=250, tmn=303, shdmax=60, shdmin=5,  lai=0.5, smois=[0.10, 0.12, 0.12, 0.13], tslb=[305.0, 303.0, 302.0, 301.0]),
    4: dict(name="Germany",      lat=51.0,   lon=13.0,    ivgtyp=5,  isltyp=6, hgt=200, tmn=282, shdmax=80, shdmin=20, lai=2.0, smois=[0.30, 0.32, 0.32, 0.31], tslb=[283.0, 283.0, 284.0, 285.0]),
    5: dict(name="BoealCanada",  lat=55.0,   lon=-107.0,  ivgtyp=1,  isltyp=3, hgt=500, tmn=275, shdmax=80, shdmin=70, lai=2.5, smois=[0.25, 0.27, 0.27, 0.26], tslb=[276.0, 276.0, 276.0, 276.0]),
    6: dict(name="GreatPlains",  lat=38.5,   lon=-98.5,   ivgtyp=10, isltyp=6, hgt=600, tmn=286, shdmax=70, shdmin=10, lai=1.0, smois=[0.25, 0.27, 0.27, 0.26], tslb=[287.0, 287.0, 287.0, 287.0]),
    7: dict(name="SpainMed",     lat=37.5,   lon=-4.5,    ivgtyp=7,  isltyp=7, hgt=700, tmn=289, shdmax=50, shdmin=15, lai=0.8, smois=[0.18, 0.20, 0.20, 0.19], tslb=[290.0, 290.0, 290.0, 290.0]),
    8: dict(name="IndiaDeccan",  lat=18.0,   lon=76.0,    ivgtyp=9,  isltyp=9, hgt=550, tmn=298, shdmax=65, shdmin=10, lai=1.0, smois=[0.22, 0.24, 0.24, 0.23], tslb=[299.0, 298.0, 298.0, 298.0]),
    9: dict(name="Australia",    lat=-25.0,  lon=133.0,   ivgtyp=7,  isltyp=3, hgt=300, tmn=296, shdmax=30, shdmin=5,  lai=0.5, smois=[0.08, 0.10, 0.10, 0.09], tslb=[297.0, 296.0, 296.0, 296.0]),
}

# ── GCC/NetCDF environment for hrldas.exe ─────────────────────────────────────
_GCC11    = "/opt/ohpc/pub/apps/BGC-easybuilded/GCCcore/11.2.0"
_BINUTILS = "/opt/ohpc/pub/apps/BGC-easybuilded/binutils/2.37-GCCcore-11.2.0"
_NCDIR    = "/opt/ohpc/pub/apps/BGC-easybuilded/netCDF/4.8.1-gompi-2021b"
_NCFDIR   = "/opt/ohpc/pub/apps/BGC-easybuilded/netCDF-Fortran/4.5.3-gompi-2021b"
_HDF5DIR  = "/opt/ohpc/pub/apps/BGC-easybuilded/HDF5/1.12.1-gompi-2021b"
_HRLDAS_ENV = {
    **os.environ,
    "PATH":             f"{_BINUTILS}/bin:{_GCC11}/bin:" + os.environ.get("PATH", ""),
    "LD_LIBRARY_PATH":  (f"{_GCC11}/lib64:{_GCC11}/lib:{_NCFDIR}/lib:{_NCDIR}/lib:{_HDF5DIR}/lib:"
                         + os.environ.get("LD_LIBRARY_PATH", "")),
}


# ══════════════════════════════════════════════════════════════════════════════
# ERA5 helpers
# ══════════════════════════════════════════════════════════════════════════════

def _deaccumulate(vals: np.ndarray) -> np.ndarray:
    """ERA5 accumulated J/m² or m → per-hour → /3600 → W/m² or m/s."""
    out = np.empty(len(vals), dtype=np.float64)
    out[0] = float(vals[1]) if len(vals) > 1 else float(vals[0])
    for i in range(1, len(vals)):
        d = float(vals[i]) - float(vals[i - 1])
        out[i] = float(vals[i]) if d < 0 else d
    return out / 3600.0


def _q_from_t_rh(t_K: float, rh_pct: float, p_Pa: float) -> float:
    """Specific humidity [kg/kg] from T [K], RH [%], pressure [Pa]."""
    T_C = t_K - 273.15
    e_s = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))
    e   = (rh_pct / 100.0) * e_s
    return float(np.clip(0.622 * e / (p_Pa - 0.378 * e), 0.0, 0.05))


def load_era5_point(varname: str, lat: float, lon: float,
                    years=YEARS, deacc: bool = False) -> pd.Series:
    """
    Load hourly ERA5 time series at the nearest grid point.
    Uses netCDF4 direct index slicing — avoids loading the full 39GB array.
    Handles 0-360 longitude and both 'time'/'valid_time' dimension names.
    """
    era5_lon = lon % 360   # ERA5 uses 0-360
    frames = []
    for year in years:
        vdir = ERA5_DIR / varname / str(year)
        files = sorted(vdir.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"No ERA5 files for {varname} {year} in {vdir}")
        for f in files:
            with nc4.Dataset(str(f)) as ds:
                lats = ds.variables["latitude"][:]
                lons = ds.variables["longitude"][:]
                ilat = int(np.abs(lats - lat).argmin())
                ilon = int(np.abs(lons - era5_lon).argmin())

                time_var = "valid_time" if "valid_time" in ds.variables else "time"
                tvar     = ds.variables[time_var]
                times    = nc4.num2date(tvar[:], tvar.units, only_use_cftime_datetimes=False)
                times_pd = pd.to_datetime([str(t) for t in times])

                vkey = [v for v in ds.variables
                        if v not in ("latitude", "longitude", "time", "valid_time",
                                     "expver", "number")][0]
                vals = ds.variables[vkey][:, ilat, ilon].astype(np.float64)

            frames.append(pd.Series(vals, index=times_pd))

    series = pd.concat(frames).sort_index()
    series = series[~series.index.duplicated()]
    if deacc:
        series = pd.Series(_deaccumulate(series.values), index=series.index)
    return series


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Generate LDASIN
# ══════════════════════════════════════════════════════════════════════════════

def generate_ldasin(site: dict, ldasin_dir: Path):
    lat, lon = site["lat"], site["lon"]
    ldasin_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading ERA5 forcing (lat={lat}, lon={lon}) ...", flush=True)
    t2m    = load_era5_point("t2m",   lat, lon)
    rh     = load_era5_point("rH_cf", lat, lon)           # in % (0-100)
    psfc   = load_era5_point("sp",    lat, lon)
    ws10   = load_era5_point("ws10",  lat, lon)
    tp     = load_era5_point("tp",    lat, lon, deacc=True)    # → m/s
    lwdown = load_era5_point("strd",  lat, lon, deacc=True)    # → W/m²
    swdown = load_era5_point("ssrd",  lat, lon, deacc=True)    # → W/m²

    # Save forcing parquet — avoids re-reading ERA5 during extraction step
    forcing_df = pd.DataFrame({
        "timestamp": t2m.index,
        "Tair": t2m.values,
        "Ur":   ws10.values,
    })
    forcing_df.to_parquet(ldasin_dir.parent / "forcing.parquet", index=False)  # = site_dir/forcing.parquet
    print(f"  Forcing parquet saved.", flush=True)

    idx = t2m.index
    n   = len(idx)
    print(f"  Writing {n} LDASIN files to {ldasin_dir} ...", flush=True)

    for i, dt in enumerate(idx):
        if i % 3000 == 0:
            print(f"    {dt.date()}  ({i}/{n})", flush=True)

        fname = ldasin_dir / f"{dt.strftime('%Y%m%d%H')}.LDASIN_DOMAIN1"
        if fname.exists():
            continue

        _t   = float(t2m.get(dt, 290.0))
        _rh  = float(rh.get(dt, 70.0))          # %
        _p   = float(psfc.get(dt, 101325.0))
        _ws  = max(0.0, float(ws10.get(dt, 1.0)))
        _rain = max(0.0, float(tp.get(dt, 0.0)) * 1000.0)   # m/s → kg/m²/s
        _sw  = max(0.0, float(swdown.get(dt, 0.0)))
        _lw  = max(0.0, float(lwdown.get(dt, 350.0)))
        _q   = _q_from_t_rh(_t, _rh, _p)

        _write_ldasin_file(fname, _t, _q, _p, _ws, _rain, _sw, _lw,
                           dt.to_pydatetime().replace(tzinfo=None))

    print(f"  LDASIN done ({n} files).", flush=True)


def _write_ldasin_file(path, t2d, q2d, psfc, u2d, rain, swdown, lwdown, dt):
    with nc4.Dataset(str(path), "w", format="NETCDF4") as f:
        f.createDimension("west_east",   1)
        f.createDimension("south_north", 1)
        f.START_DATE = dt.strftime("%Y-%m-%d_%H:%M:%S")
        for name, val, units in [
            ("T2D",      t2d,    "K"),
            ("Q2D",      q2d,    "kg kg-1"),
            ("PSFC",     psfc,   "Pa"),
            ("U2D",      u2d,    "m s-1"),
            ("V2D",      0.0,    "m s-1"),
            ("RAINRATE", rain,   "kg m-2 s-1"),
            ("SWDOWN",   swdown, "W m-2"),
            ("LWDOWN",   lwdown, "W m-2"),
        ]:
            v = f.createVariable(name, "f4", ("south_north", "west_east"))
            v.units = units
            v[:] = np.float32(val)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Create hrldas_setup.nc
# ══════════════════════════════════════════════════════════════════════════════

def create_setup_nc(site: dict, out_path: Path):
    lat, lon = site["lat"], site["lon"]
    dx = 0.1

    with nc4.Dataset(str(out_path), "w", format="NETCDF4") as f:
        for dim, sz in [("south_north", 1), ("west_east", 1),
                        ("south_north_stag", 2), ("west_east_stag", 2),
                        ("soil_layers_stag", 4), ("Time", 1)]:
            f.createDimension(dim, sz)

        f.TITLE       = f"HRLDAS setup: {site['name']}  lat={lat}  lon={lon}"
        f.DX          = np.float32(dx * 111000)
        f.DY          = np.float32(dx * 111000)
        f.TRUELAT1    = np.float32(lat)
        f.TRUELAT2    = np.float32(lat)
        f.STAND_LON   = np.float32(lon)
        f.MAP_PROJ    = 1
        f.MOAD_CEN_LAT = np.float32(lat)
        f.GRID_ID     = 1
        f.grid_id     = 1
        f.ISWATER     = 17
        f.ISLAKE      = 21
        f.ISURBAN     = 13
        f.ISICE       = 15
        f.MMINLU      = "MODIFIED_IGBP_MODIS_NOAH"

        def scalar(name, val, units=""):
            v = f.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            v.units = units; v[:] = np.float32(val)

        def scalar_int(name, val):
            v = f.createVariable(name, "i4", ("Time", "south_north", "west_east"))
            v[:] = int(val)

        lat_v = f.createVariable("XLAT",  "f4", ("Time", "south_north", "west_east"))
        lon_v = f.createVariable("XLONG", "f4", ("Time", "south_north", "west_east"))
        lat_v[:] = np.float32(lat);  lon_v[:] = np.float32(lon)

        scalar("HGT",       site["hgt"],     "m")
        scalar("TMN",       site["tmn"],     "K")
        scalar("TSK",       site["tslb"][0], "K")
        scalar("SEAICE",    0.0)
        scalar("XLAND",     1.0)
        scalar("MAPFAC_MX", 1.0)
        scalar("MAPFAC_MY", 1.0)
        scalar("SHDMAX",    site["shdmax"],  "%")
        scalar("SHDMIN",    site["shdmin"],  "%")
        scalar("LAI",       site["lai"],     "m2/m2")
        scalar("SNOW",      0.0,             "mm")
        scalar("CANWAT",    0.0,             "mm")
        scalar_int("IVGTYP", site["ivgtyp"])
        scalar_int("ISLTYP", site["isltyp"])

        dzs_v = f.createVariable("DZS", "f4", ("soil_layers_stag",))
        zs_v  = f.createVariable("ZS",  "f4", ("soil_layers_stag",))
        dzs_v[:] = np.float32([0.10, 0.30, 0.60, 1.00])
        zs_v[:]  = np.float32([0.05, 0.25, 0.70, 1.50])

        tslb_v  = f.createVariable("TSLB",  "f4", ("Time", "soil_layers_stag", "south_north", "west_east"))
        smois_v = f.createVariable("SMOIS", "f4", ("Time", "soil_layers_stag", "south_north", "west_east"))
        tslb_v.units  = "K";     smois_v.units = "m3/m3"
        for k in range(4):
            tslb_v[0, k, 0, 0]  = site["tslb"][k]
            smois_v[0, k, 0, 0] = site["smois"][k]

    print(f"  Setup NC written: {out_path}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Run NOAH-MP
# ══════════════════════════════════════════════════════════════════════════════

def run_noahmp(site_dir: Path, ldasin_dir: Path, setup_nc: Path, ldasout_dir: Path):
    run_dir = site_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ldasout_dir.mkdir(parents=True, exist_ok=True)

    for name in ["hrldas.exe", "NoahmpTable.TBL", "URBPARM.TBL",
                 "snicar_drdt_bst_fit_60_c070416.nc",
                 "snicar_optics_480bnd_c012422.nc",
                 "snicar_optics_5bnd_c013122.nc"]:
        src = NOAHMP_RUN / name
        dst = run_dir / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    nml = f"""&NOAHLSM_OFFLINE

 HRLDAS_SETUP_FILE = "{setup_nc}"
 INDIR             = "{ldasin_dir}/"
 OUTDIR            = "{ldasout_dir}/"

 START_YEAR  = 2020
 START_MONTH = 01
 START_DAY   = 01
 START_HOUR  = 00
 START_MIN   = 00
 KDAY        = 732

 FORCING_NAME_T  = "T2D"
 FORCING_NAME_Q  = "Q2D"
 FORCING_NAME_U  = "U2D"
 FORCING_NAME_V  = "V2D"
 FORCING_NAME_P  = "PSFC"
 FORCING_NAME_LW = "LWDOWN"
 FORCING_NAME_SW = "SWDOWN"
 FORCING_NAME_PR = "RAINRATE"

 DYNAMIC_VEG_OPTION                = 4
 CANOPY_STOMATAL_RESISTANCE_OPTION = 1
 BTR_OPTION                        = 1
 SURFACE_RUNOFF_OPTION             = 3
 SUBSURFACE_RUNOFF_OPTION          = 3
 DVIC_INFILTRATION_OPTION          = 1
 SURFACE_DRAG_OPTION               = 1
 FROZEN_SOIL_OPTION                = 1
 SUPERCOOLED_WATER_OPTION          = 1
 RADIATIVE_TRANSFER_OPTION         = 3
 SNOW_COMPACTION_OPTION            = 2
 SNOW_ALBEDO_OPTION                = 1
 SNOW_COVER_OPTION                 = 1
 PCP_PARTITION_OPTION              = 1
 SNOW_THERMAL_CONDUCTIVITY         = 1
 TBOT_OPTION                       = 2
 TEMP_TIME_SCHEME_OPTION           = 3
 GLACIER_OPTION                    = 1
 SURFACE_RESISTANCE_OPTION         = 4
 SOIL_DATA_OPTION                  = 1
 PEDOTRANSFER_OPTION               = 1
 CROP_OPTION                       = 0
 IRRIGATION_OPTION                 = 0
 IRRIGATION_METHOD                 = 0
 TILE_DRAINAGE_OPTION              = 0
 WETLAND_OPTION                    = 0

 FORCING_TIMESTEP        = 3600
 NOAH_TIMESTEP           = 3600
 OUTPUT_TIMESTEP         = 3600

 SPLIT_OUTPUT_COUNT      = 0
 SKIP_FIRST_OUTPUT       = .false.
 RESTART_FREQUENCY_HOURS = 99999

 NSOIL=4
 soil_thick_input(1) = 0.10
 soil_thick_input(2) = 0.30
 soil_thick_input(3) = 0.60
 soil_thick_input(4) = 1.00

 ZLVL = 10.0

 SF_URBAN_PHYSICS = 0
 USE_WUDAPT_LCZ   = 0

/
"""
    (run_dir / "namelist.hrldas").write_text(nml)

    log_path = site_dir / "hrldas.log"
    print(f"  Running hrldas.exe ...", flush=True)
    result = subprocess.run(
        ["./hrldas.exe"],
        cwd=str(run_dir),
        env=_HRLDAS_ENV,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    log_path.write_text(result.stdout + "\n" + result.stderr)

    if result.returncode != 0:
        print("STDERR tail:\n", result.stderr[-2000:])
        raise RuntimeError(f"hrldas.exe failed (rc={result.returncode}). Log: {log_path}")

    print(f"  NOAH-MP done. Log: {log_path}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Extract outputs
# ══════════════════════════════════════════════════════════════════════════════

def extract_outputs(site: dict, ldasout_file: Path, out_dir: Path):
    lat, lon = site["lat"], site["lon"]

    print(f"  Extracting LDASOUT: {ldasout_file}", flush=True)
    with nc4.Dataset(str(ldasout_file)) as ds:
        n     = ds.dimensions["Time"].size
        times = [pd.Timestamp("".join(t.data.astype(str)).replace("_", " "))
                 for t in ds.variables["Times"][:]]

        def g2d(v):
            if v not in ds.variables:
                return np.full(n, np.nan)
            a = np.array(ds.variables[v][:]).squeeze()
            return np.where(a > -9000, a.astype(float), np.nan)

        def g3d(v, k=0):
            if v not in ds.variables:
                return np.full(n, np.nan)
            a = np.array(ds.variables[v][:]).reshape(n, -1)
            col = a[:, k].astype(float)
            return np.where(col > -9000, col, np.nan)

        fsa    = g2d("FSA")
        fira   = g2d("FIRA")
        rn     = fsa - fira

        df_ldasout = pd.DataFrame({
            "timestamp": times,
            "SWdown":    g2d("SWFORC"),
            "LWdown":    g2d("LWFORC"),
            "Precip_mmhr": g2d("RAINRATE"),   # mm/timestep = mm/hr
            "Rn":        rn,
            "LAI":       g2d("LAI"),
            "Et":        g2d("ETRAN"),         # mm/s
            "Eg":        g2d("EDIR"),          # mm/s
            "H":         g2d("HFX"),           # W/m²
            "LE":        g2d("LH"),            # W/m²
            "SWC":       g3d("SOIL_M", 0),     # m³/m³ layer 1
            "Tv_hidden": g2d("TV"),            # canopy temp K
            "rb_hidden": g2d("CH"),            # bulk aerodyn. conductance m/s (1/rb)
            "Wc_hidden": g2d("CANLIQ"),        # canopy intercepted water mm
        })

    df_ldasout["timestamp"] = pd.to_datetime(df_ldasout["timestamp"])
    df_ldasout = df_ldasout.sort_values("timestamp").reset_index(drop=True)
    df_ldasout = df_ldasout.iloc[1:].reset_index(drop=True)   # drop IC row (all NaN)

    # Precip: mm/hr → mm/s
    df_ldasout["Precip"] = df_ldasout["Precip_mmhr"] / 3600.0
    df_ldasout = df_ldasout.drop(columns=["Precip_mmhr"])

    forcing_cache = out_dir / "forcing.parquet"
    if forcing_cache.exists():
        print(f"  Loading Tair/Ur from forcing cache ...", flush=True)
        era5_df = pd.read_parquet(forcing_cache)[["timestamp", "Tair", "Ur"]]
    else:
        print(f"  Loading ERA5 Tair and Ur (no cache, reading ERA5 files) ...", flush=True)
        tair = load_era5_point("t2m",  lat, lon)
        ur   = load_era5_point("ws10", lat, lon)
        era5_df = pd.DataFrame({
            "timestamp": tair.index,
            "Tair":      tair.values,
            "Ur":        ur.reindex(tair.index).values,
        })
        era5_df.to_parquet(forcing_cache, index=False)   # cache for future runs
    era5_df["timestamp"] = pd.to_datetime(era5_df["timestamp"])

    df = df_ldasout.merge(era5_df, on="timestamp", how="inner")
    df["site_id"]   = site["id"]
    df["site_name"] = site["name"]
    df["lat"]       = lat
    df["lon"]       = lon

    obs_cols = ["timestamp", "site_id", "site_name", "lat", "lon",
                "Tair", "Ur", "SWdown", "LWdown", "Precip",
                "Rn", "LAI", "Et", "Eg", "H", "LE", "SWC"]
    hid_cols = ["timestamp", "site_id", "site_name",
                "Tv_hidden", "rb_hidden", "Wc_hidden"]

    out_dir.mkdir(parents=True, exist_ok=True)
    df[obs_cols].to_csv(out_dir / "observed.csv", index=False)
    df[hid_cols].to_csv(out_dir / "hidden.csv",   index=False)
    print(f"  Saved {len(df)} rows → {out_dir}", flush=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site_id",      type=int, required=True)
    parser.add_argument("--skip_ldasin",  action="store_true", help="reuse existing LDASIN dir")
    parser.add_argument("--skip_noahmp",  action="store_true", help="skip run, go straight to extraction")
    args = parser.parse_args()

    if args.site_id not in SITES:
        sys.exit(f"Unknown site_id {args.site_id}. Valid: {sorted(SITES)}")

    site     = {**SITES[args.site_id], "id": args.site_id}
    site_dir = SITES_DIR / f"site_{args.site_id}"
    site_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Site {args.site_id}: {site['name']}  lat={site['lat']}  lon={site['lon']}", flush=True)
    print(f"{'='*60}", flush=True)

    ldasin_dir  = site_dir / "LDASIN"
    setup_nc    = site_dir / "hrldas_setup.nc"
    ldasout_dir = site_dir / "LDASOUT"

    if not args.skip_ldasin and not args.skip_noahmp:
        generate_ldasin(site, ldasin_dir)

    if not args.skip_noahmp:
        create_setup_nc(site, setup_nc)
        run_noahmp(site_dir, ldasin_dir, setup_nc, ldasout_dir)

    ldasout_files = sorted(ldasout_dir.glob("*.LDASOUT_DOMAIN1"))
    if not ldasout_files:
        sys.exit(f"No LDASOUT files found in {ldasout_dir}. Run without --skip_noahmp first.")

    extract_outputs(site, ldasout_files[0], site_dir)
    print(f"\nDone: site_{args.site_id}", flush=True)


if __name__ == "__main__":
    main()
