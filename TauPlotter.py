import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from pathlib import Path

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
})

def plot_lifetime(
    filepaths,
    log_x=False,
    log_y=False,
    title=None,
    xlabel=None,
    ylabel="counts",
    fit=False,
    fit_order=1,               # number of exponential components; can be scalar or list per-file
    fit_range=(5.0, 12.0),      # (x_min, x_max) fürs Fitting; can be a list per-file
    show_chi_in_legend=False,   # optionally include reduced chi value in fit legend entries
    show_tau_avg=False,          # optionally include τavg in fit legend entries
    interpolate=True,
    interp_to_y=1000,            # bis zu welchem y-Wert extrapolieren (Standard: 1 = 10^0)
    outfile=None,               # Name ohne Endung; wenn None -> aus Dateiname abgeleitet
    filetype="png",             # "png" oder "pdf"
    outdir="Plot_Lifetime",
    xlim=None,                  # optional: (xmin, xmax) für Darstellung
    ylim=None,                  # optional: (ymin, ymax) für Darstellung
    labels=None,                # optional: list of labels for each dataset
    time_unit="ns",             # time unit: "ns" (nanosecond), "us" (microsecond), or "ms" (millisecond)
    palette=None,            # Palette: seaborn sequential palette ("Blues", "Greens", "Reds", etc.) or "pyplot" for standard matplotlib colors
    use_seaborn=False,            # Use seaborn palette (True) or standard pyplot colors (False)
    specalpha=1.0
):
    """
    Plottet eine Zeitreihe aus 'filepath', optional mit n-exponentiellem Fit
    (Summe aus A_i * exp(-t/tau_i), C fest = 0) und Extrapolation bis zu einem bestimmten y-Wert.

    Datendatei: 2 Spalten, Komma-separiert, ohne Header.
    Assumes input data is in nanoseconds.
    """

    # Accept single filepath or list/tuple of filepaths
    if isinstance(filepaths, (str, Path)):
        file_list = [Path(filepaths)]
    else:
        file_list = [Path(p) for p in filepaths]
    n_files = len(file_list)
    
    # --- Validate input parameters ---
    if interp_to_y is not None and (not isinstance(interp_to_y, (int, float)) or interp_to_y <= 0):
        raise ValueError("interp_to_y must be a positive number or None")

    # Normalize parameters that may be scalar or lists
    def _broadcast_param(param, name, default=None):
        if isinstance(param, (list, tuple)):
            if len(param) != n_files:
                raise ValueError(f"Parameter '{name}' must have length equal to number of filepaths ({n_files})")
            return list(param)
        else:
            return [param] * n_files
    
    # Special handling for fit_range: distinguish between a single range tuple and a list of ranges
    def _broadcast_range(param, name, default=None):
        if param is None:
            return [None] * n_files
        if isinstance(param, (list, tuple)):
            # Check possibilities:
            # 1) A list of ranges, one per file: [(x_min,x_max), ...] with len == n_files
            # 2) A single (x_min, x_max) tuple which should be broadcast to all files
            # 3) A single-element list containing a (x_min, x_max) tuple -> broadcast it
            if len(param) == n_files and all(isinstance(p, (tuple, list)) for p in param):
                return list(param)
            # Single tuple like (x_min, x_max)
            if len(param) == 2 and all(isinstance(p, (int, float)) for p in param):
                return [param] * n_files
            # Single-element list containing a tuple: [(x_min,x_max)]
            if len(param) == 1 and isinstance(param[0], (tuple, list)) and len(param[0]) == 2 and all(isinstance(p, (int, float)) for p in param[0]):
                return [tuple(param[0])] * n_files
            raise ValueError(f"Parameter '{name}' must be either (x_min, x_max) or a list of (x_min, x_max) tuples")
        else:
            return [param] * n_files

    fit_list = _broadcast_param(fit, "fit", False)
    fit_order_list = _broadcast_param(fit_order, "fit_order", 1)
    fit_range_list = _broadcast_range(fit_range, "fit_range", None)

    # Validate fit order(s)
    for i, order in enumerate(fit_order_list):
        if not isinstance(order, (int, np.integer)) or int(order) < 1:
            raise ValueError(f"fit_order for dataset index {i} must be an integer >= 1, got: {order}")
        fit_order_list[i] = int(order)

    def multi_exp_decay(xv, *params):
        order_local = len(params) // 2
        amps = np.asarray(params[:order_local])
        taus = np.asarray(params[order_local:])
        yv = np.zeros_like(np.asarray(xv, dtype=float), dtype=float)
        for a, t in zip(amps, taus):
            yv += a * np.exp(-np.asarray(xv, dtype=float) / t)
        return yv

    def _fit_decay_segment(x_fit, y_fit, order):
        if len(x_fit) <= 2 * order:
            raise ValueError(
                f"Insufficient data points ({len(x_fit)}) for fit_order={order}. Need more than {2*order} points."
            )

        y_peak = max(float(np.max(y_fit)), 1e-12)
        x_span = max(float(np.max(x_fit) - np.min(x_fit)), 1e-12)

        amps0 = np.full(order, y_peak / order, dtype=float)
        tau_min0 = max(x_span / (20.0 * order), 1e-9)
        tau_max0 = max(x_span * 2.0, 1e-6)
        taus0 = np.geomspace(tau_min0, tau_max0, order)
        p0 = np.concatenate([amps0, taus0])

        lower_bounds = np.array([0.0] * order + [1e-12] * order, dtype=float)
        upper_bounds = np.array([np.inf] * (2 * order), dtype=float)

        params, cov = curve_fit(
            multi_exp_decay,
            x_fit,
            y_fit,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=200000,
        )

        amps = np.asarray(params[:order], dtype=float)
        taus = np.asarray(params[order:], dtype=float)

        # Sort by lifetime for consistent reporting
        sort_idx = np.argsort(taus)
        taus = taus[sort_idx]
        amps = amps[sort_idx]
        params_sorted = np.concatenate([amps, taus])

        y_pred = multi_exp_decay(x_fit, *params_sorted)
        residual = y_fit - y_pred
        # Reduced chi-square-like metric with Poisson-style weighting
        variance = np.clip(np.abs(y_pred), 1.0, None)
        chi2 = float(np.sum((residual ** 2) / variance))
        dof = max(len(x_fit) - 2 * order, 1)
        chi_red = chi2 / dof

        amp_sum = float(np.sum(amps))
        tau_avg = float(np.sum(amps * taus) / amp_sum) if amp_sum > 0 else np.nan

        return {
            "amps": amps,
            "taus": taus,
            "params": params_sorted,
            "cov": cov,
            "chi2": chi2,
            "chi_red": chi_red,
            "tau_avg": tau_avg,
        }

    def _estimate_x_target(amps, taus, y_target, x_start):
        if y_target is None or y_target <= 0:
            return None

        def model(xv):
            return float(np.sum(amps * np.exp(-xv / taus)))

        if model(x_start) <= y_target:
            return None

        x_hi = x_start + max(taus) * 20.0
        for _ in range(8):
            if model(x_hi) <= y_target:
                break
            x_hi += max(taus) * 20.0
        if model(x_hi) > y_target:
            return None

        x_grid = np.linspace(x_start, x_hi, 1200)
        y_grid = np.array([model(xx) for xx in x_grid])
        idx_cross = np.where(y_grid <= y_target)[0]
        if len(idx_cross) == 0:
            return None
        return float(x_grid[idx_cross[0]])
    
    # --- Validate and convert time units ---
    valid_units = {"ns": 1.0, "us": 1000.0, "ms": 1e6}
    if time_unit not in valid_units:
        raise ValueError(f"Invalid time_unit '{time_unit}'. Must be one of: {list(valid_units.keys())}")
    
    conversion_factor = valid_units[time_unit]

    # --- Daten laden und plotten (mehrere Dateien möglich) ---
    # --- Set xlabel if not provided ---
    if xlabel is None:
        unit_label = "μs" if time_unit == "us" else time_unit
        xlabel = f"time ({unit_label})"

    # --- Figure/Axes anlegen ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # Use seaborn palette or standard pyplot colors
    if use_seaborn:
        colors = sns.color_palette(palette, n_colors=max(n_files, 2))
    else:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    taus_out = []

    for idx, filepath in enumerate(file_list):
        # Load data (robust path handling for workspace-relative and prefixed paths)
        filepath_resolved = filepath
        if not filepath_resolved.exists():
            prefixed_parts = filepath.parts
            if len(prefixed_parts) > 1 and prefixed_parts[0] == Path.cwd().name:
                candidate = Path(*prefixed_parts[1:])
                if candidate.exists():
                    filepath_resolved = candidate
        if not filepath_resolved.exists():
            raise FileNotFoundError(
                f"Could not find data file: {filepath}. "
                f"Tried as given and with leading '{Path.cwd().name}/' removed."
            )

        df = pd.read_csv(filepath_resolved, sep=",", header=None, comment=";")
        x = df.iloc[:, 0].to_numpy() / conversion_factor  # Convert from ns to desired unit
        y = df.iloc[:, 1].to_numpy()

        # Label handling
        if labels is None:
            label_data = filepath.stem
        else:
            label_list = labels if isinstance(labels, (list, tuple)) else [labels]
            if len(label_list) != n_files:
                raise ValueError("'labels' must be a list with the same length as filepaths")
            label_data = label_list[idx]

        # Determine per-file fit parameters
        fit_i = fit_list[idx]
        fit_order_i = fit_order_list[idx]
        frange_i = fit_range_list[idx]

        # Plot data; hide raw trace legend entry when a fit legend entry will be shown
        if use_seaborn:
            color = colors[idx]
        else:
            color = colors[idx % len(colors)]
        data_label = "_nolegend_" if fit_i else label_data
        ax.plot(x, y, label=data_label, linewidth=1.0, color=color, alpha=specalpha, zorder=2)

        if fit_i:
            if frange_i is None:
                raise ValueError("fit_range must be specified for datasets with fit=True")
            x_min, x_max = frange_i
            mask = (x >= x_min) & (x <= x_max)
            x_fit = x[mask]
            y_fit = y[mask]
            if len(x_fit) < 2:
                raise ValueError(f"Insufficient data points in fit_range ({x_min}, {x_max}) for {filepath}")

            fitres = _fit_decay_segment(x_fit, y_fit, fit_order_i)
            amps = fitres["amps"]
            taus = fitres["taus"]

            tau_text = ", ".join([f"tau{k+1}={tv:.6f}" for k, tv in enumerate(taus)])
            print(
                f"{filepath.name} - fit (order={fit_order_i}): {tau_text}, "
                f"tau_avg={fitres['tau_avg']:.6f} ({time_unit}), chi_red={fitres['chi_red']:.6g}"
            )

            x_smooth = np.linspace(x_min, x_max, 400)
            y_smooth = multi_exp_decay(x_smooth, *fitres["params"])
            unit_label = "μs" if time_unit == "us" else time_unit
            legend_tau = ", ".join([f"τ{k+1}={tv:.3f}" for k, tv in enumerate(taus)])
            label_parts = [legend_tau]
            if show_tau_avg:
                label_parts.append(f"τavg={fitres['tau_avg']:.3f} {unit_label}")
            if show_chi_in_legend:
                label_parts.append(f"χr={fitres['chi_red']:.3g}")
            label_fit = f"{label_data} ({', '.join(label_parts)})"
            ax.plot(x_smooth, y_smooth, color=color, linestyle="-", linewidth=2.5, label=label_fit, zorder=10)

            if interpolate and interp_to_y is not None and interp_to_y > 0:
                x_target = _estimate_x_target(amps, taus, interp_to_y, x_max)
                if x_target is not None and x_target > x_max:
                    x_ext = np.linspace(x_max, x_target, 400)
                    y_ext = multi_exp_decay(x_ext, *fitres["params"])
                    ax.plot(x_ext, y_ext, color=color, linestyle="--", linewidth=1.5, zorder=8)

            taus_out.append({
                "taus": taus.tolist(),
                "amps": amps.tolist(),
                "tau_avg": fitres["tau_avg"],
                "chi_red": fitres["chi_red"],
            })

    # --- Achsenskalierung ---
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    # --- Grid for logarithmic axes ---
    if log_x or log_y:
        # Show minor gridlines for better logarithmic representation
        ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.7, color="gray")
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3, color="gray")
    
    # --- Labels / Titel ---
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    # --- Darstellungsbereiche ---
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Kein Abstand links/rechts
    ax.margins(x=0)

    ax.legend(fontsize=11)

    # --- Speichern ---
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if outfile is None:
        # Derive outfile from concatenated stems
        stems = "+".join([p.stem for p in file_list])
        outfile = stems + "_lifetime"

    savepath = outdir / f"{outfile}.{filetype}"
    should_save = True
    if savepath.exists():
        resp = input(f"File exists: {savepath.resolve()}\nOverwrite? Type Y to confirm: ")
        if resp.strip() != "Y":
            should_save = False
            print(f"Skipped saving (overwrite not confirmed): {savepath.resolve()}")

    if should_save:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"Plot gespeichert unter: {savepath.resolve()}")

    plt.show()

    # Return taus per dataset (None for datasets without fit)
    return taus_out

plot_lifetime(
    filepaths=[
        "DMAC-s2PFL-PN/20260610tau/s2pfl_degassed/2us.dat"
    ],
    log_y=True,
    #log_x=True,
    title="Degassed DMAC-s2PFL-PN in Toluene - 2us",
    fit=[True],
    fit_order=[1],
    fit_range=[(0.175, 0.75)],
    labels=["DMAc-s2PFL-PN degassed"],
    outfile="0610_DMAC_degassed_2us",
    interp_to_y=1,
    xlim=(0.15, 2.0),
    ylim=(1, 1500),
    time_unit="us",
    show_chi_in_legend=True,
    use_seaborn=True,
    palette="muted",
    specalpha=0.8
)

# Temperature dependent template
'''plot_lifetime(
    filepaths=[
        "DMAC-s2PFL-PN/260228tau/15wt%DMAC-s2PFL-PN DPEPO film/250K.dat",
        "DMAC-s2PFL-PN/260228tau/15wt%DMAC-s2PFL-PN DPEPO film/260K.dat",
        "DMAC-s2PFL-PN/260228tau/15wt%DMAC-s2PFL-PN DPEPO film/270K.dat",
        "DMAC-s2PFL-PN/260228tau/15wt%DMAC-s2PFL-PN DPEPO film/280K.dat",
        "DMAC-s2PFL-PN/260228tau/15wt%DMAC-s2PFL-PN DPEPO film/290K.dat",
        "DMAC-s2PFL-PN/260228tau/15wt%DMAC-s2PFL-PN DPEPO film/300K.dat",
    ],
    log_y=True,
    title="Temperature Dependent Lifetime of 1E-5M DMAC-s2PFL-PN in Toluene",
    fit=[True, True, True, True, True, True],
    fit_range=[(0.9, 3), (0.9, 3), (0.9, 3), (0.9, 3), (0.9, 3), (0.9, 3)],
    labels=["250K", "260K", "270K", "280K", "290K", "300K"],
    outfile="0228_DMAC_temp_tau_REPORT",
    interp_to_y=1,
    xlim=(0.71, 6),
    ylim=(1, 7000),
    time_unit="us",
    use_seaborn=True,
    palette="icefire",
    specalpha=0.7
)'''

# Palette note: deep is good for qualitative, like comparisons, "Blues", "Reds" are sequentially darker, good for
# Temp dependent etc, as well as Viridis. All sequential can be reverse with "*_r" suffix
# icefire is nice for cold-hot
