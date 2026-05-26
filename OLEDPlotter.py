from pathlib import Path
import re
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np
import seaborn as sns

# Match plotting style used by TauPlotter/SpecPlotter.
sns.set_style("whitegrid")


def _read_text_with_japanese_encodings(path):
    """Read text trying common Japanese encodings used by instrument exports."""
    encodings = ["cp932", "shift_jis", "iso2022_jp", "euc_jp", "utf-8", "latin1"]
    errors = []

    for enc in encodings:
        try:
            text = Path(path).read_text(encoding=enc)
            return text, enc
        except UnicodeDecodeError as exc:
            errors.append(f"{enc}: {exc}")

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        "Could not decode file. Tried encodings: " + "; ".join(errors),
    )


def _extract_eqe_jv_arrays(path):
    """
    Parse primary ALL measurement table and extract:
    - J: current density (mA/cm^2)
    - V: voltage (V)
    - EQE: external quantum efficiency (%)
    """
    text, used_encoding = _read_text_with_japanese_encodings(path)
    lines = text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "電流密度" in line and "電圧" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not locate data header in {path}")

    # Build column index map directly from the Japanese header row.
    header_cols = [c.strip() for c in lines[header_idx].split("\t") if c.strip()]

    def _find_col(label_prefix):
        for idx, col in enumerate(header_cols):
            if col.startswith(label_prefix):
                return idx
        return None

    j_idx = _find_col("電流密度")
    v_idx = _find_col("電圧")
    # Ensure we use the main EQE column, not differential EQE.
    eqe_idx = None
    for idx, col in enumerate(header_cols):
        if col.startswith("外部量子効率"):
            eqe_idx = idx
            break

    if j_idx is None or v_idx is None or eqe_idx is None:
        raise ValueError(
            f"Could not resolve required columns from header in {path}. "
            f"Found columns: {header_cols}"
        )

    j_vals = []
    v_vals = []
    eqe_vals = []

    started_table = False
    previous_v = None

    for line in lines[header_idx + 1 :]:
        parts_raw = line.split("\t")
        parts = [p.strip() for p in parts_raw]

        # The primary data block is contiguous. Once it started, a non-numeric
        # line indicates end-of-block and we should stop to avoid junk sections.
        try:
            _ = float(parts[0])
            is_numeric_row = True
        except (IndexError, ValueError):
            is_numeric_row = False

        if not is_numeric_row:
            if started_table:
                break
            continue

        started_table = True

        max_idx = max(j_idx, v_idx, eqe_idx)
        if len(parts) <= max_idx:
            continue

        try:
            j = float(parts[j_idx])
            v = float(parts[v_idx])
            eqe = float(parts[eqe_idx])
        except ValueError:
            continue

        if not (np.isfinite(j) and np.isfinite(v) and np.isfinite(eqe)):
            continue

        # Stop when the voltage sweep restarts or jumps backwards sharply.
        if previous_v is not None and v < previous_v - 0.05:
            break

        j_vals.append(j)
        v_vals.append(v)
        eqe_vals.append(eqe)
        previous_v = v

    if not j_vals:
        raise ValueError(f"No numeric data rows found in {path}")

    return np.array(j_vals), np.array(v_vals), np.array(eqe_vals), used_encoding


def _extract_oled_measurement_arrays(path):
    """Parse the primary OLED measurement table including luminance."""
    text, used_encoding = _read_text_with_japanese_encodings(path)
    lines = text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "電流密度" in line and "電圧" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not locate data header in {path}")

    header_cols = [c.strip() for c in lines[header_idx].split("\t") if c.strip()]

    def _find_col(label_prefix):
        for idx, col in enumerate(header_cols):
            if col.startswith(label_prefix):
                return idx
        return None

    j_idx = _find_col("電流密度")
    v_idx = _find_col("電圧")
    luminance_idx = _find_col("輝度")

    eqe_idx = None
    for idx, col in enumerate(header_cols):
        if col.startswith("外部量子効率"):
            eqe_idx = idx
            break

    if j_idx is None or v_idx is None or eqe_idx is None or luminance_idx is None:
        raise ValueError(
            f"Could not resolve required columns from header in {path}. "
            f"Found columns: {header_cols}"
        )

    j_vals = []
    v_vals = []
    eqe_vals = []
    luminance_vals = []

    started_table = False
    previous_v = None

    for line in lines[header_idx + 1 :]:
        parts_raw = line.split("\t")
        parts = [p.strip() for p in parts_raw]

        try:
            _ = float(parts[0])
            is_numeric_row = True
        except (IndexError, ValueError):
            is_numeric_row = False

        if not is_numeric_row:
            if started_table:
                break
            continue

        started_table = True

        max_idx = max(j_idx, v_idx, eqe_idx, luminance_idx)
        if len(parts) <= max_idx:
            continue

        try:
            j = float(parts[j_idx])
            v = float(parts[v_idx])
            eqe = float(parts[eqe_idx])
            luminance = float(parts[luminance_idx])
        except ValueError:
            continue

        if not (np.isfinite(j) and np.isfinite(v) and np.isfinite(eqe) and np.isfinite(luminance)):
            continue

        if previous_v is not None and v < previous_v - 0.05:
            break

        j_vals.append(j)
        v_vals.append(v)
        eqe_vals.append(eqe)
        luminance_vals.append(luminance)
        previous_v = v

    if not j_vals:
        raise ValueError(f"No numeric data rows found in {path}")

    return {
        "j": np.array(j_vals),
        "v": np.array(v_vals),
        "eqe": np.array(eqe_vals),
        "luminance": np.array(luminance_vals),
        "encoding": used_encoding,
    }


def _extract_oled_spectrum(
    path,
    selection="max_luminance",
    target_voltage=10,
    normalize=True,
    wavelength_range=None,
):
    """Extract one representative EL spectrum from the lower spectrum table."""
    if selection not in {"max_luminance", "last", "max_eqe"}:
        raise ValueError("selection must be one of: 'max_luminance', 'last', 'max_eqe'")

    measurement = _extract_oled_measurement_arrays(path)
    text, used_encoding = _read_text_with_japanese_encodings(path)
    lines = text.splitlines()

    spectrum_header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("波長") and "Intensity1 [counts]" in line:
            spectrum_header_idx = i
            break

    if spectrum_header_idx is None:
        raise ValueError(f"Could not locate spectrum header in {path}")

    header_cols = [c.strip() for c in lines[spectrum_header_idx].split("\t") if c.strip()]
    if len(header_cols) < 2:
        raise ValueError(f"Spectrum header is malformed in {path}")

    spectrum_voltages = []
    for col in header_cols[1:]:
        match = re.search(r"at\s+([0-9.]+)\s*\[V\]", col)
        spectrum_voltages.append(float(match.group(1)) if match else np.nan)

    wavelengths = []
    spectra_rows = []

    for line in lines[spectrum_header_idx + 1 :]:
        parts_raw = line.split("\t")
        parts = [p.strip() for p in parts_raw]

        try:
            wavelength = float(parts[0])
            is_numeric_row = True
        except (IndexError, ValueError):
            is_numeric_row = False

        if not is_numeric_row:
            if wavelengths:
                break
            continue

        if len(parts) <= len(spectrum_voltages):
            continue

        try:
            intensity_row = [float(value) for value in parts[1 : 1 + len(spectrum_voltages)]]
        except ValueError:
            continue

        wavelengths.append(wavelength)
        spectra_rows.append(intensity_row)

    if not wavelengths:
        raise ValueError(f"No spectrum rows found in {path}")

    wavelengths = np.array(wavelengths)
    spectra = np.array(spectra_rows)
    voltage_axis = np.array(spectrum_voltages, dtype=float)

    if target_voltage is not None:
        selected_idx = int(np.nanargmin(np.abs(voltage_axis - float(target_voltage))))
        selection_label = f"{voltage_axis[selected_idx]:g} V"
    elif selection == "last":
        selected_idx = len(voltage_axis) - 1
        selection_label = f"last ({voltage_axis[selected_idx]:g} V)"
    elif selection == "max_eqe":
        selected_voltage = float(measurement["v"][int(np.nanargmax(measurement["eqe"]))])
        selected_idx = int(np.nanargmin(np.abs(voltage_axis - selected_voltage)))
        selection_label = f"max EQE ({voltage_axis[selected_idx]:g} V)"
    else:
        selected_voltage = float(measurement["v"][int(np.nanargmax(measurement["luminance"]))])
        selected_idx = int(np.nanargmin(np.abs(voltage_axis - selected_voltage)))
        selection_label = f"max luminance ({voltage_axis[selected_idx]:g} V)"

    intensity = spectra[:, selected_idx]

    if wavelength_range is not None:
        wl_min, wl_max = wavelength_range
        mask = np.isfinite(wavelengths) & np.isfinite(intensity)
        if wl_min is not None:
            mask &= wavelengths >= float(wl_min)
        if wl_max is not None:
            mask &= wavelengths <= float(wl_max)

        wavelengths = wavelengths[mask]
        intensity = intensity[mask]

        if len(wavelengths) == 0:
            raise ValueError(
                f"No spectrum points remain in range {wavelength_range} for {path}"
            )

    if normalize:
        peak = np.nanmax(intensity)
        if peak != 0:
            intensity = intensity / peak

    return wavelengths, intensity, selection_label, used_encoding


def select_oled_files(initialdir="DMAC-s2PFL-PN/260305_DMAC-s2PFL-PN OLED"):
    """Open a multi-file picker for AllMesData text files."""
    root = tk.Tk()
    root.withdraw()

    selected = filedialog.askopenfilenames(
        title="Select AllMesData files",
        initialdir=str(Path(initialdir).resolve()),
        filetypes=[("AllMesData text", "*AllMesData*.txt"), ("Text files", "*.txt"), ("All files", "*.*")],
    )

    root.destroy()
    return [Path(p) for p in selected]


def _prepare_eqe_plot_data(
    j,
    v,
    eqe,
    filter_mode="analysis-safe",
    min_voltage_for_eqe=3.0,
    eqe_valid_range=(0.0, 100.0),
    percentile_clip=(1.0, 99.0),
):
    """Prepare EQE-J data with optional filtering for physically meaningful range."""
    if filter_mode not in {"none", "analysis-safe"}:
        raise ValueError("filter_mode must be 'none' or 'analysis-safe'")

    mask = np.isfinite(j) & np.isfinite(v) & np.isfinite(eqe)

    if filter_mode == "analysis-safe":
        mask &= v >= float(min_voltage_for_eqe)
        mask &= j > 0

        if eqe_valid_range is not None:
            lo, hi = eqe_valid_range
            if lo is not None:
                mask &= eqe >= float(lo)
            if hi is not None:
                mask &= eqe <= float(hi)

    j2 = j[mask]
    eqe2 = eqe[mask]

    if filter_mode == "analysis-safe" and percentile_clip is not None and len(eqe2) > 4:
        p_lo, p_hi = percentile_clip
        lo_v, hi_v = np.percentile(eqe2, [p_lo, p_hi])
        p_mask = (eqe2 >= lo_v) & (eqe2 <= hi_v)
        j2 = j2[p_mask]
        eqe2 = eqe2[p_mask]

    return j2, eqe2


def plot_oled_curves(
    filepaths=None,
    labels=None,
    title_prefix=None,
    outfile_base=None,
    filetype="png",
    outdir="Plot_OLED",
    use_seaborn=True,
    palette="muted",
    linewidth=1.6,
    alpha=0.95,
    eqe_filter_mode="analysis-safe",
    min_voltage_for_eqe=3.0,
    eqe_valid_range=(0.0, 100.0),
    eqe_percentile_clip=(1.0, 99.0),
    eqe_log_x=False,
    eqe_log_y=False,
    eqe_ylim=None,
    plot_spectrum=False,
    spectrum_normalize=True,
    spectrum_selection="max_luminance",
    spectrum_voltage=10,
    spectrum_xlim=None,
    spectrum_ylim=None,
    jv_xlim=None,
    jv_ylim=None,
    jv_log_y=False,
    show=True,
):
    """
    Plot from one or more AllMesData files:
    - EQE-J: y = External Quantum Efficiency (%), x = Current Density (mA/cm^2)
    - J-V:   y = Current Density (mA/cm^2), x = Voltage (V)
    - EL:    y = Normalized Intensity (a.u.), x = Wavelength (nm)

    Spectrum selection defaults to the nearest available trace at `spectrum_voltage`.
    To use dynamic selection instead, set `spectrum_voltage=None` and choose
    `spectrum_selection="max_luminance"`, `"max_eqe"`, or `"last"`.
    """
    if filepaths is None:
        path_list = select_oled_files()
        if not path_list:
            print("No files selected. Exiting.")
            return
    elif isinstance(filepaths, (str, Path)):
        path_list = [Path(filepaths)]
    else:
        path_list = [Path(p) for p in filepaths]

    n = len(path_list)
    if n == 0:
        raise ValueError("No input files were provided.")

    if labels is not None and len(labels) != n:
        raise ValueError("'labels' must have same length as 'filepaths'.")

    if use_seaborn:
        colors = sns.color_palette(palette, n_colors=max(n, 2))
    else:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    parsed = []
    for i, p in enumerate(path_list):
        j, v, eqe, enc = _extract_eqe_jv_arrays(p)
        label = labels[i] if labels is not None else p.stem.replace("AllMesData", "")
        spectrum_data = None
        if plot_spectrum:
            wl, intensity, spectrum_note, _spectrum_enc = _extract_oled_spectrum(
                p,
                selection=spectrum_selection,
                target_voltage=spectrum_voltage,
                normalize=spectrum_normalize,
                wavelength_range=spectrum_xlim,
            )
            spectrum_data = (wl, intensity, spectrum_note)
        parsed.append((p, j, v, eqe, label, enc, spectrum_data))

    fig_eqe, ax_eqe = plt.subplots(figsize=(8, 5))
    for i, (_p, j, _v, eqe, label, _enc, _spectrum_data) in enumerate(parsed):
        color = colors[i] if use_seaborn else colors[i % len(colors)]
        j_plot, eqe_plot = _prepare_eqe_plot_data(
            j,
            _v,
            eqe,
            filter_mode=eqe_filter_mode,
            min_voltage_for_eqe=min_voltage_for_eqe,
            eqe_valid_range=eqe_valid_range,
            percentile_clip=eqe_percentile_clip,
        )

        if len(j_plot) == 0:
            print(f"Warning: no EQE points remaining after filtering for '{label}'.")
            continue

        if eqe_log_y:
            positive_mask = eqe_plot > 0
            removed = int(len(eqe_plot) - np.count_nonzero(positive_mask))
            if removed > 0:
                print(
                    f"Warning: removed {removed} non-positive EQE points for log-y plot "
                    f"('{label}')."
                )
            j_plot = j_plot[positive_mask]
            eqe_plot = eqe_plot[positive_mask]

            if len(j_plot) == 0:
                print(f"Warning: no positive EQE points remaining for log-y plot in '{label}'.")
                continue

        sort_idx = np.argsort(j_plot)
        ax_eqe.plot(
            j_plot[sort_idx],
            eqe_plot[sort_idx],
            label=label,
            linewidth=linewidth,
            alpha=alpha,
            color=color,
        )

    ax_eqe.set_xlabel("Current Density (mA/cm^2)")
    ax_eqe.set_ylabel("External Quantum Efficiency (%)")
    eqe_title = "EQE-J"
    if title_prefix:
        eqe_title = f"{title_prefix} - {eqe_title}"
    ax_eqe.set_title(eqe_title)
    if eqe_log_x:
        ax_eqe.set_xscale("log")
        ax_eqe.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.7, color="gray")
        ax_eqe.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3, color="gray")
    if eqe_log_y:
        ax_eqe.set_yscale("log")
        # Show halfway points (0.5, 5, 50, etc.) between powers of 10
        ax_eqe.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 5)))
        # Use decimal formatter for log scale (e.g., "0.1", "1", "10" instead of "10^-1", "10^0", "10^1")
        ax_eqe.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{y:.10g}")
        )
        ax_eqe.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.7, color="gray")
        ax_eqe.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3, color="gray")
    ax_eqe.legend()
    ax_eqe.margins(x=0)
    if eqe_ylim is not None:
        ax_eqe.set_ylim(eqe_ylim)

    fig_jv, ax_jv = plt.subplots(figsize=(8, 5))
    for i, (_p, j, v, _eqe, label, _enc, _spectrum_data) in enumerate(parsed):
        color = colors[i] if use_seaborn else colors[i % len(colors)]
        v_plot = v
        j_plot = j

        if jv_log_y:
            positive_mask = j_plot > 0
            removed = int(len(j_plot) - np.count_nonzero(positive_mask))
            if removed > 0:
                print(
                    f"Warning: removed {removed} non-positive current density points for log-y plot "
                    f"('{label}')."
                )
            v_plot = v_plot[positive_mask]
            j_plot = j_plot[positive_mask]

            if len(j_plot) == 0:
                print(f"Warning: no positive J points remaining for log-y plot in '{label}'.")
                continue

        sort_idx = np.argsort(v_plot)
        ax_jv.plot(v_plot[sort_idx], j_plot[sort_idx], label=label, linewidth=linewidth, alpha=alpha, color=color)

    ax_jv.set_xlabel("Voltage (V)")
    ax_jv.set_ylabel("Current Density (mA/cm^2)")
    jv_title = "J-V"
    if title_prefix:
        jv_title = f"{title_prefix} - {jv_title}"
    ax_jv.set_title(jv_title)
    if jv_log_y:
        ax_jv.set_yscale("log")
    ax_jv.legend()
    ax_jv.margins(x=0)
    if jv_xlim is not None:
        ax_jv.set_xlim(jv_xlim)
    if jv_ylim is not None:
        ax_jv.set_ylim(jv_ylim)

    fig_spectrum = None
    ax_spectrum = None
    if plot_spectrum:
        fig_spectrum, ax_spectrum = plt.subplots(figsize=(8, 5))
        for i, (_p, _j, _v, _eqe, label, _enc, spectrum_data) in enumerate(parsed):
            color = colors[i] if use_seaborn else colors[i % len(colors)]
            if spectrum_data is None:
                continue

            wavelength, intensity, spectrum_note = spectrum_data
            ax_spectrum.plot(
                wavelength,
                intensity,
                label=f"{label} ({spectrum_note})",
                linewidth=linewidth,
                alpha=alpha,
                color=color,
            )

        ax_spectrum.set_xlabel("Wavelength (nm)")
        ax_spectrum.set_ylabel("Normalized Intensity (a.u.)" if spectrum_normalize else "Intensity (counts)")
        spectrum_title = "EL Spectrum"
        if title_prefix:
            spectrum_title = f"{title_prefix} - {spectrum_title}"
        ax_spectrum.set_title(spectrum_title)
        ax_spectrum.legend()
        ax_spectrum.margins(x=0)
        if spectrum_xlim is not None:
            ax_spectrum.set_xlim(spectrum_xlim)
        if spectrum_ylim is not None:
            ax_spectrum.set_ylim(spectrum_ylim)

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    if outfile_base is None:
        stems = "+".join([p.stem for p in path_list])
        outfile_base = stems + "_oled"

    save_eqe = outdir_path / f"{outfile_base}_EQEJ.{filetype}"
    save_jv = outdir_path / f"{outfile_base}_JV.{filetype}"
    save_spectrum = outdir_path / f"{outfile_base}_Spectrum.{filetype}"

    save_eqe_ok = True
    save_jv_ok = True
    save_spectrum_ok = True

    if save_eqe.exists():
        resp = input(f"File exists: {save_eqe.resolve()}\nOverwrite? Type Y to confirm: ")
        if resp.strip() != "Y":
            save_eqe_ok = False
            print(f"Skipped saving (overwrite not confirmed): {save_eqe.resolve()}")

    if save_jv.exists():
        resp = input(f"File exists: {save_jv.resolve()}\nOverwrite? Type Y to confirm: ")
        if resp.strip() != "Y":
            save_jv_ok = False
            print(f"Skipped saving (overwrite not confirmed): {save_jv.resolve()}")

    if plot_spectrum and save_spectrum.exists():
        resp = input(f"File exists: {save_spectrum.resolve()}\nOverwrite? Type Y to confirm: ")
        if resp.strip() != "Y":
            save_spectrum_ok = False
            print(f"Skipped saving (overwrite not confirmed): {save_spectrum.resolve()}")

    if save_eqe_ok:
        fig_eqe.savefig(save_eqe, dpi=300, bbox_inches="tight")
    if save_jv_ok:
        fig_jv.savefig(save_jv, dpi=300, bbox_inches="tight")
    if plot_spectrum and save_spectrum_ok and fig_spectrum is not None:
        fig_spectrum.savefig(save_spectrum, dpi=300, bbox_inches="tight")

    print(f"Decoded/parsed {n} file(s):")
    for p, _j, _v, _eqe, _label, enc, _spectrum_data in parsed:
        print(f"  - {p} (encoding: {enc})")
    if save_eqe_ok:
        print(f"EQE-J plot saved to: {save_eqe.resolve()}")
    if save_jv_ok:
        print(f"J-V plot saved to:   {save_jv.resolve()}")
    if plot_spectrum and save_spectrum_ok:
        print(f"EL spectrum saved to: {save_spectrum.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig_eqe)
        plt.close(fig_jv)
        if fig_spectrum is not None:
            plt.close(fig_spectrum)


if __name__ == "__main__":
    # If no filepaths are provided, a file picker opens.
    plot_oled_curves(
        filepaths=["DMAC-s2PFL-PN/260305_DMAC-s2PFL-PN OLED/Sample0012AllMesData.txt"
                ], 
        title_prefix="DMAC-s2PFL-PN OLED",
        use_seaborn=True,
        palette="muted",
        show=True,
        #JV
        jv_log_y=True,
        jv_xlim=(2.5, 15),
        #EQE
        eqe_filter_mode="analysis-safe",
        eqe_log_x=True,
        eqe_log_y=True,
        #eqe_log_y=True,
        eqe_percentile_clip=None,
        outfile_base="DMAC-s2PFL-PN_OLED_012",
        min_voltage_for_eqe=5,
        eqe_valid_range=(0, 100),
        #Spectrum
        plot_spectrum=True,
        spectrum_xlim=(400, 650),
        spectrum_voltage=10,
        spectrum_normalize=True,
    )
