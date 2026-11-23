import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})


def load_tls_matrix(path):
    """Load a JASCO-style TLS matrix from a text file.

    The file format is expected to contain a header followed by an XYDATA block.
    The first data row after XYDATA lists the excitation wavelengths. Each
    subsequent row lists one emission wavelength followed by the intensity values
    for each excitation wavelength.
    """
    path = Path(path)
    with path.open("r", encoding="latin1") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if "XYDATA" not in lines:
        raise ValueError(f"No XYDATA section found in {path}")

    xy_index = lines.index("XYDATA")
    data_lines = [line for line in lines[xy_index + 1 :] if not line.startswith("#")]
    if not data_lines:
        raise ValueError(f"No TLS data rows found in {path}")

    excitation_wavelengths = np.array(
        [float(token) for token in data_lines[0].replace(",", " ").split()],
        dtype=float,
    )

    emission_wavelengths = []
    intensity_rows = []
    for line in data_lines[1:]:
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            emission_wavelengths.append(float(parts[0]))
            intensity_rows.append([float(token) for token in parts[1:]])
        except ValueError:
            continue

    if not emission_wavelengths:
        raise ValueError(f"No valid emission/intensity rows found in {path}")

    intensity_matrix = np.array(intensity_rows, dtype=float)
    if intensity_matrix.shape[1] != len(excitation_wavelengths):
        raise ValueError(
            f"Expected {len(excitation_wavelengths)} intensity values per row but got {intensity_matrix.shape[1]}"
        )

    emission_wavelengths = np.array(emission_wavelengths, dtype=float)
    return excitation_wavelengths, emission_wavelengths, intensity_matrix


def _normalize_spectra(matrix):
    """Scale all spectra relative to the single strongest peak across the matrix."""
    matrix = np.array(matrix, dtype=float, copy=True)
    if matrix.size == 0:
        return matrix

    maxval = np.nanmax(matrix)
    if np.isnan(maxval) or maxval <= 0:
        return matrix

    return matrix / maxval


def plot_tls_slices(
    path,
    outfile=None,
    normalize=True,
    palette="viridis",
    linewidth=1.4,
    title=None,
    outdir="Plot_TLS",
    excitation_range=None,
    emission_range=None,
    figsize=(10, 6),
    elev=24,
    azim=45,
):
    excitation_wavelengths, emission_wavelengths, intensity_matrix = load_tls_matrix(path)

    if excitation_range is not None:
        min_exc, max_exc = excitation_range
        if min_exc > max_exc:
            min_exc, max_exc = max_exc, min_exc
        mask = (excitation_wavelengths >= min_exc) & (excitation_wavelengths <= max_exc)
        if not np.any(mask):
            raise ValueError(f"No excitation wavelengths found between {min_exc} and {max_exc} nm")
        excitation_wavelengths = excitation_wavelengths[mask]
        intensity_matrix = intensity_matrix[:, mask]

    if emission_range is not None:
        min_em, max_em = emission_range
        if min_em > max_em:
            min_em, max_em = max_em, min_em
        mask = (emission_wavelengths >= min_em) & (emission_wavelengths <= max_em)
        if not np.any(mask):
            raise ValueError(f"No emission wavelengths found between {min_em} and {max_em} nm")
        emission_wavelengths = emission_wavelengths[mask]
        intensity_matrix = intensity_matrix[mask, :]

    if normalize:
        intensity_matrix = _normalize_spectra(intensity_matrix)

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    if outfile is None:
        outfile = Path(path).stem

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    n_slices = len(excitation_wavelengths)
    colors = sns.color_palette(palette, n_colors=max(n_slices, 2))

    for idx, exc_wl in enumerate(excitation_wavelengths):
        spectrum = intensity_matrix[:, idx]
        if np.all(np.isnan(spectrum)):
            continue
        ax.plot(
            np.full_like(emission_wavelengths, exc_wl),
            emission_wavelengths,
            spectrum,
            color=colors[idx % len(colors)],
            linewidth=linewidth,
            alpha=0.95,
        )

    ax.set_xlabel("Excitation wavelength (nm)")
    ax.set_ylabel("Emission wavelength (nm)")
    ax.set_zlabel("Intensity (a.u.)")
    ax.set_title(title or f"TLS slices from {Path(path).name}")

    if excitation_wavelengths.size > 1:
        ax.set_xlim(excitation_wavelengths.min() - 1, excitation_wavelengths.max() + 1)
    if emission_wavelengths.size > 1:
        ax.set_ylim(emission_wavelengths.min() - 1, emission_wavelengths.max() + 1)

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.2, 1.0, 0.7))
    fig.tight_layout()

    savepath = outdir / f"{outfile}.png"
    if savepath.exists():
        response = input(f"File exists: {savepath.resolve()}\nOverwrite? Type Y to confirm: ")
        if response.strip() != "Y":
            print(f"Skipped saving (overwrite not confirmed): {savepath.resolve()}")
            plt.close(fig)
            return

    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    print(f"Saved TLS plot to {savepath.resolve()}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot TLS spectra as a 3D stack of excitation slices")
    parser.add_argument("path", help="Path to the TLS text file")
    parser.add_argument("--outfile", default=None, help="Output file stem (without extension)")
    parser.add_argument("--no-normalize", action="store_true", help="Disable global normalization")
    parser.add_argument("--title", default=None, help="Optional figure title")
    parser.add_argument("--palette", default="viridis", help="Seaborn palette name")
    parser.add_argument("--linewidth", type=float, default=1.4, help="Line width for each slice")
    parser.add_argument("--outdir", default="Plot_TLS", help="Directory to save the output image")
    parser.add_argument(
        "--excitation-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Restrict the plot to a range of excitation wavelengths (min max)",
    )
    parser.add_argument(
        "--emission-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Restrict the plot to a range of emission wavelengths (min max)",
    )
    parser.add_argument("--figsize", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"), default=(10, 6), help="Figure size in inches")
    parser.add_argument("--elev", type=float, default=24, help="Elevation angle for the 3D view")
    parser.add_argument("--azim", type=float, default=45, help="Azimuth angle for the 3D view")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists() or not path.is_file():
        parser.error(f"TLS file not found or not a file: {path}")

    normalize = not args.no_normalize
    try:
        plot_tls_slices(
            path,
            outfile=args.outfile,
            normalize=normalize,
            palette=args.palette,
            linewidth=args.linewidth,
            title=args.title,
            outdir=args.outdir,
            excitation_range=args.excitation_range,
            emission_range=args.emission_range,
            figsize=tuple(args.figsize),
            elev=args.elev,
            azim=args.azim,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
