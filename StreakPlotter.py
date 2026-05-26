import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import seaborn as sns
from pathlib import Path
import re
import os

# Set seaborn style
sns.set_style("whitegrid")


def _load_streak_file(filepath):
    """
    Load a streak camera data file and extract metadata.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the streak camera .txt file
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'time': numpy array of time values
        - 'wavelengths': numpy array of wavelength values (in nm)
        - 'intensity': 2D numpy array of photon counts (time x wavelength)
        - 'time_unit': str ('ns' or 'us')
        - 'temperature': str (e.g., '300K')
        - 'time_window': str (e.g., '200ns' or '20us')
        - 'filepath': Path object
    """
    filepath = Path(filepath)
    
    # Extract metadata from filename
    # Expected format: "15wt%DMAC-s2PFL-PN DPEPO 200ns 300K.txt"
    filename = filepath.stem
    
    # Extract time window (e.g., "200ns" or "20us")
    time_window_match = re.search(r'(\d+)(ns|us)', filename, re.IGNORECASE)
    if time_window_match:
        time_window = time_window_match.group(0)
        time_unit = time_window_match.group(2).lower()
    else:
        time_window = "unknown"
        time_unit = "ns"
    
    # Extract temperature (e.g., "300K")
    temp_match = re.search(r'(\d+)K', filename)
    temperature = temp_match.group(0) if temp_match else "unknown"
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Find the data start marker
    data_start_idx = None
    for i, line in enumerate(lines):
        if '<<END ASCII COMMENT>>' in line:
            data_start_idx = i + 1
            break
    
    if data_start_idx is None:
        raise ValueError(f"Could not find data start marker in {filepath}")
    
    # Read wavelength headers (first row after marker)
    header_line = lines[data_start_idx].strip()
    headers = header_line.split(',')
    
    # First column is time unit label (e.g., "ns/nm" or "us/nm"), rest are wavelengths
    # Extract the actual time unit from the column header instead of filename
    time_label = headers[0].strip()
    if '/' in time_label:
        time_unit_from_header = time_label.split('/')[0].strip().lower()
        if time_unit_from_header in ['ns', 'us', 'ms']:
            time_unit = time_unit_from_header
    
    wavelengths = np.array([float(h) for h in headers[1:]])
    
    # Read data rows
    data_lines = lines[data_start_idx + 1:]
    time_values = []
    intensity_values = []
    
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        values = line.split(',')
        if len(values) < 2:
            continue
        try:
            time_values.append(float(values[0]))
            intensity_row = [float(v) for v in values[1:]]
            intensity_values.append(intensity_row)
        except ValueError:
            continue
    
    time = np.array(time_values)
    intensity = np.array(intensity_values)
    
    return {
        'time': time,
        'wavelengths': wavelengths,
        'intensity': intensity,
        'time_unit': time_unit,
        'temperature': temperature,
        'time_window': time_window,
        'filepath': filepath
    }


def _broadcast_param(param, n_files, name):
    """
    Broadcast a parameter to match the number of files.
    
    Parameters
    ----------
    param : scalar or list
        Parameter value(s)
    n_files : int
        Number of files
    name : str
        Parameter name for error messages
        
    Returns
    -------
    list
        List of parameter values, one per file
    """
    if isinstance(param, (list, tuple)):
        if len(param) != n_files:
            raise ValueError(f"Parameter '{name}' must have length equal to number of filepaths ({n_files})")
        return list(param)
    else:
        return [param] * n_files


def _format_time_unit(time_unit):
    """
    Format time unit for display (e.g., 'us' -> 'μs').
    
    Parameters
    ----------
    time_unit : str
        Time unit string ('ns', 'us', 'ms', etc.)
        
    Returns
    -------
    str
        Formatted time unit for display
    """
    if time_unit.lower() == 'us':
        return 'μs'
    elif time_unit.lower() == 'ms':
        return 'ms'
    elif time_unit.lower() == 'ns':
        return 'ns'
    else:
        return time_unit


def plot_transient_lifetimes(
    filepaths,
    labels=None,
    colors=None,
    palette="husl",
    log_scale=False,
    normalize=False,
    wavelength_range=None,
    title=None,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    auto_xlim=True,
    xlim_pad_factor=0.1,
    outfile=None,
    filetype="png",
    outdir="Plot_Lifetime",
    linewidth=1.5
):
    """
    Plot transient photoluminescence lifetimes from streak camera data.
    
    Integrates intensity across all wavelengths (or a specified range) to create
    intensity vs. time plots. Automatically groups files by time window (200ns vs 20us)
    and creates separate plots for each window.
    
    Parameters
    ----------
    filepaths : str, Path, or list
        Path(s) to streak camera .txt file(s)
    labels : list of str, optional
        Custom labels for each file. If None, uses temperature from filename
    colors : list, optional
        Custom colors for each file
    palette : str, optional
        Seaborn color palette name (default: "husl")
    log_scale : bool, optional
        Use logarithmic y-axis (default: False)
    normalize : bool, optional
        Normalize each trace to its maximum value (default: False)
    wavelength_range : tuple, optional
        (min_wavelength, max_wavelength) for integration. If None, uses full range
    title : str, optional
        Custom plot title
    xlabel : str, optional
        Custom x-axis label
    ylabel : str, optional
        Custom y-axis label
    xlim : tuple, optional
        (xmin, xmax) for x-axis limits
    ylim : tuple, optional
        (ymin, ymax) for y-axis limits
    outfile : str, optional
        Output filename without extension. If None, auto-generates from input files
    filetype : str, optional
        Output file type: "png" or "pdf" (default: "png")
    outdir : str, optional
        Output directory (default: "Plot_Lifetime")
    linewidth : float, optional
        Line width for plots (default: 1.5)
        
    Returns
    -------
    dict
        Dictionary with time_window keys, each containing a dict with 'fig' and 'data'
    """
    # Ensure filepaths is a list
    if isinstance(filepaths, (str, Path)):
        filepaths = [filepaths]
    filepaths = [Path(fp) for fp in filepaths]
    
    n_files = len(filepaths)
    
    # Load all data files
    print(f"Loading {n_files} streak camera files...")
    data_list = []
    for fp in filepaths:
        try:
            data = _load_streak_file(fp)
            data_list.append(data)
            print(f"  Loaded: {fp.name} ({data['temperature']}, {data['time_window']})")
        except Exception as e:
            print(f"  Warning: Could not load {fp.name}: {e}")
    
    if not data_list:
        raise ValueError("No valid data files loaded")
    
    # Group files by time window
    time_window_groups = {}
    for data in data_list:
        tw = data['time_window']
        if tw not in time_window_groups:
            time_window_groups[tw] = []
        time_window_groups[tw].append(data)
    
    print(f"\nFound {len(time_window_groups)} time window(s): {list(time_window_groups.keys())}")
    
    # Broadcast parameters
    if labels is None:
        labels = [data['temperature'] for data in data_list]
    else:
        labels = _broadcast_param(labels, n_files, 'labels')
    
    # Create label lookup by data object
    label_map = {id(data): labels[i] for i, data in enumerate(data_list)}
    
    # Create plots for each time window
    results = {}
    
    for time_window, group_data in time_window_groups.items():
        n_traces = len(group_data)
        print(f"\n{'='*60}")
        print(f"Plotting transient lifetimes for {time_window} window ({n_traces} traces)")
        print(f"{'='*60}")
        
        # Get colors for this group
        if colors is None:
            group_colors = sns.color_palette(palette, n_traces)
        else:
            # Extract colors for files in this group
            group_colors = [colors[data_list.index(data)] for data in group_data]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot each trace
        plotted_data = []
        for i, data in enumerate(group_data):
            time = data['time']
            intensity = data['intensity']
            wavelengths = data['wavelengths']
            
            # Filter wavelength range if specified
            if wavelength_range is not None:
                wl_min, wl_max = wavelength_range
                wl_mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
                intensity_filtered = intensity[:, wl_mask]
            else:
                intensity_filtered = intensity
            
            # Integrate across wavelengths (sum along wavelength axis)
            integrated_intensity = np.sum(intensity_filtered, axis=1)
            
            # Normalize if requested
            if normalize:
                max_val = np.max(integrated_intensity)
                if max_val > 0:
                    integrated_intensity = integrated_intensity / max_val
            
            # Get label for this trace
            label = label_map[id(data)]
            
            # Plot
            ax.plot(time, integrated_intensity, 
                   label=label, 
                   color=group_colors[i],
                   linewidth=linewidth)
            
            plotted_data.append({
                'time': time,
                'intensity': integrated_intensity,
                'label': label,
                'temperature': data['temperature']
            })
        
        # Set labels and title
        time_unit = group_data[0]['time_unit']
        time_unit_display = _format_time_unit(time_unit)
        
        if xlabel is None:
            xlabel_text = f'Time ({time_unit_display})'
        else:
            xlabel_text = xlabel
        
        if ylabel is None:
            if normalize:
                ylabel_text = 'Normalized PL Intensity (a.u.)'
            else:
                ylabel_text = 'Integrated PL Intensity (counts)'
        else:
            ylabel_text = ylabel
        
        if title is None:
            title_text = f'Transient PL ({time_window} window)'
        else:
            title_text = title
        
        ax.set_xlabel(xlabel_text, fontsize=12)
        ax.set_ylabel(ylabel_text, fontsize=12)
        ax.set_title(title_text, fontsize=14, fontweight='bold')
        
        # Set log scale if requested
        if log_scale:
            ax.set_yscale('log')
            # Add grid ticks for log scale
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
            ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7, color='gray')
            ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.3, color='gray')
        
        # Set axis limits if specified
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        
        # Add legend
        ax.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        # Grid (will be overridden with more specific grid for log scale above)
        if not log_scale:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        
        if outfile is None:
            # Auto-generate filename from input files
            if n_traces == 1:
                base_name = group_data[0]['filepath'].stem
            else:
                # Use compound name
                temps = sorted(set(d['temperature'] for d in group_data))
                base_name = f"transients_{time_window}_{'_'.join(temps)}"
        else:
            base_name = outfile
        
        # Append time window to filename if multiple windows
        if len(time_window_groups) > 1:
            output_filename = f"{base_name}_{time_window}.{filetype}"
        else:
            output_filename = f"{base_name}.{filetype}"
        
        output_path = outdir_path / output_filename
        
        # Check if file exists
        if output_path.exists():
            response = input(f"File {output_path} already exists. Type Y to overwrite: ")
            if response.upper() != 'Y':
                print("Save cancelled.")
                plt.close(fig)
                continue
        
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {output_path}")
        
        results[time_window] = {
            'fig': fig,
            'data': plotted_data,
            'output_path': output_path
        }
    
    return results


def plot_timesliced_spectra(
    filepaths,
    time_slice,
    labels=None,
    colors=None,
    palette="muted",
    normalize=True,
    annotate_max=False,
    linestyles=None,
    linewidth=1.5,
    title=None,
    xlabel=r'$\lambda$ (nm)',
    ylabel='Intensity (a.u.)',
    xlim=None,
    ylim=None,
    outfile=None,
    filetype="png",
    outdir="Plot_Spectra"
):
    """
    Plot wavelength-dependent PL spectra from time slices or integrated time ranges.
    
    For each file, extracts spectra at one or more time points and/or integrates
    spectra over one or more time windows, then overlays them.
    
    Parameters
    ----------
    filepaths : str, Path, or list
        Path(s) to streak camera .txt file(s)
    time_slice : float, tuple, list, or nested list
        Time selector(s) in ns or us for extracting spectra.
        A selector can be either:
        - float: nearest-time spectrum at that time
        - (t_min, t_max): integrated spectrum over that time range

        Accepted structures:
        - Single float: one time slice applied to all files
        - Single-file + list: multiple selectors overlaid for that file
        - Multi-file + list (len == n_files): one selector per file
        - Multi-file + nested list (len == n_files): multiple selectors per file
    labels : list of str, optional
        Custom labels for each file. If None, auto-generates from temperature and time
    colors : list, optional
        Custom colors for each spectrum
    palette : str, optional
        Seaborn color palette name (default: "muted")
    normalize : bool, optional
        Normalize each spectrum to its maximum value (default: True)
    annotate_max : bool, optional
        Annotate the peak wavelength on each spectrum (default: False)
    linestyles : list of str, optional
        Line styles for each spectrum. If None, all traces use solid lines.
    linewidth : float, optional
        Line width for plots (default: 1.5)
    title : str, optional
        Custom plot title
    xlabel : str, optional
        Custom x-axis label (default: wavelength in nm)
    ylabel : str, optional
        Custom y-axis label
    xlim : tuple, optional
        (xmin, xmax) for x-axis limits
    ylim : tuple, optional
        (ymin, ymax) for y-axis limits
    outfile : str, optional
        Output filename without extension. If None, auto-generates from input files
    filetype : str, optional
        Output file type: "png" or "pdf" (default: "png")
    outdir : str, optional
        Output directory (default: "Plot_Spectra")
        
    Returns
    -------
    dict
        Dictionary with 'fig', 'data', and 'output_path' keys
    """
    # Ensure filepaths is a list
    if isinstance(filepaths, (str, Path)):
        filepaths = [filepaths]
    filepaths = [Path(fp) for fp in filepaths]
    
    n_files = len(filepaths)
    
    # Load all data files
    print(f"Loading {n_files} streak camera files...")
    data_list = []
    for fp in filepaths:
        try:
            data = _load_streak_file(fp)
            data_list.append(data)
            print(f"  Loaded: {fp.name} ({data['temperature']}, {data['time_window']})")
        except Exception as e:
            print(f"  Warning: Could not load {fp.name}: {e}")
    
    if not data_list:
        raise ValueError("No valid data files loaded")
    
    def _is_sequence(val):
        return isinstance(val, (list, tuple, np.ndarray))

    def _is_range_selector(val):
        return _is_sequence(val) and len(val) == 2 and all(not _is_sequence(v) for v in val)

    def _parse_selector(selector):
        if _is_range_selector(selector):
            t0 = float(selector[0])
            t1 = float(selector[1])
            if t0 <= t1:
                return {'mode': 'range', 't_min': t0, 't_max': t1}
            return {'mode': 'range', 't_min': t1, 't_max': t0}
        if _is_sequence(selector):
            raise ValueError(
                "Invalid nested selector in 'time_slice'. Use float for point slices "
                "or (t_min, t_max) for integrated ranges."
            )
        return {'mode': 'point', 'time': float(selector)}

    # Normalize selectors into per-file lists
    if not _is_sequence(time_slice):
        per_file_specs = [[_parse_selector(time_slice)] for _ in range(n_files)]
    else:
        time_slice_list = list(time_slice)

        # Single-file + list of selectors => overlay multiple selectors
        if n_files == 1 and len(time_slice_list) > 0 and not all(_is_sequence(v) and not _is_range_selector(v) for v in time_slice_list):
            per_file_specs = [[_parse_selector(v) for v in time_slice_list]]

        # Multi-file + one selector per file
        elif len(time_slice_list) == n_files and all(not (_is_sequence(v) and not _is_range_selector(v)) for v in time_slice_list):
            per_file_specs = [[_parse_selector(v)] for v in time_slice_list]

        # Multi-file + nested selector lists per file
        elif len(time_slice_list) == n_files and all(_is_sequence(v) and not _is_range_selector(v) for v in time_slice_list):
            per_file_specs = []
            for idx, specs_for_file in enumerate(time_slice_list):
                if len(specs_for_file) == 0:
                    raise ValueError(f"time_slice[{idx}] is empty; provide at least one selector")
                per_file_specs.append([_parse_selector(v) for v in specs_for_file])
        else:
            raise ValueError(
                "Invalid 'time_slice' format. Use scalar/range, single-file selector list, "
                "one-selector-per-file list (len == n_files), or nested selector list per file (len == n_files)."
            )

    plot_entries = []
    for file_idx, data in enumerate(data_list):
        for spec in per_file_specs[file_idx]:
            plot_entries.append({
                'file_idx': file_idx,
                'data': data,
                'spec': spec,
            })

    n_traces = len(plot_entries)

    same_temperature = len(set(d['temperature'] for d in data_list)) == 1

    # Labels
    if labels is None:
        labels_final = []
        for entry in plot_entries:
            d = entry['data']
            unit_text = _format_time_unit(d['time_unit'])
            spec = entry['spec']
            if spec['mode'] == 'point':
                if same_temperature:
                    labels_final.append(f"{spec['time']}{unit_text}")
                else:
                    labels_final.append(f"{d['temperature']} @ {spec['time']}{unit_text}")
            else:
                if same_temperature:
                    labels_final.append(f"{spec['t_min']}-{spec['t_max']}{unit_text}")
                else:
                    labels_final.append(f"{d['temperature']} @ {spec['t_min']}-{spec['t_max']}{unit_text}")
    else:
        if isinstance(labels, (list, tuple)) and len(labels) == n_traces:
            labels_final = list(labels)
        elif isinstance(labels, (list, tuple)) and len(labels) == n_files:
            # If one label per file is supplied, append the target time when needed.
            labels_final = []
            for entry in plot_entries:
                base_label = labels[entry['file_idx']]
                d = entry['data']
                unit_text = _format_time_unit(d['time_unit'])
                spec = entry['spec']
                if spec['mode'] == 'point':
                    labels_final.append(f"{base_label} @ {spec['time']}{unit_text}")
                else:
                    labels_final.append(f"{base_label} @ {spec['t_min']}-{spec['t_max']}{unit_text}")
        else:
            raise ValueError(
                f"Parameter 'labels' must have length {n_traces} (per trace) "
                f"or {n_files} (per file), got {len(labels) if isinstance(labels, (list, tuple)) else 'scalar'}"
            )

    # Linestyles
    if linestyles is None:
        linestyles_final = ['-'] * n_traces
    else:
        linestyles_final = _broadcast_param(linestyles, n_traces, 'linestyles')

    # Colors
    if colors is None:
        colors_final = sns.color_palette(palette, n_traces)
    else:
        colors_final = _broadcast_param(colors, n_traces, 'colors')
    
    # Create figure
    print(f"\nPlotting time-sliced spectra...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot each spectrum
    plotted_data = []
    for i, entry in enumerate(plot_entries):
        data = entry['data']
        time = data['time']
        intensity = data['intensity']
        wavelengths = data['wavelengths']
        spec = entry['spec']

        if spec['mode'] == 'point':
            target_time = spec['time']
            # Find closest time index
            time_idx = np.argmin(np.abs(time - target_time))
            actual_time = float(time[time_idx])
            # Extract spectrum at this time
            spectrum = intensity[time_idx, :]
            target_range = None
            actual_range = None
        else:
            t_min = spec['t_min']
            t_max = spec['t_max']
            range_mask = (time >= t_min) & (time <= t_max)
            if not np.any(range_mask):
                # Fallback to closest single point if range does not hit sampled times
                mid_time = 0.5 * (t_min + t_max)
                time_idx = np.argmin(np.abs(time - mid_time))
                actual_time = float(time[time_idx])
                spectrum = intensity[time_idx, :]
                target_range = (t_min, t_max)
                actual_range = (actual_time, actual_time)
            else:
                time_selected = time[range_mask]
                spectrum = np.sum(intensity[range_mask, :], axis=0)
                actual_time = None
                target_range = (t_min, t_max)
                actual_range = (float(time_selected.min()), float(time_selected.max()))
        
        # Normalize if requested
        if normalize:
            max_val = np.max(spectrum)
            if max_val > 0:
                spectrum = spectrum / max_val
        
        # Plot
        ax.plot(wavelengths, spectrum,
               label=labels_final[i],
               color=colors_final[i],
               linestyle=linestyles_final[i],
               linewidth=linewidth)
        
        # Annotate peak if requested (one annotation per plotted trace)
        if annotate_max and getattr(spectrum, 'size', 0) > 0 and np.any(np.isfinite(spectrum)):
            # If x-limits are applied, annotate the maximum within the visible window.
            if xlim is not None:
                x_lo, x_hi = float(xlim[0]), float(xlim[1])
                visible_mask = (wavelengths >= x_lo) & (wavelengths <= x_hi)
            else:
                x_lo, x_hi = float(np.min(wavelengths)), float(np.max(wavelengths))
                visible_mask = np.ones_like(wavelengths, dtype=bool)

            if np.any(visible_mask):
                spec_visible = spectrum[visible_mask]
                wl_visible = wavelengths[visible_mask]
                peak_idx = int(np.nanargmax(spec_visible))
                peak_wl = float(wl_visible[peak_idx])
                peak_int = float(spec_visible[peak_idx])
            else:
                peak_idx = int(np.nanargmax(spectrum))
                peak_wl = float(wavelengths[peak_idx])
                peak_int = float(spectrum[peak_idx])

            # Stagger text offsets and keep labels inside the plot area.
            if ylim is not None:
                y_lo, y_hi = float(ylim[0]), float(ylim[1])
            else:
                y_lo, y_hi = float(np.nanmin(spectrum)), float(np.nanmax(spectrum))

            near_right = peak_wl > (x_lo + 0.8 * (x_hi - x_lo))
            near_top = peak_int > (y_lo + 0.85 * (y_hi - y_lo))

            base_x = 5 + (i % 3) * 8
            base_y = 5 + (i % 4) * 6
            x_offset = -base_x if near_right else base_x
            y_offset = -base_y if near_top else base_y

            ax.annotate(f'{peak_wl:.1f} nm',
                        xy=(peak_wl, peak_int),
                        xytext=(x_offset, y_offset),
                        textcoords='offset points',
                        fontsize=8,
                        color=colors_final[i],
                        annotation_clip=True)
        
        plotted_data.append({
            'wavelengths': wavelengths,
            'spectrum': spectrum,
            'label': labels_final[i],
            'target_time': spec['time'] if spec['mode'] == 'point' else None,
            'target_range': target_range,
            'actual_time': actual_time,
            'actual_range': actual_range,
            'time_unit': data['time_unit'],
            'temperature': data['temperature'],
            'mode': spec['mode']
        })

        time_unit_display = _format_time_unit(data['time_unit'])
        if spec['mode'] == 'point':
            print(f"  {labels_final[i]}: target time={spec['time']}{time_unit_display}, "
                  f"actual time={actual_time:.3f}{time_unit_display}")
        else:
            if actual_range is not None and actual_range[0] == actual_range[1]:
                print(f"  {labels_final[i]}: target range={target_range[0]}-{target_range[1]}{time_unit_display}, "
                      f"no sampled points in window, fallback to {actual_range[0]:.3f}{time_unit_display}")
            else:
                print(f"  {labels_final[i]}: integrated range={target_range[0]}-{target_range[1]}{time_unit_display}, "
                      f"actual sampled range={actual_range[0]:.3f}-{actual_range[1]:.3f}{time_unit_display}")
    
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if title is None:
        title_text = 'Time-Sliced PL Spectra'
    else:
        title_text = title
    
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    
    # Set axis limits if specified
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add legend
    ax.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    
    if outfile is None:
        # Auto-generate filename
        if n_files == 1:
            time_unit_file = data_list[0]['time_unit']  # Use raw unit for filename
            first_spec = per_file_specs[0][0]
            if len(per_file_specs[0]) == 1 and first_spec['mode'] == 'point':
                base_name = f"{data_list[0]['filepath'].stem}_spectrum_{first_spec['time']}{time_unit_file}"
            elif len(per_file_specs[0]) == 1 and first_spec['mode'] == 'range':
                base_name = (
                    f"{data_list[0]['filepath'].stem}_spectrum_{first_spec['t_min']}-"
                    f"{first_spec['t_max']}{time_unit_file}"
                )
            else:
                base_name = f"{data_list[0]['filepath'].stem}_spectra_multislice"
        else:
            temps = sorted(set(d['temperature'] for d in data_list))
            time_unit_file = data_list[0]['time_unit']  # Use raw unit for filename
            first_spec = per_file_specs[0][0]
            if first_spec['mode'] == 'point':
                time_str = f"{first_spec['time']}{time_unit_file}"
            else:
                time_str = f"{first_spec['t_min']}-{first_spec['t_max']}{time_unit_file}"
            base_name = f"spectra_{time_str}_{'_'.join(temps)}"
    else:
        base_name = outfile
    
    output_filename = f"{base_name}.{filetype}"
    output_path = outdir_path / output_filename
    
    # Check if file exists
    if output_path.exists():
        response = input(f"File {output_path} already exists. Type Y to overwrite: ")
        if response.upper() != 'Y':
            print("Save cancelled.")
            plt.close(fig)
            return None
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    return {
        'fig': fig,
        'data': plotted_data,
        'output_path': output_path
    }


def plot_2d_heatmap(
    filepath,
    cmap='viridis',
    log_scale=False,
    wavelength_range=None,
    wavelength_center=None,
    wavelength_span=120.0,
    time_range=None,
    title=None,
    xlabel=r'Wavelength (nm)',
    ylabel=None,
    outfile=None,
    filetype="png",
    outdir="Plot_Spectra",
    figsize=(10, 6),
    colorbar_label='Photon Counts'
):
    """
    Plot 2D heatmap of time-resolved photoluminescence data.
    
    Creates a color-coded intensity map with time on the y-axis and wavelength
    on the x-axis.
    
    Parameters
    ----------
    filepath : str or Path
        Path to streak camera .txt file
    cmap : str, optional
        Matplotlib colormap name (default: 'viridis')
    log_scale : bool, optional
        Use logarithmic color scale (default: False)
    wavelength_range : tuple, optional
        (min, max) wavelength range to display. If None, shows full range
    wavelength_center : float or str, optional
        Center wavelength in nm for windowed plotting. If set to "peak", the
        center is chosen automatically from the wavelength of maximum integrated
        intensity. Mutually exclusive with wavelength_range.
    wavelength_span : float, optional
        Width of the wavelength window (in nm) when wavelength_center is used.
        Default is 120.0 nm.
    time_range : tuple, optional
        (min, max) time range to display. If None, shows full range
    title : str, optional
        Custom plot title
    xlabel : str, optional
        Custom x-axis label
    ylabel : str, optional
        Custom y-axis label
    outfile : str, optional
        Output filename without extension. If None, uses input filename
    filetype : str, optional
        Output file type: "png" or "pdf" (default: "png")
    outdir : str, optional
        Output directory (default: "Plot_Spectra")
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10, 6))
    colorbar_label : str, optional
        Label for the colorbar (default: 'Photon Counts')
        
    Returns
    -------
    dict
        Dictionary with 'fig', 'ax', 'mesh', and 'output_path' keys
    """
    filepath = Path(filepath)
    
    # Load data
    print(f"Loading streak camera file: {filepath.name}")
    data = _load_streak_file(filepath)
    
    time = data['time']
    wavelengths = data['wavelengths']
    intensity = data['intensity']
    time_unit = data['time_unit']
    temperature = data['temperature']
    time_window = data['time_window']
    
    print(f"  Loaded: {temperature}, {time_window}")
    print(f"  Time range: {time.min():.3f} to {time.max():.3f} {time_unit}")
    print(f"  Wavelength range: {wavelengths.min():.1f} to {wavelengths.max():.1f} nm")
    print(f"  Data shape: {intensity.shape} (time x wavelength)")
    
    # Apply range filters
    time_mask = np.ones(len(time), dtype=bool)
    wl_mask = np.ones(len(wavelengths), dtype=bool)
    
    if time_range is not None:
        t_min, t_max = time_range
        time_mask = (time >= t_min) & (time <= t_max)
        print(f"  Filtering time range: {t_min} to {t_max} {time_unit}")
    
    if wavelength_range is not None and wavelength_center is not None:
        raise ValueError("Use either wavelength_range or wavelength_center, not both.")

    if wavelength_center is not None:
        if wavelength_span is None or wavelength_span <= 0:
            raise ValueError("wavelength_span must be a positive number when wavelength_center is used.")

        if isinstance(wavelength_center, str) and wavelength_center.lower() == "peak":
            # Determine peak center after any time filtering is applied.
            intensity_for_peak = intensity[time_mask, :]
            integrated_spectrum = np.sum(intensity_for_peak, axis=0)
            peak_idx = int(np.argmax(integrated_spectrum))
            center_nm = float(wavelengths[peak_idx])
            print(f"  Auto-selected wavelength center (peak): {center_nm:.2f} nm")
        else:
            center_nm = float(wavelength_center)

        half_span = float(wavelength_span) / 2.0
        wl_min = max(float(wavelengths.min()), center_nm - half_span)
        wl_max = min(float(wavelengths.max()), center_nm + half_span)
        wl_mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        print(
            f"  Filtering wavelength window centered at {center_nm:.2f} nm "
            f"(span {float(wavelength_span):.1f} nm): {wl_min:.2f} to {wl_max:.2f} nm"
        )

    elif wavelength_range is not None:
        wl_min, wl_max = wavelength_range
        wl_mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        print(f"  Filtering wavelength range: {wl_min} to {wl_max} nm")
    
    # Filter data
    time_filtered = time[time_mask]
    wavelengths_filtered = wavelengths[wl_mask]
    intensity_filtered = intensity[time_mask, :][:, wl_mask]
    
    # Create figure
    print("\nCreating 2D heatmap...")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Apply log scale to intensity if requested
    if log_scale:
        # Add small offset to avoid log(0)
        plot_data = np.log10(intensity_filtered + 1)
        if colorbar_label == 'Photon Counts':
            colorbar_label = 'log10(Photon Counts)'
    else:
        plot_data = intensity_filtered
    
    # Create mesh plot
    # pcolormesh expects edges, so we need to create bin edges
    # For simplicity, we'll use imshow with proper extent
    extent = [
        wavelengths_filtered.min(),
        wavelengths_filtered.max(),
        time_filtered.min(),
        time_filtered.max()
    ]
    
    mesh = ax.imshow(
        plot_data,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap=cmap,
        interpolation='nearest'
    )
    
    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax, label=colorbar_label)
    cbar.ax.tick_params(labelsize=10)
    
    # Set labels and title
    time_unit_display = _format_time_unit(time_unit)
    if ylabel is None:
        ylabel_text = f'Time ({time_unit_display})'
    else:
        ylabel_text = ylabel
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel_text, fontsize=12)
    
    if title is None:
        title_text = f'Time-Resolved PL: {temperature} ({time_window})'
    else:
        title_text = title
    
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    
    if outfile is None:
        base_name = f"{filepath.stem}_heatmap"
    else:
        base_name = outfile
    
    output_filename = f"{base_name}.{filetype}"
    output_path = outdir_path / output_filename
    
    # Check if file exists
    if output_path.exists():
        response = input(f"File {output_path} already exists. Type Y to overwrite: ")
        if response.upper() != 'Y':
            print("Save cancelled.")
            plt.close(fig)
            return None
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    return {
        'fig': fig,
        'ax': ax,
        'mesh': mesh,
        'output_path': output_path,
        'data': {
            'time': time_filtered,
            'wavelengths': wavelengths_filtered,
            'intensity': intensity_filtered
        }
    }


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Define the data directory
    data_dir = Path("DMAC-s2PFL-PN/260228streak")
    
    # Example 1: Plot transient lifetimes for all temperatures at 200ns window
    # -------------------------------------------------------------------------
    files_200ns = [
         data_dir / "15wt%DMAC-s2PFL-PN DPEPO 200ns 10K.txt",
         data_dir / "15wt%DMAC-s2PFL-PN DPEPO 200ns 50K.txt",
         data_dir / "15wt%DMAC-s2PFL-PN DPEPO 200ns 100K.txt",
         data_dir / "15wt%DMAC-s2PFL-PN DPEPO 200ns 150K.txt",
         data_dir / "15wt%DMAC-s2PFL-PN DPEPO 200ns 200K.txt",
         data_dir / "15wt%DMAC-s2PFL-PN DPEPO 200ns 250K.txt",
         data_dir / "15wt%DMAC-s2PFL-PN DPEPO 200ns 300K.txt",
     ]
     
    results_200ns = plot_transient_lifetimes(
         files_200ns,
         log_scale=True,
         normalize=True,
         outfile="DMAC-s2PFL-PN_streak_200ns_norm",
         palette="icefire",
         xlim=(5, 200)
     )
    
    # Example 2: Plot transient lifetimes for all temperatures at 20us window
    # ------------------------------------------------------------------------
    files_20us = [
         data_dir / "15wt%DMAC-s2PFL-PN DPEPO 20us 10K.txt",
         data_dir / "15wt%DMAC-s2PFL-PN DPEPO 20us 50K.txt",
         data_dir / "15wt%DMAC-s2PFL-PN DPEPO 20us 100K.txt",
         data_dir / "15wt%DMAC-s2PFL-PN DPEPO 20us 150K.txt",
         data_dir / "15wt%DMAC-s2PFL-PN DPEPO 20us 200K.txt",
         data_dir / "15wt%DMAC-s2PFL-PN DPEPO 20us 250K.txt",
         data_dir / "15wt%DMAC-s2PFL-PN DPEPO 20us 300K.txt",
    ]
    # 
    results_20us = plot_transient_lifetimes(
        files_20us,
        log_scale=True,
        normalize=True,
        outfile="DMAC-s2PFL-PN_streak_20us_norm",
        palette="icefire",
        xlim=(0.2, 20)
    )
    
        # Time-sliced spectra and heatmap examples are intentionally moved to
        # standalone runner scripts so they can be executed independently:
        #   - run_streak_timesliced.py
        #   - run_streak_heatmap.py
    
    # Example 5: Mixed time windows (auto-grouped)
    # ---------------------------------------------
    # all_files = files_200ns + files_20us
    # 
    # results_all = plot_transient_lifetimes(
    #     all_files,
    #     log_scale=False,
    #     normalize=True,
    #     outfile="DMAC-s2PFL-PN_all_transients"
    # )
    
    # print("\nNo example code executed. Uncomment examples in the script to run.")
