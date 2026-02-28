# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair==6.0.0",
#     "marimo>=0.20.2",
#     "matplotlib>=3.10.8",
#     "numpy>=2.2.6",
#     "polars>=1.38.1",
#     "pyarrow>=23.0.1",
#     "pyzmq>=27.1.0",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.9.0.post1",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import pandas as pd
    import numpy as np
    import altair as alt
    alt.data_transformers.enable("marimo")
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from pathlib import Path
    import io

    alt.themes.enable('fivethirtyeight')
    return Path, alt, io, mcolors, mo, np, pd, pl, plt


@app.cell
def _(io, np, pl):
    from scipy.ndimage import median_filter, gaussian_filter1d


    def parse_polysomes(file_info):
        """Parse a fractionator CSV file from a marimo FileInfo object."""
        content = file_info.contents  # bytes
        text_lines = content.decode("utf-8").splitlines()

        # Find the header row (same logic as original)
        skip = None
        for i, line in enumerate(text_lines):
            if "Data Columns" in line:
                skip = i + 1
                break
        if skip is None:
            return pl.DataFrame()

        df = pl.read_csv(io.BytesIO(content), separator=",", skip_rows=skip)

        # Strip whitespace from string columns and column names
        df = df.with_columns(
            [
                pl.col(c).str.replace_all(" ", "")
                for c in df.columns
                if df[c].dtype == pl.Utf8
            ]
        )
        df = df.rename({c: c.replace(" ", "") for c in df.columns})

        # Cast everything possible to Float32
        num_cols = []
        for c in df.columns:
            try:
                df = df.with_columns(pl.col(c).cast(pl.Float32))
                num_cols.append(c)
            except Exception:
                pass

        df = df.select(num_cols).with_columns(
            pl.lit(file_info.name).alias("file_name"),
            pl.Series(np.arange(0, len(df))).alias("position_index"),
        )
        return df


    def clean_signal(signal, downsample=1, demean=True):
        signal = median_filter(signal[::downsample], size=26 // downsample)
        signal = gaussian_filter1d(signal, sigma=20 // downsample)
        if demean:
            signal -= signal.mean()
        return signal


    def next_pow2(n):
        return 1 << (n - 1).bit_length()


    def fft_cross_correlation_shift(x, y):
        x, y = clean_signal(x), clean_signal(y)
        n = next_pow2(len(x) + len(y))
        X = np.fft.rfft(x, n=n)
        Y = np.fft.rfft(y, n=n)
        cc = np.real(np.fft.irfft(X * np.conj(Y), n=n))
        shift = int(np.argmax(cc))
        if shift > n // 2:
            shift -= n
        return shift, cc

    return clean_signal, fft_cross_correlation_shift, parse_polysomes


@app.cell
def _(mo):
    mo.md("""
    # Fractionator output nice-ifier
    """)
    return


@app.cell
def _(mo):
    file_upload = mo.ui.file(
        label="Upload fractionator output (.csv files)",
        multiple=True,
        kind="area",
    )
    file_upload
    return (file_upload,)


@app.cell
def _(Path, file_upload, mcolors, mo, parse_polysomes, pl, plt):
    """Parse files and build initial metadata table."""
    mo.stop(not file_upload.value, mo.md("*Upload one or more CSV files to begin.*"))

    raw_profiles = [parse_polysomes(f) for f in file_upload.value]

    # Build colour palette
    n_files = len(file_upload.value)
    cmap = plt.colormaps["tab10"]
    palette = [mcolors.rgb2hex(cmap(i)) for i in range(n_files)]
    fnames = [f.name for f in file_upload.value]
    flabs  = [Path(f.name).stem for f in file_upload.value]

    meta_init = pl.DataFrame({
        "file_name": fnames,
        "label":     flabs,
        "color":     palette,
    })
    return meta_init, raw_profiles


@app.cell
def _(meta_init, mo):
    sample_table = mo.ui.data_editor(
        data=meta_init.to_pandas(), 
        label="Edit sample metadata",
        editable_columns=['label', 'color']
    )
    sample_table
    return (sample_table,)


@app.cell
def _(mo, pl, sample_table):
    switches = mo.ui.array([
        mo.ui.switch(label=label, value=True)
        for label in pl.from_pandas(sample_table.value)["label"].to_list()
    ])
    return (switches,)


@app.cell
def _(mo):
    mo.md("""
    ## Controls
    """)
    return


@app.cell
def _(mo, sample_table, switches):
    selected_labels = [
        label
        for label, active in zip(sample_table.value["label"], switches.value)
        if active
    ]

    switch_suite = mo.vstack(switches)
    return selected_labels, switch_suite


@app.cell
def _(combined_raw, mo, switch_suite):
    slice_range = mo.ui.range_slider(
        start=0, stop=combined_raw['position_index'].max(), 
        value=[3000, 5000], label="Select range for alignment",
        debounce=True
    )
    raw_controls = mo.vstack([mo.md('Toggle samples to viz'), switch_suite, slice_range])
    return raw_controls, slice_range


@app.cell
def _(pd, pl, raw_profiles, sample_table, selected_labels):
    _meta_df = pl.from_pandas(pd.DataFrame(sample_table.value))
    _all_raw = pl.concat(raw_profiles)
    combined_raw = _all_raw.join(_meta_df, on="file_name")
    label_colors = dict(zip(_meta_df["label"], _meta_df["color"]))
    label_colors = {k:v for k,v in label_colors.items() if k in selected_labels}
    return combined_raw, label_colors


@app.cell
def _(combined_processed, mo):
    y_metric_select = mo.ui.dropdown(
        options=[
            "smoothed_absorbance",
            "zeroed_absorbance",
            "AUC_norm_smoothed_absorbance",
            "Absorbance",
        ],
        value="smoothed_absorbance",
        label="Y Axis",
    )
    processed_x_range = mo.ui.range_slider(
        start=0, stop=combined_processed['relative_position'].max(), 
        value=[0, combined_processed['relative_position'].max()], label="Select positions range",
        debounce=True
    )
    processed_y_range = mo.ui.range_slider(
        start=0, stop=100, value=[0, 100], label="Select y axis range",
        debounce=True
    )
    line_width = mo.ui.slider(
        start=0.25, stop=3, value=1.5, step=0.25, label="Line width",
        debounce=True
    )
    line_alpha = mo.ui.slider(
        start=0.0, stop=1.0, value=1.0, step=0.05, label="Line alpha",
        debounce=True
    )
    processed_controls = mo.vstack([
        y_metric_select, processed_x_range, processed_y_range,
        line_width, line_alpha,
    ])
    return (
        line_alpha,
        line_width,
        processed_controls,
        processed_x_range,
        processed_y_range,
        y_metric_select,
    )


@app.cell
def _(
    alt,
    combined_raw,
    label_colors,
    mo,
    np,
    pl,
    raw_controls,
    selected_labels,
    slice_range,
):
    mo.md("## Raw profiles")

    start, end = slice_range.value
    _rect = pl.DataFrame({
        "xmin": [start], "xmax": [end],
        "ymin": [-np.inf], "ymax": [np.inf],
    })

    _lines = (
        alt.Chart(combined_raw.filter(pl.col("label").is_in(selected_labels)).to_pandas())
        .mark_line()
        .encode(
            x=alt.X("position_index:Q", title="Position index"),
            y=alt.Y("Absorbance:Q", title="Absorbance"),
            color=alt.Color(
                "label:N",
                title="Sample",
                scale=alt.Scale(
                    domain=list(label_colors.keys()),
                    range=list(label_colors.values()),
                ),
            ),
        )
    )

    _rect = (
        alt.Chart(pl.DataFrame({
            "xmin": [start], "xmax": [end],
            "ymin": [combined_raw["Absorbance"].min()],
            "ymax": [combined_raw["Absorbance"].max()],
        }).to_pandas())
        .mark_rect(fill="orange", opacity=0.2)
        .encode(
            x=alt.X("xmin:Q"),
            x2=alt.X2("xmax:Q"),
            y=alt.Y("ymin:Q"),
            y2=alt.Y2("ymax:Q"),
        )
    )

    p_base = (_rect + _lines).properties(title="Raw absorbance profiles", width="container")
    mo.hstack([raw_controls, p_base], widths=[1, 3])
    return


@app.cell
def _(
    clean_signal,
    fft_cross_correlation_shift,
    mo,
    np,
    pd,
    pl,
    raw_profiles,
    sample_table,
    slice_range,
):
    _start, _end = slice_range.value
    _ref = raw_profiles[0]["Absorbance"].to_numpy()

    processed_profiles = []
    _max_shift = 0

    for _profile in raw_profiles:
        _abs = _profile["Absorbance"].to_numpy()
        _shift, _cc = fft_cross_correlation_shift(
            _ref[_start:_end], _abs[_start:_end]
        )
        _max_shift = max(_max_shift, abs(_shift))
        _rel_pos    = np.arange(len(_abs)) + _shift
        _zeroed     = _abs - _abs.min()
        _smoothed   = clean_signal(_zeroed, demean=False)

        processed_profiles.append(
            _profile.with_columns(
                pl.Series("zeroed_absorbance",            _zeroed),
                pl.Series("smoothed_absorbance",          _smoothed),
                pl.Series("relative_position",            _rel_pos.astype(int)),
                pl.Series("AUC_norm_smoothed_absorbance", _smoothed / _smoothed.sum()),
            )
        )

    _meta_df2 = pl.from_pandas(pd.DataFrame(sample_table.value))
    _all_proc = pl.concat(processed_profiles)
    combined_processed = _all_proc.join(_meta_df2, on="file_name")

    mm_shift = combined_processed.select(pl.max('Position(mm)', 'position_index')).to_numpy()
    mm_shift = mm_shift[0, 0] / mm_shift[0, 1]

    mo.callout(
        mo.md(f"**Maximum calculated shift:** {_max_shift} positions (about {mm_shift*_max_shift:.02} mm)"),
        kind="info",
    )
    return (combined_processed,)


@app.cell
def _(
    alt,
    combined_processed,
    label_colors,
    line_alpha,
    line_width,
    mo,
    pl,
    processed_controls,
    processed_x_range,
    processed_y_range,
    selected_labels,
    y_metric_select,
):
    mo.md("## Processed profiles")

    _x_start, _x_end = processed_x_range.value
    _y_start, _y_end = processed_y_range.value
    _metric = y_metric_select.value

    _df_melted = combined_processed.unpivot(
        index=["relative_position", "label"],
        on=["smoothed_absorbance", "zeroed_absorbance",
            "AUC_norm_smoothed_absorbance", "Absorbance"],
        variable_name="y_metric",
        value_name="y_value",
    )

    _df_plot = (
        _df_melted
        .filter(
            pl.col('label').is_in(selected_labels),
            pl.col("y_metric") == _metric,
            pl.col("relative_position") > _x_start,
            pl.col("relative_position") < _x_end,
        )
        .with_columns(
            (100 * pl.col("y_value") / pl.col("y_value").max())
            .clip(_y_start, _y_end)
            .alias("y_value")
        )
    )

    processed_df_obj   = _df_plot

    _p_proc = (
        alt.Chart(_df_plot.to_pandas())
        .mark_line(strokeWidth=line_width.value, opacity=line_alpha.value)
        .encode(
            x=alt.X("relative_position:Q", title="Position", axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("y_value:Q", title=_metric),
            color=alt.Color(
                "label:N",
                title="Sample",
                scale=alt.Scale(
                    domain=list(label_colors.keys()),
                    range=list(label_colors.values()),
                ),
            ),
        )
        .properties(title="Nicer plot", width="container")
    )

    mo.hstack([processed_controls, _p_proc], widths=[1, 3])
    return (processed_df_obj,)


@app.cell
def _(io, mo, pl, processed_df_obj):
    # --- TSV download ---
    def _make_tsv():
        _df = processed_df_obj
        _max_pos = int(_df["relative_position"].max())
        _grid = pl.DataFrame({"relative_position": list(range(_max_pos + 1))})
        _wide = (
            _df.drop("y_metric")
            .pivot(on=["label"], index=["relative_position"], values=["y_value"])
            .join(_grid, on="relative_position", how="right")
            .sort("relative_position")
            .select([pl.col('relative_position'), pl.all().exclude('relative_position')])
        )
        _buf = io.StringIO()
        _wide.write_csv(_buf, separator="\t")
        return _buf.getvalue().encode()

    _dl_tsv = mo.download(
        data=_make_tsv,
        filename="processed_data.tsv",
        label="Download prism-ready table (.tsv)",
        mimetype="text/tab-separated-values",
    )

    mo.hstack([_dl_tsv])
    return


if __name__ == "__main__":
    app.run()
