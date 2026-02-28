import numpy as np
import polars as pl
import io
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


def phase_correlation_shift(x, y, eps=1e-12):
    x, y = clean_signal(x), clean_signal(y)
    n = next_pow2(len(x) + len(y))
    X = np.fft.fft(x, n=n)
    Y = np.fft.fft(y, n=n)
    R = X * np.conj(Y)
    R /= np.abs(R) + eps
    cc = np.fft.ifft(R)
    cc = np.real(cc)
    shift = np.argmax(cc)
    if shift > n // 2:
        shift -= n
    return shift, cc
