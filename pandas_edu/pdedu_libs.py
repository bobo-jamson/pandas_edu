import pandas as pd
import typing

def _select_cols(
    df: pd.DataFrame,
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[str]:
    """selects dataframe columns based on include/exclude criteria

    Args:
        df (pd.DataFrame): source dataframe
        include (list[str] | None): list of columns to include
        exclude (list[str] | None): list of columns to exclude

    Returns:
        list[str]: list of columns selected
    """
    # start with all columns
    cols = df.columns.to_list()
    # limit using include columns if specified
    if include is not None:
        cols = include
    if exclude is not None:
        cols = [col for col in cols if col not in exclude]
    return cols

def _describe_stats(
        df: pd.DataFrame
) -> pd.DataFrame:
    """return a dataframe with aggregated statistics: mean, std, min, max, skew, kurtosis

    Args:
        df (pd.DataFrame): source dataframe

    Returns:
        pd.DataFrame: new dataframe with aggregated statistics where original columns are on the index
    """
    df_desc = df.agg(["mean", "std", "min", "max", "skew", "kurtosis"]).T
    return df_desc

def _describe_count_stats(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """return a dataframe with counting stats of the dataframe: count, nunique

    Args:
        df (pd.DataFrame): source dataframe

    Returns:
        pd.DataFrame: new transposed dataframe with aggregated stats where original columns are on the index
    """
    df_desc = df.agg(["count", "nunique"]).T
    return df_desc

def _describe_percentiles(
        df: pd.DataFrame,
        percentiles: list[float],
) -> pd.DataFrame:
    """return a dataframe with the percentile values of each column based on `percentiles`

    Args:
        df (pd.DataFrame): source dataframe
        percentiles (list[float]): list of percentiles to calculate, range is [0,1]

    Returns:
        pd.DataFrame: new transposed dataframe with percentiles calculated for each column, the original columns are on the index
    """
    df_perc = df.quantile(q=percentiles).T
    return df_perc

def _describe_dtype(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """return a dataframe with the dtype of each column specified

    Args:
        df (pd.DataFrame): source dataframe

    Returns:
        pd.DataFrame: new transposed dataframe with dtypes listed, the original columns are on the index
    """
    return df.dtypes.rename("dtype").to_frame()

def _describe_missing(
        df: pd.DataFrame
) -> pd.DataFrame:
    """return a dataframe with missing counts and fraction, columns are on the index

    Args:
        df (pd.DataFrame): source dataframe

    Returns:
        pd.DataFrame: new transposed dataframe with missing stats listed, the original columns are on the index
    """
    df_missing = (df
                  .isnull()
                  .sum()
                  .rename("missing_n")
                  .to_frame()
                  .join(
                      df.isnull().mean().rename("missing_frac")
                      )
    )
    return df_missing

def _describe_top_freq(
        df: pd.DataFrame,
        dropna: bool,
) -> pd.DataFrame:
    """returns a dataframe with the most frequent value and fraction

    Args:
        df (pd.DataFrame): source dataframe
        dropna (bool): if `True` `NaN` values are excluded from most frequent

    Returns:
        pd.DataFrame: new transposed dataframe with top stats listed, the original columns are on the index
    """
    top_freq = []
    for col in df.columns:
        df_tmp = (df.loc[:, col]
                .value_counts(dropna=dropna)
                .nlargest(1)
                .rename("freq")
                .to_frame()
                .reset_index()
                .rename(columns={col:"top"})
                .assign(
                    column=col,
                    freq_frac=lambda d: d["freq"].div(df.shape[0]),
                )
                )
        # df_tmp.index.name = None
        top_freq.append(df_tmp)
    df_top_freq = pd.concat(top_freq).set_index("column").rename_axis(None, axis=0)
    return df_top_freq


def _summarize_unique(
    df: pd.DataFrame,
    dropna: bool,
    excl_nan_str: str,
) -> pd.DataFrame:
    """generates a summary of unique values for `df`

    Args:
        df (pd.DataFrame): source dataframe
        dropna (bool): if `True` NaN values will not be counted in the unique count
        excl_nan_str (str): used for generating the column name

    Returns:
        pd.DataFrame: dataframe with the uniqueness summarized
    """
    df_summary = (
        df
        # n-unique values
        .nunique(dropna=dropna)
        .to_frame()
        .assign(uniq_perc=lambda d: d.loc[:, 0].div(df.shape[0]).mul(100).round(2))
        .rename(columns={"index": "feature", 0: f"uniq_n_{excl_nan_str}_nan"})
    )
    return df_summary


def _summarize_nans(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """generates a summary about the presence, count and fraction of NaN values.

    Args:
        df (pd.DataFrame): source dataframe

    Returns:
        pd.DataFrame: dataframe with a summary of NaN values
    """
    df_summary = (
        df.isna()
        .sum()
        .rename("nan_n")
        .to_frame()
        .assign(
            nan_present=lambda d: d.iloc[:, 0].gt(0),
            nan_frac=lambda d: d.iloc[:, 0].div(df.shape[0]).mul(100).round(2),
        )
        .iloc[:, [-2, -3, -1]]
    )
    return df_summary


def _summarize_dtypes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """generates a summary about the datatypes within each column of the dataframe passed

    Args:
        df (pd.DataFrame): source dataframe

    Returns:
        pd.DataFrame: dataframe with a summary of datatypes used
    """
    # basic dtype report
    summary_df = df.dtypes.rename("dtype").to_frame()
    # temporary
    df_tmp = df.map(lambda x: type(x).__name__).apply(
        lambda s: s.value_counts(normalize=True).mul(100).round(2)
    )
    # number of dtypes
    df_dtype_n = df_tmp.apply(lambda s: s.dropna().count()).rename("dtype_n").to_frame()
    # percentage alloted
    df_dtype_perc = (
        df_tmp.apply(lambda s: s.dropna().to_dict()).rename("dtype_perc").to_frame()
    )
    # combine results
    summary_df = summary_df.join(df_dtype_n).join(df_dtype_perc)
    return summary_df


def _summarize_top_n(
    df: pd.DataFrame,
    dropna: bool,
    top_n: int,
) -> pd.DataFrame:
    """generates a summary of the N most frequent entries in each column

    Args:
        df (pd.DataFrame): source dataframe
        dropna (bool): if `True` NaN values will count as a unique entry
        top_n (int): depth of the rankings to return

    Returns:
        pd.DataFrame: dataframe with a summary of most frequent values
    """
    summary_df = (
        df.apply(
            lambda s: (
                s.value_counts(dropna=dropna, normalize=True)
                .mul(100)
                .round(1)
                .nlargest(top_n)
                .to_dict()
            )
        )
        .rename(f"top_{top_n}")
        .to_frame()
    )
    return summary_df


def _optional_fillna(
    df: pd.DataFrame,
    value: typing.Any,
    fill_bool: bool,
) -> pd.DataFrame:
    """a fillna function that can _optionally_ fillna values, to be used with `.pipe()`

    Args:
        df (pd.DataFrame): source dataframe
        value (typing.Any): value to fill NaNs with
        fill_bool (bool): if `True` NaNs will be filled otherwise nothing happens

    Returns:
        pd.DataFrame: dataframe with NaN values optionally filled
    """
    if fill_bool:
        return df.fillna(value)
    else:
        return df


def _grouper(
    df: pd.DataFrame,
    by: str | list[str] | tuple[str] | None,
    sort: bool = True,
    dropna: bool = True,
) -> pd.DataFrame:
    """helper function that returns a dataframe or a groupby transformed dataframe

    Args:
        df (pd.DataFrame): source dataframe
        by (str | list[str] | tuple[str] | None): column or columns to groupby on, if `None` no grouping
        sort (bool, optional): toggle to sort the groupings. Defaults to True.
        dropna (bool, optional): if `True` NaN is included in groups. Defaults to True.

    Returns:
        pd.DataFrame: dataframe or groupby dataframe
    """
    if by is None:
        new_df = df
    else:
        new_df = df.groupby(by=by, sort=sort, dropna=dropna)
    return new_df


def _stats_val_count(
    df: pd.DataFrame,
    col: str,
    name: str,
    sort: bool = True,
    ascending: bool = False,
    dropna: bool = True,
    normalize: bool = False,
) -> pd.DataFrame:
    """generates `df[col].value_counts()` with any user specified changes

    Args:
        df (pd.DataFrame): source dataframe
        col (str): column on which to perform the value_counts on
        name (str): what to name the resulting statistic
        sort (bool, optional): if `True` sort the values. Defaults to True.
        ascending (bool, optional): toggle for which way values are sorted. Defaults to False.
        dropna (bool, optional): if `True` NaN values are counted. Defaults to True.
        normalize (bool, optional): normalize to the total. Defaults to False.

    Returns:
        pd.DataFrame: a new dataframe
    """
    new_df = (
        df[col]
        .value_counts(
            sort=sort, dropna=dropna, ascending=ascending, normalize=normalize
        )
        .rename(name)
        .to_frame()
    )
    return new_df


def _stats_val_rank(
    df: pd.DataFrame,
    col: str,
    by: str | list[str] | tuple[str] | None,
    dropna: bool = True,
    method: str = "min",
    ascending: bool = False,
) -> pd.DataFrame:
    """generates a ranking column within the return dataframe based on value_counts()

    Args:
        df (pd.DataFrame): source dataframe
        col (str): column on which to perform value_counts on
        by (str | list[str] | tuple[str] | None): column or columns to groupby on, if `None` no grouping
        dropna (bool, optional): if `True` NaN values are counted. Defaults to True.
        method (str, optional): method to use for ranking, see `pd.DataFrame.rank()` for details. Defaults to "min".
        ascending (bool, optional): toggle for which way values are sorted. Defaults to False.

    Returns:
        pd.DataFrame: a new dataframe
    """
    if by is None:
        new_df = (
            df[col]
            .value_counts(dropna=dropna, ascending=ascending)
            .rank(method=method, ascending=ascending)
            .rename("rank")
            .to_frame()
            .astype(int)
        )
    else:
        new_df = (
            df[col]
            .value_counts(ascending=ascending, dropna=dropna)
            .rename("rank")
            .to_frame()
            .groupby(by=by, dropna=dropna)
            .rank(method=method, ascending=ascending)
            .astype(int)
        )
    return new_df


def _stats_val_cumfrac(
    df: pd.DataFrame,
    col: str,
    by: str | list[str] | tuple[str] | None,
    dropna: bool = True,
    ascending: bool = False,
) -> pd.DataFrame:
    """generates a column with the cumulative total of the fraction or percentage

    Args:
        df (pd.DataFrame): source dataframe
        col (str): column on which to perform value_counts on
        by (str | list[str] | tuple[str] | None): column or columns to groupby on, if `None` no grouping
        dropna (bool, optional): if `True` NaN values are counted. Defaults to True.
        ascending (bool, optional): toggle for which way values are sorted. Defaults to False.

    Returns:
        pd.DataFrame: a new dataframe
    """
    if by is None:
        new_df = (
            df[col]
            .value_counts(dropna=dropna, ascending=ascending, normalize=True)
            .rename("cumfrac")
            .cumsum()
            .to_frame()
        )
    else:
        new_df = (
            df[col]
            .value_counts(dropna=dropna, ascending=ascending, normalize=True)
            .rename("cumfrac")
            .to_frame()
            .groupby(by=by, dropna=dropna)
            .cumsum()
        )
    return new_df
