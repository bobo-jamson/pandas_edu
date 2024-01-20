import numpy as np
import pandas as pd
import pdedu_libs as helpers
import typing
import functools


@pd.api.extensions.register_dataframe_accessor("edu")
class EduDataFrame:
    def __init__(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """a collection of commonly used methods for exploratory data analysis
        grouped together under the `edu` accessor.

        Args:
            df (pd.DataFrame): source dataframe

        Returns:
            pd.DataFrame: new dataframe
        """
        self.df = df

    def concatenate_columns(
        self,
        new_col: str,
        cols: list[str],
        separator: str = "-",
        fill_na: str | bool = "auto",
    ) -> pd.DataFrame:
        """returns `df` with a `new_col` appended where `new_col` is `cols` converted to a string and
        joined together.

        Args:
            new_col (str): name of new column to append to end of dataframe
            cols (list[str]): columns to join together
            separator (str, optional): separator between values joined. Defaults to "-".
            fill_na (str | bool, optional): replace NaN values with this value, if `auto` the string representation of NaN will be used. Defaults to "auto".

        Returns:
            pd.DataFrame: new dataframe
        """
        df = self.df
        fill_bool = False
        if fill_na != "auto":
            fill_bool = True
        df[new_col] = (
            df.loc[:, cols]
            .pipe(helpers._optional_fillna, value=fill_na, fill_bool=fill_bool)
            .map(str)
            .agg(separator.join, axis="columns")
        )
        return df
    
    
    def describe_extended(
        self,
        percentiles: list[float] | None = None,
        include: list[str] | None  = None,
        exclude: list[str] | None = None,
        fillna: typing.Any | None = '-',
        dropna: bool = False,
    ) -> pd.DataFrame:
        """`.describe()` with more details, including most frequent values, nans stats, 
        dtype information, etc. 

        Args:
            percentiles (list[float] | None, optional): list of percentiles (0-1) to return, if None then [0.25, 0.50, 0.75] are returned. Defaults to None.
            include (list[str] | None, optional): columns to include, if `None` all columns are returned. Defaults to None.
            exclude (list[str] | None, optional): columns to exlcude, if `None` no columns are excluded. Defaults to None.
            fillna (typing.Any | None, optional): how to fill `NaN` values in to final dataframe. Defaults to '-'.
            dropna (bool, optional): if `True` count `NaN` in value counts. Defaults to False.

        Returns:
            pd.DataFrame: a new dataframe much like `.describe()`
        """
        percentiles = percentiles if percentiles is not None else [0.25, 0.50, 0.75]
        df = self.df
        cols = helpers._select_cols(df, include=include, exclude=exclude)
        df = df.loc[:, cols]
        describe = [] # where describe dataframes are stored 
        # process numerical selected columns
        df_num = df.select_dtypes(include="number")
        if df_num.shape[1] > 0:
            df_num_desc_stats = helpers._describe_stats(df=df_num)
            df_num_desc_count = helpers._describe_count_stats(df=df_num)
            df_num_desc_perc =  helpers._describe_percentiles(df=df_num, percentiles=percentiles)
            df_num_desc_top = helpers._describe_top_freq(df=df_num, dropna=dropna)
            df_num_desc_nan = helpers._describe_missing(df=df_num)
            df_num_desc_dtype = helpers._describe_dtype(df=df_num)
            df_num_desc = (df_num_desc_stats
                           .join(df_num_desc_count)
                           .join(df_num_desc_perc)
                           .join(df_num_desc_top)
                           .join(df_num_desc_nan)
                           .join(df_num_desc_dtype)
            )
            describe.append(df_num_desc)
        # process non-numerical
        df_cat = df.select_dtypes(exclude="number")
        if df_cat.shape[1] > 0:
            df_cat_desc_top = helpers._describe_top_freq(df=df_cat, dropna=dropna)
            df_cat_desc_count = helpers._describe_count_stats(df=df_cat)
            df_cat_desc_nan = helpers._describe_missing(df=df_cat)
            df_cat_desc_dtype = helpers._describe_dtype(df=df_cat)
            df_cat_desc = (df_cat_desc_top
                           .join(df_cat_desc_count)
                           .join(df_cat_desc_nan)
                           .join(df_cat_desc_dtype)
                           )
            describe.append(df_cat_desc)
        # combine, rename, order
        df_desc = pd.concat(describe, axis="rows")
        percentile_labels = [f"{q*100:0.0f}%" for q in percentiles]
        desc_col_order_ = [
            "count", "nunique", "top", 
            "freq", "freq_frac", 
            "missing_n", "missing_frac",
            "dtype",
            "mean", "std", 
            "min",
            *percentile_labels,
            "max",
            "skew", "kurtosis",
            ]
        desc_col_order = [col for col in desc_col_order_ if col in df_desc.columns.to_list()]
        df_desc = (
            df_desc
            .rename(columns=dict(list(zip(percentiles, percentile_labels))))
            .loc[cols, desc_col_order]
            .round(3)
            .dropna(axis="columns", how="all")
            .fillna(fillna)
        )
        return df_desc

    def sample_graded(
            self,
            n: int = 5,
            head_n: int = None,
            sample_n: int = None,
            tail_n: int = None,
            random_state: int = None, 
    ) -> pd.DataFrame:
        """returns a dataframe that is the combination of `.head()`, `.sample()` and `.tail()` in one command

        Args:
            n (int, optional): number of rows to sample at each level (head, sample, tail). Defaults to 5.
            head_n (int, optional): number of head rows, overrides `n`. Defaults to None.
            sample_n (int, optional): number of body rows, overrides `n`. Defaults to None.
            tail_n (int, optional): number of tail rows, overrides `n`. Defaults to None.
            random_state (int, optional): random seed used for sampling. Defaults to None.

        Returns:
            pd.DataFrame: a new dataframe with `.head()`, `sample()`, and `tail()` combined
        """
        df = self.df
        # logic around sampling
        if head_n is None:
            head_n = n
        if sample_n is None:
            sample_n = n
        if tail_n is None:
            tail_n = n
        # do the work
        df_sample = pd.concat(
            [
                df.head(head_n),
                # dummy/aesthetic spacer
                pd.DataFrame(data=[['...']*df.shape[1]], columns=df.columns.to_list(), index=[':']),
                df.sample(sample_n, random_state=random_state).sort_index(),
                # dummy/aesthetic spacer
                pd.DataFrame(data=[['...']*df.shape[1]], columns=df.columns.to_list(), index=[':']),
                df.tail(tail_n),
            ],
            axis="rows"
        )
        return df_sample


    def summarize_columns(
        self,
        include: list[str] | tuple[str] | None = None,
        exclude: list[str] | tuple[str] | None = None,
        dropna: bool = False,
        top_n: int = 3,
    ) -> pd.DataFrame:
        """Summarize each column with respect to unique values (counts, fraction, most N frequent),
        missing values (presence, counts, fraction), and datatypes present.

        Args:
            include (list[str] | tuple[str] | None, optional): columns to include, if `None` all columns included. Defaults to None.
            exclude (list[str] | tuple[str] | None, optional): columns to exclude. Defaults to None.
            dropna (bool, optional): if `True` NaNs and counts on `NaN` values are included. Defaults to False.
            top_n (int, optional): number of most frequent values to return. Defaults to 3.

        Returns:
            pd.DataFrame: New pandas dataframe with columns summarized.
        """
        ## define
        df = self.df
        ## helper string
        excl_nan_str = "excl" if dropna else "incl"
        # get columns
        cols = helpers._select_cols(df, include=include, exclude=exclude)
        # subset dataframe
        df = df.loc[:, cols]
        ## summarize unique
        df_summary_unique = df.loc[:, cols].pipe(
            helpers._summarize_unique, dropna=dropna, excl_nan_str=excl_nan_str
        )
        ## summarize nans
        df_summary_nan = df.loc[:, cols].pipe(helpers._summarize_nans)
        ## summarize dtypes
        df_summary_dtype = df.loc[:, cols].pipe(helpers._summarize_dtypes)
        ## summarize top-n features
        df_summary_top_n = df.loc[:, cols].pipe(
            helpers._summarize_top_n, dropna=dropna, top_n=top_n
        )
        ## combine results
        summary_df = (
            df
            # simple counts
            .count()
            .rename("count")
            .to_frame()
            # add details on unique values
            .join(df_summary_unique)
            # add details on nan values
            .join(df_summary_nan)
            # add dtype data
            .join(df_summary_dtype)
            # add top-n data
            .join(df_summary_top_n)
        )
        return summary_df

    def reduce_memory_size(
        self,
        cols: str | list[str] | None = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """convert each columns dtype to the smallest dtype capable of fitting the data

        Args:
            cols (str | list[str] | None, optional): column or list of columns to process, if `None` all columns are processed. Defaults to None.
            verbose (bool, optional): if `True` print the before and after memory footprint. Defaults to False.

        Returns:
            pd.DataFrame: dataframe with dtypes optimized for smallest memory footprint

        Note:
            this was taken from:
            https://gist.githubusercontent.com/BexTuychiev/4e34c55454c50c6fb1d0043d2848de6a/raw/f8af2217bdf3cb19881f068a9ba42ce67b1d6d8c/10206.py
        """
        df = self.df
        numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
        # get columns if not specified
        if cols is None:
            cols = df.columns
        # memory usage
        start_mem = df.memory_usage().sum() / 1024**2
        # do the work!
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        df[col] = df[col].astype(np.int64)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df[col] = df[col].astype(np.float16)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose:
            print(
                "Mem. usage decreased to from {:.2f} Mb to {:.2f} Mb ({:.1f}% reduction)".format(
                    start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem
                )
            )
        return df

    def try_convert(
        self,
        col: str,
        converter: typing.Callable,
        fallback: typing.Any = "auto",
    ) -> pd.DataFrame:
        """tries to convert values in `col` using `converter` returning `fallback` if
        `converter` raises a `ValueError` or `TypeError`.

        Leaving `fallback` as `auto` will keep the original value if an error is raised.

        Example:
        `df.edu.try_convert("col_with_nan", int, -9999)`
        this will convert all entries in `col_with_nan` to `int` and the `NaN` values,
        which cannot be converted to `int` will be replaced with `-9999`.

        `df.edu.try_convert("col_with_nan", int, "auto")`
        same as above, except `NaN` values which cannot be converted will be kept `NaN`.

        Args:
            col (str): column to convert
            converter (typing.Callable): a callable to run each value through, such as `int()`, `float()`, `str()`
            fallback (typing.Any, optional): a value to replace any failed values with, if `auto` keeps the original value. Defaults to "auto".

        Returns:
            pd.DataFrame: dataframe with the values possibly converted
        """

        def _helper(value, converter, fallback):
            if fallback == "auto":
                fallback = value
            try:
                return converter(value)
            except (ValueError, TypeError):
                return fallback

        _micro_convert = functools.partial(
            _helper, converter=converter, fallback=fallback
        )
        df = self.df
        df[col] = df[col].map(_micro_convert)
        return df

    def value_stats(
        self,
        col: str,
        groupby: str | list[str] | tuple[str] | None = None,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
        as_perc: bool = False,
        incl_rank: bool = True,
        incl_cumulative: bool = True,
    ) -> pd.DataFrame:
        """return counting metrics of `col` from the dataframe, much like
        `df['col_name'].value_counts()` but returns counts, fractions, rank,
        cumulative fraction and allows for grouping.

        Args:
            col (str): column to generate counting metrics on
            groupby (str | list[str] | tuple[str] | None, optional): apply `.groupby(by=$groupby)` to the dataframe. Defaults to None.
            sort (bool, optional): sort values by count. Defaults to True.
            ascending (bool, optional): sort values ascending. Defaults to False.
            dropna (bool, optional): include NaN in counts. Defaults to True.
            as_perc (bool, optional): scale fractions to percentages. Defaults to False.
            incl_rank (bool, optional): include a column with ranks. Defaults to True.
            incl_cumulative (bool, optional): include column with cumulative value. Defaults to True.

        Returns:
            pd.DataFrame: new dataframe with counting metrics
        """
        df = helpers._grouper(df=self.df, by=groupby, sort=sort, dropna=dropna)
        # counts
        summary_df = helpers._stats_val_count(
            df,
            col,
            "count",
            sort=sort,
            ascending=ascending,
            dropna=dropna,
            normalize=False,
        )
        if incl_rank:
            # rank
            df_rank = helpers._stats_val_rank(
                df, col, by=groupby, dropna=dropna, method="min", ascending=ascending
            )
            summary_df = summary_df.join(df_rank)
        # fraction or percent
        df_fp = helpers._stats_val_count(
            df,
            col,
            "frac",
            sort=sort,
            ascending=ascending,
            dropna=dropna,
            normalize=True,
        )
        if as_perc:
            # percent
            df_fp = df_fp.mul(100).round(2).rename(columns={"frac": "perc"})
        summary_df = summary_df.join(df_fp)
        # cumulative fraction/percent
        if incl_cumulative:
            df_cfrac = helpers._stats_val_cumfrac(
                df, col, by=groupby, ascending=ascending, dropna=dropna
            )
            if as_perc:
                df_cfrac = (
                    df_cfrac.mul(100).round(2).rename(columns={"cumfrac": "cumperc"})
                )
            summary_df = summary_df.join(df_cfrac)
        #
        return summary_df
