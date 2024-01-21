# Pandas [E]xploratory [D]ata [U]tilities (pandas_edu)

`pandas_edu` provides a new pandas dataframe accessor, `df.edu.`, 
with convenient methods for data exploration, the package is highly motivated by 
the excellent python package, [`py-janitor`](https://github.com/pyjanitor-devs/pyjanitor). 

## Quick Start

* Installation: `pip install pandas_edu`
* Usage:

```python

import pandas as pd
import pandas_edu

df = pd.read_csv(...)

df.edu.summarize_columns()
```

## Why `pandas_edu`?

Many of the methods provided by `pandas_edu` are nothing more than a series of 
pandas operations joined together to make a summary dataframe, by packaging them
into an accessor they are only a few keystrokes away. 

Encapsulating within the `edu` accessor is an opinionated API choice, informing
readers of code that the methods are not from the standard pandas namespace.

## Documentation

In lieu of a real documentation page, below are examples of the API. 

### `edu.concatenate_columns()`

Returns `df` with a `new_col` appended where `new_col` is `cols` converted to a string and joined together.

<details>
<summary>Function Docs</summary>

```raw
Args:
    new_col (str): name of new column to append to end of dataframe
    cols (list[str]): columns to join together
    separator (str, optional): separator between values joined. Defaults to "-".
    fill_na (str | bool, optional): replace NaN values with this value, 
        if `auto` the string representation of NaN will be used. Defaults to "auto".
Returns:
    pd.DataFrame: new dataframe
```

</details>

### `edu.describe_extended()`

`.describe()` with more details, including most frequent values, nans stats, dtype information, etc. 

<details>
<summary>Function Docs</summary>

```raw
Args:
    percentiles (list[float] | None, optional): list of percentiles (0-1) to return, 
            if None then [0.25, 0.50, 0.75] are returned. Defaults to None.
    include (list[str] | None, optional): columns to include, if `None` all columns 
            are returned. Defaults to None.
    exclude (list[str] | None, optional): columns to exlcude, if `None` no columns a
            re excluded. Defaults to None.
    fillna (typing.Any | None, optional): how to fill `NaN` values in to final 
            dataframe. Defaults to '-'.
    dropna (bool, optional): if `True` count `NaN` in value counts. Defaults to False.

Returns:
    pd.DataFrame: a new dataframe much like `.describe()`
```
</details>

### `edu.sample_graded()`

Returns a dataframe that is the combination of `.head()`, `.sample()` and `.tail()` in one command

<details>
    <summary>Function Docs</summary>

```raw
Args:
    n (int, optional): number of rows to sample at each level (head, sample, tail). Defaults to 5.
    head_n (int, optional): number of head rows, overrides `n`. Defaults to None.
    sample_n (int, optional): number of body rows, overrides `n`. Defaults to None.
    tail_n (int, optional): number of tail rows, overrides `n`. Defaults to None.
    random_state (int, optional): random seed used for sampling. Defaults to None.

Returns:
    pd.DataFrame: a new dataframe with `.head()`, `sample()`, and `tail()` combined
```
</details>


### `edu.summarize_columns()`

Summarize each column with respect to unique values (counts, fraction, most N frequent),
missing values (presence, counts, fraction), and datatypes present.

<details>
    <summary>Function Docs</summary>

```raw
Args:
    include (list[str] | tuple[str] | None, optional): columns to include, 
            if `None` all columns included. Defaults to None.
    exclude (list[str] | tuple[str] | None, optional): columns to exclude. 
            Defaults to None.
    dropna (bool, optional): if `True` NaNs and counts on `NaN` values are included. 
            Defaults to False.
    top_n (int, optional): number of most frequent values to return. Defaults to 3.

Returns:
    pd.DataFrame: New pandas dataframe with columns summarized.
```
</details>


### `edu.reduce_memory_size()`

Convert each columns dtype to the smallest dtype capable of fitting the data ([source](https://gist.githubusercontent.com/BexTuychiev/4e34c55454c50c6fb1d0043d2848de6a/raw/f8af2217bdf3cb19881f068a9ba42ce67b1d6d8c/10206.py))

<details>
    <summary>Function Docs</summary>

```raw
Args:
    cols (str | list[str] | None, optional): column or list of columns 
            to process, if `None` all columns are processed. Defaults to None.
    verbose (bool, optional): if `True` print the before and after 
            memory footprint. Defaults to False.

Returns:
    pd.DataFrame: dataframe with dtypes optimized for smallest memory footprint

Note:
    this was taken from:
    https://gist.githubusercontent.com/BexTuychiev/4e34c55454c50c6fb1d0043d2848de6a/raw/f8af2217bdf3cb19881f068a9ba42ce67b1d6d8c/10206.py
```
</details>


### `edu.try_convert()`

Tries to convert values in `col` using `converter` returning `fallback` if
`converter` raises a `ValueError` or `TypeError`.
Leaving `fallback` as `auto` will keep the original value if an error is raised.

Example:

`df.edu.try_convert("col_with_nan", int, -9999)`
this will convert all entries in `col_with_nan` to `int` and the `NaN` values,
which cannot be converted to `int` will be replaced with `-9999`.


`df.edu.try_convert("col_with_nan", int, "auto")`
same as above, except `NaN` values which cannot be converted will be kept `NaN`.

<details>
    <summary>Function Docs</summary>

```raw
Args:
    col (str): column to convert
    converter (typing.Callable): a callable to run each value through, 
            such as `int()`, `float()`, `str()`
    fallback (typing.Any, optional): a value to replace any failed values with, 
            if `auto` keeps the original value. Defaults to "auto".

Returns:
    pd.DataFrame: dataframe with the values possibly converted
```
</details>


### `edu.value_stats()`

Return counting metrics of `col` from the dataframe, much like
`df['col_name'].value_counts()` but returns counts, fractions, rank,
cumulative fraction and allows for grouping.

_note: I could not make the `edu` space available within grouped dataframes, so this method provides an internal `groupby=` keyword._

<details>
    <summary>Function Docs</summary>

```raw
Args:
    col (str): column to generate counting metrics on
    groupby (str | list[str] | tuple[str] | None, optional): 
            apply `.groupby(by=$groupby)` to the dataframe. Defaults to None.
    sort (bool, optional): sort values by count. Defaults to True.
    ascending (bool, optional): sort values ascending. Defaults to False.
    dropna (bool, optional): include NaN in counts. Defaults to True.
    as_perc (bool, optional): scale fractions to percentages. Defaults to False.
    incl_rank (bool, optional): include a column with ranks. Defaults to True.
    incl_cumulative (bool, optional): include column with 
            cumulative value. Defaults to True.

Returns:
    pd.DataFrame: new dataframe with counting metrics
```
</details>


## References

Testing Dataset: Used Car Sales of Toyota Corollas from Kaggle ([source](https://www.kaggle.com/datasets/vishakhdapat/price-of-used-toyota-corolla-cars))