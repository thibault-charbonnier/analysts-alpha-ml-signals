import polars as pl
from datetime import date, datetime


def _validate_date(date: date | datetime | pl.Date | str) -> date:
    """
    Validate and convert input date to a polars date object.

    Parameters
    ----------
    date: date, datetime, polars.Date or string
        Input date to validate

    Returns
    -------
    date
        Validated date object
    """
    if isinstance(date, pl.Date):
        return date
    elif isinstance(date, datetime):
        return pl.date(date.date())
    elif isinstance(date, date):
        return pl.date(date)
    elif isinstance(date, str):
        try:
            return pl.date(datetime.strptime(date, "%Y-%m-%d").date())
        except ValueError:
            raise ValueError(f"String date '{date}' is not in 'YYYY-MM-DD' format.")
    else:
        raise TypeError("Date must be of type date, datetime, polars.Date, or string.")
