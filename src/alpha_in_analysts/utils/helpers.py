import polars as pl
from datetime import date, datetime, timedelta
import pandas_market_calendars as mcal

def _validate_date(dt: date | datetime | str) -> date:
    """
    Validate and convert input date to a valid date object.

    Parameters
    ----------
    date: date, datetime, or string
        Input date to validate

    Returns
    -------
    date
        Validated date object
    """
    if isinstance(dt, datetime):
        return dt.date()
    if isinstance(dt, date):
        return dt
    if isinstance(dt, str):
        try:
            return datetime.strptime(dt, "%Y-%m-%d").date()
        except ValueError as e:
            raise ValueError(f"String date '{dt}' must be in 'YYYY-MM-DD' format.") from e

    raise TypeError("Date must be of type date, datetime, or string.")

def _last_trading_day_of_month(y: int, m: int) -> date:
    start = date(y, m, 1)

    # first day of next month
    y2 = y + (m // 12)
    m2 = (m % 12) + 1
    end = date(y2, m2, 1) - timedelta(days=1)

    sched = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    if sched.empty:
        raise ValueError(f"No trading days found for {y}-{m:02d} on calendar NYSE")

    return sched.index.max().date()

def _parse_date(ym: str) -> date:
    y, m = ym.split("-")
    return _last_trading_day_of_month(int(y), int(m))

def _add_months(d: date, n: int) -> date:
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return _last_trading_day_of_month(y, m)

def _month_range(start: date, end: date) -> list[date]:
    out = []
    d = start
    while d <= end:
        out.append(d)
        d = _add_months(d, 1)
    return out