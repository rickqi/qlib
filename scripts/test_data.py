"""Quick test: verify qlib can read tradingagents data."""
import qlib
from qlib.constant import REG_CN
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/tradingagents", region=REG_CN)

cal = D.calendar(start_time="2020-12-14", end_time="2020-12-20", freq="day")
print("Calendar first 3:", cal[:3])

instruments = D.instruments("all")
stocks = D.list_instruments(instruments=instruments, start_time="2026-05-08", end_time="2026-05-08", as_list=True)
print("Stocks on 2026-05-08:", len(stocks))

fields = ["$close", "$open", "$high", "$low", "$volume"]
df = D.features(["SH688041"], fields, start_time="2026-05-06", end_time="2026-05-08", freq="day")
print(df)
