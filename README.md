# Changelog
## 2020-02-10
Changed join_cust_info_to_preds() function in predict.py to now be able to determine which additional columns we can join in

## 2020-02-19
Added additional metrics to model diagnostics output: calculation date, training time range start and end date (this is the time window between which we use training samples), training actual start and end date (these are the start and end dates in which first actual customers started their trip)
