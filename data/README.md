# UCI Adult Census Income

Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5XW20

Source: https://archive.ics.uci.edu/dataset/2/adult
License: Creative Commons Attribution 4.0 International (CC BY 4.0).
https://creativecommons.org/licenses/by/4.0/

The unmodified official adult.data, adult.test and adult.names files are in raw/.
Source hashes and the derived fitting-data baseline hash are recorded in
src/models/model_summary.json each time training runs. The separate official
test file is not used for preprocessing fitting or deployment admission.
Schema and target normalization are in src/models/dataset.py.
