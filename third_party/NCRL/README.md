# NCRL integration

PPENet uses NCRL as an external, offline logical-rule miner. NCRL is not
proposed by the PPENet authors.

Official project: https://github.com/vivian1993/NCRL

The NCRL repository does not contain an explicit software license at the time
this release was prepared. For that reason, its source files are not copied
into the PPENet archive. Run the following command to fetch the official code:

```bash
python scripts/fetch_ncrl.py --target third_party/NCRL-src
```

The PPENet repository provides:

- `scripts/prepare_ncrl_dataset.py` for creating leakage-controlled NCRL input;
- `scripts/convert_ncrl_rules.py` for converting NCRL text rules to PPENet JSON;
- the exact rule-length and Top-500 command pattern in the root README.

Users remain responsible for reviewing and complying with the terms published
by the NCRL authors.
