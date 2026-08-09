# Data availability statement

## Public benchmarks

WN18RR and FB15k-237 are publicly available benchmarks. PPENet uses their
standard splits. This repository does not redistribute those benchmarks.

## Restricted EFKG-v1.0

The complete EFKG-v1.0 graph was constructed from operational fire-incident
records supplied by the Tianjin Fire Research Institute of the Ministry of
Emergency Management, China, for authorized research. The raw workbook and the
complete graph cannot be placed in an unrestricted public repository because
they retain combinations of regional, operational, and domain attributes that
may present residual linkage and re-identification risks even after direct
identifiers are removed.

## Public subset

`data/EFKG-Public-Subset` contains a deterministic 10% event sample with
re-encoded event identifiers, no geographic entities or relations, no raw
text, no source linkage, and only coarse categorical attributes. It supports
pipeline verification but is not a substitute for the complete graph and does
not reproduce the manuscript's EFKG metrics exactly.

## Controlled verification

Qualified academic requests for controlled verification of the complete EFKG
may be considered subject to data-provider approval, institutional review, and
a data-use agreement prohibiting redistribution and re-identification. The
authors cannot guarantee approval because the source data are owned and
governed by the provider.

Before uploading this package publicly, the authors must obtain the provider's
written approval for the included subset and replace its provisional data-use
notice with the approved dataset license.
