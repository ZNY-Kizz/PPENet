# EFKG-Public-Subset-v1.0

This directory contains a privacy-filtered subset derived from the restricted
EFKG-v1.0 graph used in the PPENet study. It is released to verify file formats,
preprocessing, prior training, NCRL integration, evidence retrieval, and the
end-to-end PPENet execution path.

It is **not** structurally identical to the restricted EFKG-v1.0 graph and must
not be used to claim exact reproduction of the EFKG metrics reported in the
paper.

## Released graph

| Item | Value |
|---|---:|
| Sampled event entities | 1,500 |
| Total entities | 1,608 |
| Relations | 8 |
| Training triples | 9,603 |
| Validation triples | 1,197 |
| Test triples | 1,200 |
| Total triples | 12,000 |

The files use contiguous integer identifiers:

```text
head_entity_id<TAB>relation_id<TAB>tail_entity_id
```

## Privacy boundary

The release removes source rows, source linkage, detailed addresses, roads,
organization and administrative codes, free-text incident descriptions,
province/city nodes, geographic relations, continuous measurements,
timestamps, and fine-grained topic relations. Event identifiers are newly
generated non-semantic labels.

The retained coarse categorical values may still encode domain information.
The subset is therefore described as privacy-filtered and de-identified, not
as irreversibly anonymous. Re-identification, linkage attacks, and attempts to
associate released nodes with real incidents are prohibited.

## Validation

`validation_report.json` records privacy checks, duplicate checks, split
overlap checks, and cold-start checks. `checksums.sha256` provides integrity
hashes for every released data file.

## Full EFKG access

The original incident workbook and the complete experimental EFKG graph are
not publicly downloadable. Requests for controlled academic verification may
be considered only after approval by the data provider and execution of an
appropriate data-use agreement.
