# Third-party notices

PPENet contains modifications and extensions of the open-source training,
prompt integration, and graph-adapter implementation released with DrKGC:

- Yongkang Xiao et al., "DrKGC: Dynamic Subgraph Retrieval-Augmented LLMs for
  Knowledge Graph Completion across General and Biomedical Domains," Findings
  of EMNLP 2025.
- Source: https://github.com/TheYKXiao/DrKGC
- License: Apache License 2.0. The repository-level `LICENSE` is retained.

The PPENet release renames its own modules to match the terminology used in the
revised manuscript and adds the graph-wide relational prior, disentangled
evidence components, relation-aware fusion, independence regularization, EFKG
data adapters, and controlled backbone support. The upstream attribution is
retained as required by the Apache License.

NCRL is a separate third-party project:

- Kewei Cheng, Nesreen K. Ahmed, and Yizhou Sun, "Neural Compositional Rule
  Learning for Knowledge Graph Reasoning," ICLR 2023.
- Paper: https://openreview.net/forum?id=F8VKQyDgRVj
- Official source: https://github.com/vivian1993/NCRL

The NCRL source repository does not currently include an explicit software
license file. It is therefore not redistributed in this archive. The included
fetch helper and conversion scripts obtain NCRL from its official repository
and keep it as an external component.
