# CO3Dv2 statistics used for Stage 0 planning

These values were calculated on 2026-07-18 from the official CO3Dv2
`v2_231130` archive metadata and few-view train/dev/test set lists referenced by
the upstream `links.json`.

- 51 categories in the archive index.
- 36,507 sequence annotations.
- 36,458 sequences represented by the official usable sequence sets.
- Official split counts: 31,834 train, 2,586 validation, 2,038 test.
- 6,789,905 annotated frames across those usable sequences.
- Frames per sequence: mean 186.24, median 202, observed range 2-232.

The full configuration discovers categories from the local dataset root,
requires the exact official 51-name set, and records exact counts again. These
planning statistics are not substituted for the generated manifest summary.

| Category | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| apple | 816 | 50 | 50 | 916 |
| backpack | 1,215 | 106 | 50 | 1,371 |
| ball | 802 | 74 | 50 | 926 |
| banana | 572 | 50 | 50 | 672 |
| baseballbat | 109 | 14 | 14 | 137 |
| baseballglove | 111 | 15 | 15 | 141 |
| bench | 726 | 50 | 50 | 826 |
| bicycle | 499 | 50 | 50 | 599 |
| book | 2,160 | 83 | 50 | 2,293 |
| bottle | 747 | 50 | 50 | 847 |
| bowl | 600 | 82 | 20 | 702 |
| broccoli | 385 | 49 | 49 | 483 |
| cake | 383 | 48 | 48 | 479 |
| car | 476 | 41 | 50 | 567 |
| carrot | 869 | 95 | 50 | 1,014 |
| cellphone | 1,045 | 50 | 50 | 1,145 |
| chair | 1,331 | 88 | 50 | 1,469 |
| couch | 644 | 50 | 50 | 744 |
| cup | 500 | 50 | 50 | 600 |
| donut | 217 | 28 | 28 | 273 |
| frisbee | 115 | 34 | 20 | 169 |
| hairdryer | 777 | 50 | 50 | 877 |
| handbag | 549 | 95 | 20 | 664 |
| hotdog | 102 | 14 | 14 | 130 |
| hydrant | 625 | 50 | 50 | 725 |
| keyboard | 958 | 81 | 50 | 1,089 |
| kite | 215 | 27 | 27 | 269 |
| laptop | 862 | 70 | 50 | 982 |
| microwave | 118 | 26 | 19 | 163 |
| motorcycle | 556 | 50 | 50 | 656 |
| mouse | 1,135 | 50 | 50 | 1,235 |
| orange | 798 | 70 | 50 | 918 |
| parkingmeter | 46 | 6 | 6 | 58 |
| pizza | 159 | 21 | 21 | 201 |
| plant | 1,132 | 78 | 50 | 1,260 |
| remote | 1,349 | 50 | 50 | 1,449 |
| sandwich | 311 | 40 | 40 | 391 |
| skateboard | 141 | 18 | 18 | 177 |
| stopsign | 543 | 50 | 50 | 643 |
| suitcase | 641 | 50 | 50 | 741 |
| teddybear | 1,143 | 93 | 50 | 1,286 |
| toaster | 390 | 50 | 50 | 490 |
| toilet | 995 | 33 | 20 | 1,048 |
| toybus | 207 | 27 | 27 | 261 |
| toyplane | 316 | 40 | 40 | 396 |
| toytrain | 302 | 39 | 39 | 380 |
| toytruck | 623 | 50 | 50 | 723 |
| tv | 21 | 3 | 3 | 27 |
| umbrella | 849 | 50 | 50 | 949 |
| vase | 942 | 86 | 50 | 1,078 |
| wineglass | 707 | 62 | 50 | 819 |
