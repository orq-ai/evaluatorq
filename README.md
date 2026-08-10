# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/orq-ai/evaluatorq/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                     |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/evaluatorq/\_\_init\_\_.py                                           |       19 |        0 |    100% |           |
| src/evaluatorq/\_\_main\_\_.py                                           |        3 |        3 |      0% |       1-4 |
| src/evaluatorq/cli.py                                                    |       50 |        2 |     96% |  117, 178 |
| src/evaluatorq/common/\_\_init\_\_.py                                    |        0 |        0 |    100% |           |
| src/evaluatorq/common/async\_utils.py                                    |       54 |        3 |     94% |60, 92, 97 |
| src/evaluatorq/common/cli\_epilog.py                                     |        6 |        0 |    100% |           |
| src/evaluatorq/common/cli\_errors.py                                     |       18 |        0 |    100% |           |
| src/evaluatorq/common/cli\_help.py                                       |        2 |        0 |    100% |           |
| src/evaluatorq/common/cli\_json.py                                       |        6 |        0 |    100% |           |
| src/evaluatorq/common/cli\_tty.py                                        |        4 |        0 |    100% |           |
| src/evaluatorq/common/cli\_width.py                                      |       10 |        2 |     80% |     19-20 |
| src/evaluatorq/common/content\_filter.py                                 |       29 |        1 |     97% |        81 |
| src/evaluatorq/common/fields.py                                          |        6 |        0 |    100% |           |
| src/evaluatorq/common/hook\_compose.py                                   |       16 |        0 |    100% |           |
| src/evaluatorq/common/judge.py                                           |      134 |        4 |     97% |110-111, 205, 219 |
| src/evaluatorq/common/jury.py                                            |      266 |        8 |     97% |84, 101, 114-115, 198, 315, 535, 584 |
| src/evaluatorq/common/llm\_call.py                                       |       95 |        2 |     98% |   217-218 |
| src/evaluatorq/common/llm\_client.py                                     |       47 |        0 |    100% |           |
| src/evaluatorq/common/messages.py                                        |       16 |        0 |    100% |           |
| src/evaluatorq/common/output\_adapters.py                                |       99 |        8 |     92% |31-33, 52, 108, 117-119 |
| src/evaluatorq/common/replay.py                                          |       85 |        5 |     94% |81-82, 101, 178, 182 |
| src/evaluatorq/common/reports/\_\_init\_\_.py                            |        9 |        0 |    100% |           |
| src/evaluatorq/common/reports/console.py                                 |       27 |        0 |    100% |           |
| src/evaluatorq/common/reports/executive\_summary.py                      |       34 |        2 |     94% |     91-92 |
| src/evaluatorq/common/reports/html\_helpers.py                           |      187 |       12 |     94% |57-59, 114-116, 200, 239, 265, 375, 402, 429 |
| src/evaluatorq/common/reports/md\_helpers.py                             |       44 |        1 |     98% |       106 |
| src/evaluatorq/common/reports/palette.py                                 |        9 |        0 |    100% |           |
| src/evaluatorq/common/reports/render.py                                  |       54 |        2 |     96% |     82-86 |
| src/evaluatorq/common/reports/rich\_styles.py                            |        8 |        0 |    100% |           |
| src/evaluatorq/common/reports/vega.py                                    |      114 |        5 |     96% |58-59, 92-94 |
| src/evaluatorq/common/retry.py                                           |       49 |        5 |     90% |54, 58-60, 79 |
| src/evaluatorq/common/run\_manifest.py                                   |      131 |       11 |     92% |69-72, 101, 112, 197-200, 234-238 |
| src/evaluatorq/common/run\_store\_dir.py                                 |        8 |        0 |    100% |           |
| src/evaluatorq/common/sanitize.py                                        |       12 |        0 |    100% |           |
| src/evaluatorq/common/target\_call.py                                    |       91 |        1 |     99% |        88 |
| src/evaluatorq/common/template\_engine.py                                |       60 |        2 |     97% |    68, 72 |
| src/evaluatorq/common/thread\_context.py                                 |       62 |        1 |     98% |        75 |
| src/evaluatorq/common/tracing.py                                         |      275 |       23 |     92% |155-166, 192, 204-206, 213, 397, 434-436, 496-497 |
| src/evaluatorq/common/ui/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| src/evaluatorq/common/ui/launch.py                                       |       20 |       12 |     40% |23, 35-39, 55-79 |
| src/evaluatorq/contracts.py                                              |      484 |       18 |     96% |57, 80, 114-116, 690, 753, 755, 757, 759, 809, 814, 818, 823, 825, 941, 1193, 1303, 1476 |
| src/evaluatorq/dashboard/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| src/evaluatorq/dashboard/\_compat.py                                     |       23 |       14 |     39% | 37-51, 69 |
| src/evaluatorq/dashboard/app.py                                          |      271 |       39 |     86% |87-88, 91-92, 257-258, 278-279, 316, 319, 322, 325-327, 343, 347-349, 376, 405, 409-411, 442, 445, 453-455, 481, 492, 500-502, 516, 547, 550, 558-560 |
| src/evaluatorq/dashboard/filter\_request.py                              |       13 |        1 |     92% |        42 |
| src/evaluatorq/dashboard/filters.py                                      |      165 |       12 |     93% |139-140, 154, 160, 178, 185, 238-239, 293, 299, 341, 408 |
| src/evaluatorq/dashboard/launch.py                                       |       48 |       10 |     79% |53-54, 81-89, 124 |
| src/evaluatorq/dashboard/library.py                                      |      153 |        8 |     95% |136-140, 159-160, 170, 206 |
| src/evaluatorq/dashboard/metrics.py                                      |      543 |       58 |     89% |88-89, 101-102, 120, 279-281, 298, 306-307, 310-311, 347, 350-351, 394-395, 398-399, 415-416, 419-420, 428-429, 432-433, 436-437, 580, 644, 648, 673, 678, 686-687, 763-764, 783-784, 787-788, 886, 900-907, 934-935, 946-947, 950-951 |
| src/evaluatorq/dashboard/orq\_links.py                                   |       40 |        0 |    100% |           |
| src/evaluatorq/dashboard/orq\_workspace.py                               |       15 |        0 |    100% |           |
| src/evaluatorq/dashboard/redteam\_charts.py                              |      189 |       20 |     89% |80-82, 94, 99-104, 114, 117, 176, 178, 207, 346, 359, 405, 419-420 |
| src/evaluatorq/dashboard/redteam\_transcripts.py                         |      111 |        4 |     96% |133, 135-136, 171 |
| src/evaluatorq/dashboard/redteam\_views.py                               |       90 |       11 |     88% |56-57, 60, 63-65, 126-127, 134, 147-148 |
| src/evaluatorq/dashboard/report\_kit.py                                  |      224 |       10 |     96% |109, 113, 233, 267, 305, 422-423, 483-484, 528 |
| src/evaluatorq/dashboard/report\_tabs.py                                 |      960 |       69 |     93% |61, 229, 255, 389, 432, 857-858, 862, 878, 1050-1069, 1097, 1131, 1150, 1153-1158, 1185, 1287, 1320, 1340-1370, 1375-1397, 1417, 1425, 1500, 1502, 1564, 1718, 1736, 2146, 2149, 2170, 2214 |
| src/evaluatorq/dashboard/shell.py                                        |       52 |        4 |     92% |60-61, 78-79 |
| src/evaluatorq/dashboard/sim\_compare.py                                 |      257 |       11 |     96% |156, 300, 308, 322, 335, 410, 518, 521, 587, 593, 600 |
| src/evaluatorq/dashboard/sim\_views.py                                   |      217 |       10 |     95% |67, 70-72, 96, 255, 257, 259, 572-573 |
| src/evaluatorq/dashboard/styles.py                                       |       12 |        0 |    100% |           |
| src/evaluatorq/dashboard/surfaces.py                                     |       66 |        0 |    100% |           |
| src/evaluatorq/dashboard/theme.py                                        |        2 |        0 |    100% |           |
| src/evaluatorq/dashboard/trace\_links.py                                 |       38 |        0 |    100% |           |
| src/evaluatorq/dashboard/view.py                                         |      413 |       32 |     92% |106-107, 280, 484, 491, 566, 622, 817, 858-859, 889-890, 1238-1278 |
| src/evaluatorq/deployment.py                                             |       73 |       46 |     37% |76-95, 139-166, 177-204, 236-244 |
| src/evaluatorq/evaluatorq.py                                             |      142 |       49 |     65% |51, 158, 223-321, 368, 393 |
| src/evaluatorq/evaluators.py                                             |       37 |       18 |     51% |52, 73, 109-144 |
| src/evaluatorq/fetch\_data.py                                            |      145 |       24 |     83% |44-55, 89-91, 129, 190-191, 195-198, 258, 266, 300, 321, 330, 333-334 |
| src/evaluatorq/integrations/\_\_init\_\_.py                              |       22 |       19 |     14% |     31-55 |
| src/evaluatorq/integrations/callable\_integration/\_\_init\_\_.py        |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/callable\_integration/target.py              |       55 |        0 |    100% |           |
| src/evaluatorq/integrations/crewai\_integration/\_\_init\_\_.py          |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/crewai\_integration/target.py                |       65 |        8 |     88% |   131-142 |
| src/evaluatorq/integrations/langchain\_integration/\_\_init\_\_.py       |        4 |        0 |    100% |           |
| src/evaluatorq/integrations/langchain\_integration/convert.py            |      150 |       99 |     34% |44-46, 99-200, 218, 220-221, 226, 240, 242, 285-294, 300-313, 318-332, 337-354, 359, 364, 369, 374, 379-385 |
| src/evaluatorq/integrations/langchain\_integration/types.py              |        4 |        0 |    100% |           |
| src/evaluatorq/integrations/langchain\_integration/wrap\_agent.py        |       72 |       21 |     71% |29-41, 167, 170-188 |
| src/evaluatorq/integrations/langgraph\_integration/\_\_init\_\_.py       |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/langgraph\_integration/target.py             |      172 |       16 |     91% |53, 57, 83, 103-104, 112-116, 203, 222-229, 278 |
| src/evaluatorq/integrations/openai\_agents\_integration/\_\_init\_\_.py  |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/openai\_agents\_integration/target.py        |      155 |       41 |     74% |155-176, 211, 219, 258-272, 284, 286, 303, 308-309 |
| src/evaluatorq/integrations/pydantic\_ai\_integration/\_\_init\_\_.py    |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/pydantic\_ai\_integration/target.py          |       95 |        9 |     91% |120-127, 170 |
| src/evaluatorq/integrations/vercel\_ai\_sdk\_integration/\_\_init\_\_.py |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/vercel\_ai\_sdk\_integration/target.py       |      121 |        8 |     93% |229, 258, 261, 266-267, 271-272, 330 |
| src/evaluatorq/job\_helper.py                                            |       25 |        2 |     92% |    81, 98 |
| src/evaluatorq/llm\_jury.py                                              |      185 |        4 |     98% |141, 253, 615, 685 |
| src/evaluatorq/openresponses/\_\_init\_\_.py                             |        4 |        0 |    100% |           |
| src/evaluatorq/openresponses/client.py                                   |        7 |        0 |    100% |           |
| src/evaluatorq/openresponses/convert\_models.py                          |      121 |        3 |     98% |   168-175 |
| src/evaluatorq/openresponses/dataset.py                                  |      159 |       15 |     91% |61, 67, 71, 78, 98, 125, 153, 161, 167, 176-178, 213, 223, 228 |
| src/evaluatorq/openresponses/input\_items.py                             |       34 |        1 |     97% |        34 |
| src/evaluatorq/openresponses/target.py                                   |      118 |        3 |     97% |134, 231, 254 |
| src/evaluatorq/openresponses/tracing.py                                  |       74 |       18 |     76% |40, 67-73, 85, 87, 99-103, 117, 138-145 |
| src/evaluatorq/openresponses/types.py                                    |       91 |        0 |    100% |           |
| src/evaluatorq/pairwise.py                                               |      213 |        7 |     97% |304, 307-308, 428-430, 565 |
| src/evaluatorq/pairwise\_reports/\_\_init\_\_.py                         |        3 |        0 |    100% |           |
| src/evaluatorq/pairwise\_reports/export\_html.py                         |      114 |        6 |     95% |93-98, 135, 146 |
| src/evaluatorq/pairwise\_reports/sections.py                             |       56 |        0 |    100% |           |
| src/evaluatorq/pairwise\_run.py                                          |       85 |        4 |     95% |145-149, 152 |
| src/evaluatorq/processings.py                                            |       64 |        5 |     92% |51, 87-89, 270-272 |
| src/evaluatorq/progress.py                                               |      101 |       53 |     48% |55-95, 99-104, 137, 141-148, 152-157, 166-180, 213-231 |
| src/evaluatorq/ranking.py                                                |      195 |        0 |    100% |           |
| src/evaluatorq/redteam/\_\_init\_\_.py                                   |       33 |        5 |     85% |   249-258 |
| src/evaluatorq/redteam/adaptive/\_\_init\_\_.py                          |        7 |        0 |    100% |           |
| src/evaluatorq/redteam/adaptive/agent\_context.py                        |       12 |       12 |      0% |      3-42 |
| src/evaluatorq/redteam/adaptive/attack\_generator.py                     |       54 |        2 |     96% |  214, 229 |
| src/evaluatorq/redteam/adaptive/blackbox\_classifier.py                  |      124 |        7 |     94% |299-304, 360 |
| src/evaluatorq/redteam/adaptive/capability\_classifier.py                |      108 |        6 |     94% |165, 184, 255, 258, 323, 335 |
| src/evaluatorq/redteam/adaptive/evaluator.py                             |       87 |        1 |     99% |       115 |
| src/evaluatorq/redteam/adaptive/objective\_generator.py                  |      139 |       33 |     76% |54, 142, 166, 284-307, 419-433, 476-480, 495, 545-546, 548-549, 652-686 |
| src/evaluatorq/redteam/adaptive/orchestrator.py                          |      386 |       70 |     82% |98-125, 131-138, 147-160, 166-167, 173-187, 192-196, 201, 392-393, 459-465, 614-615, 890, 986, 1006, 1044-1054 |
| src/evaluatorq/redteam/adaptive/pipeline.py                              |      231 |       31 |     87% |86, 88, 237-241, 270, 274-313, 391, 400, 658-659, 703-704, 792, 800-804 |
| src/evaluatorq/redteam/adaptive/strategy\_planner.py                     |       99 |        6 |     94% |71-85, 173-177 |
| src/evaluatorq/redteam/adaptive/strategy\_registry.py                    |      101 |        2 |     98% |  259, 263 |
| src/evaluatorq/redteam/adaptive/tool\_chaining.py                        |       78 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/\_errors.py                              |       40 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/base.py                                  |       78 |        5 |     94% |37-40, 140 |
| src/evaluatorq/redteam/backends/openai.py                                |       96 |        5 |     95% |123, 142, 198, 223, 232 |
| src/evaluatorq/redteam/backends/openresponses.py                         |       43 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/orq.py                                   |      286 |       42 |     85% |27-28, 68-83, 237-238, 397, 402, 477-487, 491-503, 533-534, 544, 548, 567, 585, 597-606 |
| src/evaluatorq/redteam/backends/registry.py                              |       53 |        6 |     89% |43, 53-57, 81, 110-111 |
| src/evaluatorq/redteam/cli.py                                            |      335 |      132 |     61% |66, 72, 86-87, 153-155, 167, 181-183, 195, 394, 417, 500-501, 503-504, 511-515, 518-520, 523-525, 528-530, 558-569, 574-579, 602-701, 724-728, 742-743, 768, 771-772, 842-861 |
| src/evaluatorq/redteam/contracts.py                                      |      748 |       46 |     94% |92, 132-134, 266-267, 452, 465-466, 479, 505, 605, 613, 617, 637, 1053, 1095-1115, 1121, 1215-1217, 1226, 1312-1318, 1383-1385, 1611, 1810-1819 |
| src/evaluatorq/redteam/delivery\_method\_registry.py                     |       59 |        1 |     98% |       110 |
| src/evaluatorq/redteam/exceptions.py                                     |        5 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/\_\_init\_\_.py                        |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/owasp/\_\_init\_\_.py                  |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/owasp/agent\_evaluators.py             |       44 |        1 |     98% |       961 |
| src/evaluatorq/redteam/frameworks/owasp/evaluatorq\_bridge.py            |      224 |       66 |     71% |78, 107, 123-124, 133-143, 148, 159-167, 208, 213-215, 239-240, 293, 379-380, 400-404, 409-413, 418-463 |
| src/evaluatorq/redteam/frameworks/owasp/evaluators.py                    |       67 |       25 |     63% |138-151, 169, 188, 198-211, 220 |
| src/evaluatorq/redteam/frameworks/owasp/llm\_evaluators.py               |       36 |       19 |     47% |178-374, 548-655, 677-796, 818-936, 957-1075, 1096-1213, 1233 |
| src/evaluatorq/redteam/frameworks/owasp/models.py                        |       41 |        3 |     93% | 11-13, 73 |
| src/evaluatorq/redteam/frameworks/owasp\_asi.py                          |        8 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/owasp\_llm.py                          |        8 |        0 |    100% |           |
| src/evaluatorq/redteam/hooks.py                                          |      375 |       95 |     75% |305, 346, 416-420, 424-436, 471, 480-481, 493, 509, 573-574, 646, 716, 734-817, 821-823, 836-837, 845-848, 850-853, 858-859 |
| src/evaluatorq/redteam/judge.py                                          |        3 |        0 |    100% |           |
| src/evaluatorq/redteam/replay.py                                         |       59 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/\_\_init\_\_.py                           |        4 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/\_utils.py                                |       15 |        2 |     87% |     27-28 |
| src/evaluatorq/redteam/reports/converters.py                             |      522 |       34 |     93% |110, 153-159, 194-197, 235, 237-240, 243, 250, 256-257, 265-271, 433, 562-566, 854-856 |
| src/evaluatorq/redteam/reports/display.py                                |      125 |       40 |     68% |21-22, 27, 35, 81-82, 84, 89-96, 101-109, 203-232 |
| src/evaluatorq/redteam/reports/executive\_summary.py                     |       35 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/export\_html.py                           |      704 |      109 |     85% |138, 164, 189, 286-289, 326, 328-329, 368, 374-385, 464, 485, 506-526, 531-583, 594, 605, 613-616, 634, 670, 706, 729, 747-751, 753, 755, 760-766, 768-774, 776-782, 784-790, 810, 822, 849, 857, 877-886, 894, 919, 938, 944, 964, 998, 1068, 1097, 1125, 1162, 1201-1203, 1303, 1333, 1364, 1367, 1370, 1384, 1387, 1389-1390, 1405-1406, 1468, 1470, 1515, 1520 |
| src/evaluatorq/redteam/reports/export\_md.py                             |      421 |      138 |     67% |41, 44, 86, 123, 127-130, 133-134, 154, 169, 171, 173, 175, 177, 179, 181, 215, 219-227, 234-273, 278-321, 378, 393, 432, 457, 479, 488, 509-557, 562-580, 585-592, 602, 615, 666, 687, 689, 694, 716, 718, 720-721, 733-735, 823 |
| src/evaluatorq/redteam/reports/guidance.py                               |        2 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/recommendations.py                        |       75 |       22 |     71% |36, 102-131, 164, 171, 221 |
| src/evaluatorq/redteam/reports/sections.py                               |      332 |       25 |     92% |71, 86, 100-101, 103-104, 317-345, 435-440, 543, 783, 952 |
| src/evaluatorq/redteam/runner.py                                         |     1091 |      131 |     88% |177-179, 216, 304, 338, 345, 586-587, 596-601, 603-604, 654-655, 685, 798-799, 878-881, 962-963, 990, 1002-1006, 1038-1042, 1073, 1113-1115, 1171, 1345, 1506, 1514-1516, 1625-1630, 1846-1847, 1852-1859, 1867-1868, 1888, 1890-1894, 1919-1929, 2027, 2066, 2072, 2185-2186, 2188-2190, 2251, 2289, 2333-2334, 2365-2370, 2402-2412, 2446-2447, 2513-2516, 2547-2573, 2601-2603, 2611, 2658, 2684, 2707, 2723, 2728, 2730-2736, 2807, 2915, 3085-3086, 3095, 3098-3099 |
| src/evaluatorq/redteam/runtime/\_\_init\_\_.py                           |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/runtime/jobs.py                                   |      126 |       12 |     90% |78-83, 87, 135, 156, 199-201, 208, 251-252 |
| src/evaluatorq/redteam/tracing.py                                        |       51 |        6 |     88% |103-105, 142-144 |
| src/evaluatorq/redteam/ui/\_\_init\_\_.py                                |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/ui/colors.py                                      |        2 |        2 |      0% |       3-5 |
| src/evaluatorq/redteam/ui/dashboard.py                                   |     1322 |     1322 |      0% |    8-2623 |
| src/evaluatorq/redteam/utils.py                                          |        9 |        1 |     89% |        27 |
| src/evaluatorq/redteam/vulnerability\_registry.py                        |       73 |        4 |     95% |186, 300, 313, 327 |
| src/evaluatorq/send\_results.py                                          |       59 |        0 |    100% |           |
| src/evaluatorq/simulation/\_\_init\_\_.py                                |       23 |        1 |     96% |       300 |
| src/evaluatorq/simulation/\_config.py                                    |       41 |        0 |    100% |           |
| src/evaluatorq/simulation/\_datapoint\_io.py                             |       51 |        2 |     96% |    68, 92 |
| src/evaluatorq/simulation/adapters.py                                    |       24 |        7 |     71% | 25, 57-63 |
| src/evaluatorq/simulation/agents/\_\_init\_\_.py                         |        4 |        0 |    100% |           |
| src/evaluatorq/simulation/agents/base.py                                 |      173 |       15 |     91% |174-183, 191, 195-196, 304, 319-325, 382, 389, 433, 479 |
| src/evaluatorq/simulation/agents/judge.py                                |      116 |       14 |     88% |165-169, 215-217, 244, 274, 323-324, 341-342 |
| src/evaluatorq/simulation/agents/user\_simulator.py                      |       35 |       13 |     63% |64, 68-70, 74-81, 89-95 |
| src/evaluatorq/simulation/api.py                                         |      561 |       53 |     91% |408, 580, 828, 882, 946, 1007, 1058, 1075, 1177-1179, 1278, 1303-1304, 1310, 1433, 1459, 1511-1512, 1522, 1526, 1531, 1595, 1602-1619, 1686, 1756-1762, 1765-1768, 1808-1809, 1811, 1913-1916, 1923-1926 |
| src/evaluatorq/simulation/cli.py                                         |      667 |      128 |     81% |92-99, 108-109, 138, 142, 144, 153, 157, 162, 165-167, 171, 181, 208, 218, 234-235, 246-264, 269, 275-278, 708, 745, 747, 775-776, 778-779, 800, 837, 840, 1046, 1093-1094, 1096-1097, 1109, 1324-1325, 1327-1328, 1330, 1441-1501, 1604-1605, 1621, 1652-1653, 1697, 1745, 1756-1757, 1759, 1812-1813, 1827-1831, 1855, 1944-1949, 2040-2041, 2119 |
| src/evaluatorq/simulation/convert.py                                     |       44 |        0 |    100% |           |
| src/evaluatorq/simulation/evaluators/\_\_init\_\_.py                     |        3 |        0 |    100% |           |
| src/evaluatorq/simulation/evaluators/scorers.py                          |       41 |        0 |    100% |           |
| src/evaluatorq/simulation/exceptions.py                                  |        8 |        0 |    100% |           |
| src/evaluatorq/simulation/experiments.py                                 |       44 |        0 |    100% |           |
| src/evaluatorq/simulation/generators/\_\_init\_\_.py                     |        5 |        0 |    100% |           |
| src/evaluatorq/simulation/generators/datapoint\_generator.py             |       84 |       55 |     35% |60-61, 84-152, 160-179, 187-207 |
| src/evaluatorq/simulation/generators/first\_message\_generator.py        |       53 |        2 |     96% |     93-94 |
| src/evaluatorq/simulation/generators/persona\_generator.py               |      118 |       79 |     33% |98, 102-126, 165, 197, 213-334, 338-350, 354-387 |
| src/evaluatorq/simulation/generators/scenario\_generator.py              |      177 |      126 |     29% |159-167, 171-177, 198, 272-273, 283-284, 295-368, 378-438, 447-504, 515-583, 591-603, 607-620 |
| src/evaluatorq/simulation/hooks.py                                       |      213 |        8 |     96% |243-245, 357, 365, 395, 399, 432, 538 |
| src/evaluatorq/simulation/metrics.py                                     |        5 |        0 |    100% |           |
| src/evaluatorq/simulation/quality/\_\_init\_\_.py                        |        2 |        0 |    100% |           |
| src/evaluatorq/simulation/quality/message\_perturbation.py               |       66 |       42 |     36% |78-83, 87-94, 98-104, 108-111, 115-121, 140-143, 152-153, 165-172 |
| src/evaluatorq/simulation/replay.py                                      |       27 |        2 |     93% |     70-71 |
| src/evaluatorq/simulation/reports/\_\_init\_\_.py                        |        5 |        0 |    100% |           |
| src/evaluatorq/simulation/reports/display.py                             |       79 |        3 |     96% |142-143, 154 |
| src/evaluatorq/simulation/reports/executive\_summary.py                  |       35 |        0 |    100% |           |
| src/evaluatorq/simulation/reports/export\_html.py                        |      302 |        5 |     98% |137, 303, 358, 498, 607 |
| src/evaluatorq/simulation/reports/export\_md.py                          |      251 |       13 |     95% |103, 207, 232-239, 315, 370, 434 |
| src/evaluatorq/simulation/reports/recommendations.py                     |       76 |        3 |     96% |48, 157, 183 |
| src/evaluatorq/simulation/reports/sections.py                            |      258 |        1 |     99% |        94 |
| src/evaluatorq/simulation/reports/token\_usage.py                        |       26 |        0 |    100% |           |
| src/evaluatorq/simulation/runner/\_\_init\_\_.py                         |        3 |        0 |    100% |           |
| src/evaluatorq/simulation/runner/simulation.py                           |      317 |       26 |     92% |152-153, 261, 272, 418-424, 447-453, 525, 626, 756-764, 782-783, 785-786, 855, 877 |
| src/evaluatorq/simulation/token\_usage.py                                |        7 |        0 |    100% |           |
| src/evaluatorq/simulation/traces.py                                      |      220 |       13 |     94% |105, 110, 119, 185, 188, 294, 369, 401-405, 415, 461, 488, 492 |
| src/evaluatorq/simulation/tracing.py                                     |       69 |       12 |     83% |67-73, 117-123 |
| src/evaluatorq/simulation/types.py                                       |      209 |        2 |     99% |   362-363 |
| src/evaluatorq/simulation/ui/\_\_init\_\_.py                             |        0 |        0 |    100% |           |
| src/evaluatorq/simulation/ui/colors.py                                   |        2 |        2 |      0% |       7-9 |
| src/evaluatorq/simulation/ui/dashboard.py                                |      300 |      300 |      0% |    13-590 |
| src/evaluatorq/simulation/ui/token\_display.py                           |       28 |        0 |    100% |           |
| src/evaluatorq/simulation/utils/\_\_init\_\_.py                          |        5 |        0 |    100% |           |
| src/evaluatorq/simulation/utils/dataset\_export.py                       |       75 |       28 |     63% |54-56, 61-64, 88-90, 109-115, 130-136, 179-181, 189-214 |
| src/evaluatorq/simulation/utils/extract\_json.py                         |       61 |        6 |     90% |75, 94-95, 103-106 |
| src/evaluatorq/simulation/utils/prompt\_builders.py                      |       65 |        6 |     91% |45, 52, 59, 64, 80, 124 |
| src/evaluatorq/simulation/utils/run\_store.py                            |      112 |        6 |     95% |72-74, 80-82 |
| src/evaluatorq/simulation/utils/structured\_output.py                    |       47 |       16 |     66% |80, 84, 86-102, 115-128 |
| src/evaluatorq/simulation/wrap\_agent.py                                 |       29 |        3 |     90% |75, 77, 98 |
| src/evaluatorq/table\_display.py                                         |      147 |       89 |     39% |31, 36-76, 100, 106, 145, 156, 173-207, 212-230, 240-278 |
| src/evaluatorq/tracing/\_\_init\_\_.py                                   |        4 |        0 |    100% |           |
| src/evaluatorq/tracing/context.py                                        |       34 |        2 |     94% |     50-51 |
| src/evaluatorq/tracing/setup.py                                          |      136 |       39 |     71% |103-113, 134-137, 142, 171, 177-183, 191-201, 235-243, 292-293, 314 |
| src/evaluatorq/tracing/spans.py                                          |      101 |       24 |     76% |76-107, 154-161, 272 |
| src/evaluatorq/types.py                                                  |       83 |        2 |     98% |   32, 219 |
| **TOTAL**                                                                | **24224** | **4542** | **81%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/orq-ai/evaluatorq/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/orq-ai/evaluatorq/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/orq-ai/evaluatorq/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/orq-ai/evaluatorq/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Forq-ai%2Fevaluatorq%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/orq-ai/evaluatorq/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.