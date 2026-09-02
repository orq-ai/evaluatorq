# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/orq-ai/evaluatorq/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                     |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/evaluatorq/\_\_init\_\_.py                                           |       24 |        2 |     92% |     30-31 |
| src/evaluatorq/\_\_main\_\_.py                                           |        3 |        3 |      0% |       1-4 |
| src/evaluatorq/cli.py                                                    |       50 |        2 |     96% |  120, 181 |
| src/evaluatorq/common/\_\_init\_\_.py                                    |        0 |        0 |    100% |           |
| src/evaluatorq/common/apply.py                                           |      137 |        6 |     96% |208, 221-222, 228, 331-332 |
| src/evaluatorq/common/async\_utils.py                                    |       54 |        3 |     94% |60, 92, 97 |
| src/evaluatorq/common/cli\_epilog.py                                     |        6 |        0 |    100% |           |
| src/evaluatorq/common/cli\_errors.py                                     |       18 |        0 |    100% |           |
| src/evaluatorq/common/cli\_help.py                                       |        5 |        0 |    100% |           |
| src/evaluatorq/common/cli\_json.py                                       |        6 |        0 |    100% |           |
| src/evaluatorq/common/cli\_tty.py                                        |        4 |        0 |    100% |           |
| src/evaluatorq/common/cli\_width.py                                      |       10 |        2 |     80% |     19-20 |
| src/evaluatorq/common/content\_filter.py                                 |       29 |        1 |     97% |        81 |
| src/evaluatorq/common/extract\_json.py                                   |       79 |       10 |     87% |32, 40, 67-68, 121, 140-141, 149-152 |
| src/evaluatorq/common/fields.py                                          |        6 |        0 |    100% |           |
| src/evaluatorq/common/hook\_compose.py                                   |       16 |        0 |    100% |           |
| src/evaluatorq/common/judge.py                                           |      216 |        5 |     98% |230-231, 325, 339, 701 |
| src/evaluatorq/common/jury.py                                            |      299 |        8 |     97% |115, 132, 145-146, 229, 442, 777, 826 |
| src/evaluatorq/common/llm\_call.py                                       |      168 |        2 |     99% |  381, 393 |
| src/evaluatorq/common/llm\_client.py                                     |       48 |        0 |    100% |           |
| src/evaluatorq/common/llm\_limit.py                                      |       31 |        0 |    100% |           |
| src/evaluatorq/common/messages.py                                        |       16 |        0 |    100% |           |
| src/evaluatorq/common/model\_catalogue.py                                |      150 |        5 |     97% |223, 226, 282, 289, 320 |
| src/evaluatorq/common/orq\_client.py                                     |       14 |        1 |     93% |        45 |
| src/evaluatorq/common/output\_adapters.py                                |       99 |        7 |     93% |37-39, 114, 123-125 |
| src/evaluatorq/common/parallelism.py                                     |       10 |        0 |    100% |           |
| src/evaluatorq/common/prompt\_cache.py                                   |       73 |        0 |    100% |           |
| src/evaluatorq/common/recommendations.py                                 |       16 |        0 |    100% |           |
| src/evaluatorq/common/replay.py                                          |       85 |        5 |     94% |81-82, 101, 178, 182 |
| src/evaluatorq/common/reports/\_\_init\_\_.py                            |        9 |        0 |    100% |           |
| src/evaluatorq/common/reports/console.py                                 |       27 |        0 |    100% |           |
| src/evaluatorq/common/reports/executive\_summary.py                      |       46 |        2 |     96% |   111-112 |
| src/evaluatorq/common/reports/html\_helpers.py                           |      189 |       12 |     94% |57-59, 116-118, 202, 241, 267, 377, 404, 431 |
| src/evaluatorq/common/reports/md\_helpers.py                             |       52 |        1 |     98% |       134 |
| src/evaluatorq/common/reports/palette.py                                 |       11 |        0 |    100% |           |
| src/evaluatorq/common/reports/render.py                                  |       54 |        2 |     96% |     82-86 |
| src/evaluatorq/common/reports/rich\_styles.py                            |        8 |        0 |    100% |           |
| src/evaluatorq/common/reports/vega.py                                    |      121 |        7 |     94% |58-59, 93-102 |
| src/evaluatorq/common/responses.py                                       |       37 |        8 |     78% |22-24, 72-76 |
| src/evaluatorq/common/retry.py                                           |       57 |        4 |     93% |102, 134, 136, 138 |
| src/evaluatorq/common/run\_manifest.py                                   |      142 |       11 |     92% |77-80, 109, 120, 208-211, 285-289 |
| src/evaluatorq/common/run\_store\_dir.py                                 |        8 |        0 |    100% |           |
| src/evaluatorq/common/sanitize.py                                        |       12 |        0 |    100% |           |
| src/evaluatorq/common/structured\_output.py                              |      290 |        9 |     97% |164, 262, 319, 341, 526-527, 641, 704, 774 |
| src/evaluatorq/common/target\_call.py                                    |      117 |        0 |    100% |           |
| src/evaluatorq/common/template\_engine.py                                |       60 |        2 |     97% |    68, 72 |
| src/evaluatorq/common/thread\_context.py                                 |       62 |        1 |     98% |        75 |
| src/evaluatorq/common/tracing.py                                         |      322 |       27 |     92% |168-179, 217-219, 226, 422, 488-490, 569-571, 608-609, 641-642 |
| src/evaluatorq/common/ui/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| src/evaluatorq/common/ui/launch.py                                       |       20 |       12 |     40% |23, 35-39, 55-79 |
| src/evaluatorq/contracts.py                                              |      549 |       17 |     97% |58, 81, 134-136, 332, 904, 980, 984, 986, 1036, 1041, 1045, 1050, 1052, 1171, 1427, 1749 |
| src/evaluatorq/dashboard/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| src/evaluatorq/dashboard/\_compat.py                                     |       23 |       14 |     39% | 39-53, 71 |
| src/evaluatorq/dashboard/app.py                                          |      277 |       39 |     86% |84-85, 88-89, 259-260, 280-281, 318, 321, 324, 327-329, 345, 349-351, 378, 407, 411-413, 444, 447, 455-457, 483, 494, 502-504, 518, 549, 552, 560-562 |
| src/evaluatorq/dashboard/apply\_ui.py                                    |      293 |       29 |     90% |78, 154-159, 474-484, 554-555, 561-563, 595-600, 604, 643-644, 652, 675, 682, 706, 711, 765, 808 |
| src/evaluatorq/dashboard/filter\_request.py                              |       13 |        1 |     92% |        42 |
| src/evaluatorq/dashboard/filters.py                                      |      168 |       13 |     92% |132, 142-143, 157, 163, 181, 188, 241-242, 296, 302, 344, 411 |
| src/evaluatorq/dashboard/launch.py                                       |       48 |       10 |     79% |53-54, 81-89, 124 |
| src/evaluatorq/dashboard/library.py                                      |      184 |       13 |     93% |138-142, 162-163, 173, 209, 259-260, 296-297, 299 |
| src/evaluatorq/dashboard/metrics.py                                      |      682 |       59 |     91% |100-101, 107-108, 139, 339-341, 358, 366-367, 370-371, 410, 413-414, 459-460, 463-464, 484-485, 488-489, 502-503, 506-507, 510-511, 670, 734, 738, 763, 768, 776-777, 872-873, 890-891, 894-895, 962, 1092-1099, 1132-1133, 1198, 1204-1205, 1207-1208 |
| src/evaluatorq/dashboard/orq\_links.py                                   |       40 |        0 |    100% |           |
| src/evaluatorq/dashboard/orq\_workspace.py                               |       15 |        0 |    100% |           |
| src/evaluatorq/dashboard/redteam\_charts.py                              |      189 |       17 |     91% |79-81, 93, 98-101, 111, 114, 173, 175, 363, 376, 422, 436-437 |
| src/evaluatorq/dashboard/redteam\_transcripts.py                         |      116 |        6 |     95% |74, 143, 145-146, 181, 279 |
| src/evaluatorq/dashboard/redteam\_views.py                               |       88 |        8 |     91% |56-57, 60, 63-65, 151-152 |
| src/evaluatorq/dashboard/report\_kit.py                                  |      230 |       10 |     96% |121, 125, 245, 279, 317, 434-435, 495-496, 540 |
| src/evaluatorq/dashboard/report\_tabs.py                                 |     1062 |       79 |     93% |62, 299, 325, 459, 502, 927-928, 932, 948, 1120-1139, 1167, 1201, 1220, 1223-1228, 1255, 1357, 1390, 1410-1440, 1445-1467, 1487, 1495, 1577, 1579, 1845, 2115, 2117, 2310, 2316, 2337, 2375-2391, 2449 |
| src/evaluatorq/dashboard/shell.py                                        |       52 |        4 |     92% |60-61, 78-79 |
| src/evaluatorq/dashboard/sim\_compare.py                                 |      270 |        8 |     97% |156, 308, 342, 551, 554, 620, 626, 633 |
| src/evaluatorq/dashboard/sim\_views.py                                   |      226 |       10 |     96% |67, 70-72, 96, 255, 257, 259, 610-611 |
| src/evaluatorq/dashboard/styles.py                                       |       12 |        0 |    100% |           |
| src/evaluatorq/dashboard/surfaces.py                                     |       66 |        0 |    100% |           |
| src/evaluatorq/dashboard/theme.py                                        |        2 |        0 |    100% |           |
| src/evaluatorq/dashboard/trace\_links.py                                 |       38 |        0 |    100% |           |
| src/evaluatorq/dashboard/view.py                                         |      439 |       33 |     92% |104, 148-149, 331, 542, 549, 624, 681, 878, 919-920, 950-951, 1299-1339 |
| src/evaluatorq/deployment.py                                             |       72 |        9 |     88% |79-81, 134, 170, 188, 194, 252-260 |
| src/evaluatorq/evaluatorq.py                                             |      182 |        6 |     97% |66, 76, 217, 316, 322, 506 |
| src/evaluatorq/evaluators.py                                             |       37 |       18 |     51% |54, 75, 113-148 |
| src/evaluatorq/fetch\_data.py                                            |      139 |       17 |     88% |48, 82-84, 122, 183-184, 188-191, 251, 259, 293, 314, 323, 326-327 |
| src/evaluatorq/integrations/\_\_init\_\_.py                              |        6 |        3 |     50% |     39-41 |
| src/evaluatorq/integrations/callable\_integration/\_\_init\_\_.py        |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/callable\_integration/target.py              |       55 |        0 |    100% |           |
| src/evaluatorq/integrations/crewai\_integration/\_\_init\_\_.py          |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/crewai\_integration/target.py                |       66 |        8 |     88% |   135-146 |
| src/evaluatorq/integrations/langchain\_integration/\_\_init\_\_.py       |        4 |        0 |    100% |           |
| src/evaluatorq/integrations/langchain\_integration/convert.py            |      159 |       47 |     70% |120-126, 138, 175-202, 220, 222-223, 228, 242, 244, 291-293, 307-311, 321, 339-349, 365, 376, 397, 400-402 |
| src/evaluatorq/integrations/langchain\_integration/types.py              |        4 |        0 |    100% |           |
| src/evaluatorq/integrations/langchain\_integration/wrap\_agent.py        |       80 |       21 |     74% |31-43, 194, 197-215 |
| src/evaluatorq/integrations/langgraph\_integration/\_\_init\_\_.py       |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/langgraph\_integration/target.py             |      192 |       14 |     93% |61, 114, 134-135, 143-147, 249, 268-275 |
| src/evaluatorq/integrations/openai\_agents\_integration/\_\_init\_\_.py  |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/openai\_agents\_integration/target.py        |      156 |       41 |     74% |159-180, 216, 224, 263-277, 289, 291, 308, 313-314 |
| src/evaluatorq/integrations/pydantic\_ai\_integration/\_\_init\_\_.py    |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/pydantic\_ai\_integration/target.py          |       95 |        9 |     91% |122-129, 172 |
| src/evaluatorq/integrations/vercel\_ai\_sdk\_integration/\_\_init\_\_.py |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/vercel\_ai\_sdk\_integration/target.py       |      121 |        8 |     93% |234, 263, 266, 271-272, 276-277, 333 |
| src/evaluatorq/job\_helper.py                                            |       25 |        2 |     92% |    82, 99 |
| src/evaluatorq/llm\_jury.py                                              |      209 |        4 |     98% |145, 284, 740, 849 |
| src/evaluatorq/openresponses/\_\_init\_\_.py                             |       10 |        1 |     90% |        81 |
| src/evaluatorq/openresponses/client.py                                   |        9 |        0 |    100% |           |
| src/evaluatorq/openresponses/convert\_models.py                          |      121 |        3 |     98% |   168-175 |
| src/evaluatorq/openresponses/dataset.py                                  |      159 |       15 |     91% |61, 67, 71, 78, 98, 125, 153, 161, 167, 176-178, 213, 223, 228 |
| src/evaluatorq/openresponses/input\_items.py                             |       57 |        1 |     98% |        94 |
| src/evaluatorq/openresponses/otel\_messages.py                           |      117 |       25 |     79% |56-57, 63-66, 83, 115, 120-121, 136-137, 156, 161-162, 172-183 |
| src/evaluatorq/openresponses/target.py                                   |      143 |        1 |     99% |       341 |
| src/evaluatorq/openresponses/tracing.py                                  |       56 |        7 |     88% |76, 78, 128-143 |
| src/evaluatorq/openresponses/types.py                                    |       91 |        0 |    100% |           |
| src/evaluatorq/pairwise.py                                               |      306 |        5 |     98% |581, 584-585, 702, 917 |
| src/evaluatorq/pairwise\_reports/\_\_init\_\_.py                         |        3 |        0 |    100% |           |
| src/evaluatorq/pairwise\_reports/export\_html.py                         |      114 |        4 |     96% |     93-98 |
| src/evaluatorq/pairwise\_reports/sections.py                             |       56 |        0 |    100% |           |
| src/evaluatorq/pairwise\_run.py                                          |       85 |        4 |     95% |145-149, 152 |
| src/evaluatorq/processings.py                                            |       95 |        3 |     97% |245, 344-346 |
| src/evaluatorq/progress.py                                               |      113 |       47 |     58% |57-97, 101-106, 139, 143-150, 154-159, 168-182, 262-265 |
| src/evaluatorq/ranking.py                                                |      195 |        0 |    100% |           |
| src/evaluatorq/redteam/\_\_init\_\_.py                                   |       35 |        5 |     86% |   255-264 |
| src/evaluatorq/redteam/adaptive/\_\_init\_\_.py                          |        7 |        0 |    100% |           |
| src/evaluatorq/redteam/adaptive/agent\_context.py                        |       12 |       12 |      0% |      3-42 |
| src/evaluatorq/redteam/adaptive/attack\_generator.py                     |       54 |        2 |     96% |  220, 235 |
| src/evaluatorq/redteam/adaptive/blackbox\_classifier.py                  |      131 |       11 |     92% |263-268, 324-329, 389 |
| src/evaluatorq/redteam/adaptive/capability\_classifier.py                |      107 |        4 |     96% |254, 257, 326, 338 |
| src/evaluatorq/redteam/adaptive/evaluator.py                             |       87 |        1 |     99% |       119 |
| src/evaluatorq/redteam/adaptive/objective\_generator.py                  |      142 |       21 |     85% |55, 176, 315-316, 433-447, 490-494, 509, 559-560, 562-563, 666-700 |
| src/evaluatorq/redteam/adaptive/orchestrator.py                          |      389 |       70 |     82% |99-126, 132-139, 148-161, 167-168, 174-188, 193-197, 202, 393-394, 459-465, 620-621, 913, 1003, 1023, 1061-1071 |
| src/evaluatorq/redteam/adaptive/pipeline.py                              |      213 |       29 |     86% |85, 236-240, 269, 273-312, 391, 628-629, 673-674, 762, 770-774 |
| src/evaluatorq/redteam/adaptive/strategy\_planner.py                     |       99 |        3 |     97% |   177-181 |
| src/evaluatorq/redteam/adaptive/strategy\_registry.py                    |      101 |        2 |     98% |  259, 263 |
| src/evaluatorq/redteam/adaptive/tool\_chaining.py                        |       80 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/\_errors.py                              |       23 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/\_retry.py                               |       12 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/base.py                                  |       78 |        5 |     94% |37-40, 140 |
| src/evaluatorq/redteam/backends/openai.py                                |      107 |        4 |     96% |173, 192, 252, 300 |
| src/evaluatorq/redteam/backends/openresponses.py                         |       44 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/orq.py                                   |      305 |       42 |     86% |27-28, 115-130, 325-326, 492, 497, 572-582, 586-598, 628-629, 639, 643, 679, 702, 714-725 |
| src/evaluatorq/redteam/backends/registry.py                              |       57 |        6 |     89% |56, 67-71, 95, 148-149 |
| src/evaluatorq/redteam/cli.py                                            |      360 |      133 |     63% |79, 85, 99-100, 166-168, 180, 194-196, 208, 235, 513, 536, 626-627, 629-630, 637-641, 644-646, 649-651, 654-656, 706-717, 722-727, 750-849, 872-876, 890-891, 919, 922-923, 994-1013 |
| src/evaluatorq/redteam/contracts.py                                      |      820 |       43 |     95% |96, 136-138, 462, 475-476, 489, 515, 667, 671, 691, 1293, 1335-1355, 1361, 1455-1457, 1466, 1552-1558, 1635-1637, 1963, 2162-2171 |
| src/evaluatorq/redteam/delivery\_method\_registry.py                     |       59 |        1 |     98% |       106 |
| src/evaluatorq/redteam/exceptions.py                                     |        5 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/\_\_init\_\_.py                        |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/owasp/\_\_init\_\_.py                  |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/owasp/agent\_evaluators.py             |       44 |        1 |     98% |       963 |
| src/evaluatorq/redteam/frameworks/owasp/evaluatorq\_bridge.py            |      226 |       60 |     73% |79, 108, 124-125, 134-144, 149, 160-168, 223, 228-230, 263-264, 423-424, 444-448, 453-457, 462-500 |
| src/evaluatorq/redteam/frameworks/owasp/evaluators.py                    |       67 |       25 |     63% |138-151, 169, 188, 198-211, 220 |
| src/evaluatorq/redteam/frameworks/owasp/llm\_evaluators.py               |       36 |       19 |     47% |178-374, 548-655, 677-796, 818-936, 957-1075, 1096-1213, 1233 |
| src/evaluatorq/redteam/frameworks/owasp/models.py                        |       41 |        3 |     93% | 11-13, 73 |
| src/evaluatorq/redteam/frameworks/owasp\_asi.py                          |        8 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/owasp\_llm.py                          |        8 |        0 |    100% |           |
| src/evaluatorq/redteam/hooks.py                                          |      378 |       95 |     75% |313, 354, 420-424, 428-440, 475, 484-485, 497, 513, 577-578, 650, 720, 738-821, 825-827, 840-841, 849-852, 854-857, 862-863 |
| src/evaluatorq/redteam/judge.py                                          |        3 |        0 |    100% |           |
| src/evaluatorq/redteam/replay.py                                         |       59 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/\_\_init\_\_.py                           |        5 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/\_utils.py                                |       15 |        2 |     87% |     27-28 |
| src/evaluatorq/redteam/reports/apply.py                                  |       11 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/converters.py                             |      606 |       31 |     95% |157, 203, 219-225, 254-260, 301-304, 342, 344-347, 350, 357, 363-364, 374, 376, 1063-1065 |
| src/evaluatorq/redteam/reports/display.py                                |      129 |       38 |     71% |28, 36, 83-84, 86, 91-98, 103-111, 216-245 |
| src/evaluatorq/redteam/reports/executive\_summary.py                     |       36 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/export\_html.py                           |      725 |      107 |     85% |145, 172, 199, 299-302, 341-342, 381, 387-398, 477, 500, 522, 526, 549, 596, 609, 620, 628-631, 651, 687, 689, 725, 748, 766-770, 772, 774, 779-785, 787-793, 795-801, 803-809, 826-854, 864-891, 899, 924, 943, 949, 969, 1012, 1085, 1114, 1144, 1181, 1221-1223, 1286, 1329, 1359, 1390, 1393, 1396, 1410, 1413, 1415-1416, 1431-1432, 1499, 1501, 1546, 1551 |
| src/evaluatorq/redteam/reports/export\_md.py                             |      432 |      150 |     65% |44-46, 54, 57, 103, 140, 144-147, 150-151, 171, 186, 188, 190, 192, 194, 196, 198, 232, 236-244, 251-290, 295-338, 396, 412, 415, 453, 474-499, 506, 527-575, 580-598, 603-610, 620, 639, 691, 712, 714, 719, 741, 743, 745-746, 758-760, 848 |
| src/evaluatorq/redteam/reports/guidance.py                               |        2 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/recommendations.py                        |      184 |        7 |     96% |76, 221, 223, 251, 302, 305, 417 |
| src/evaluatorq/redteam/reports/sections.py                               |      348 |       16 |     95% |70, 85, 99-100, 102-103, 453-458, 576, 726, 739, 821, 983 |
| src/evaluatorq/redteam/runner.py                                         |     1138 |      121 |     89% |184-186, 223, 311, 385, 392, 687-692, 694-695, 747-748, 778, 799, 891-892, 963-966, 1048-1049, 1098-1102, 1132-1136, 1178, 1248-1250, 1500, 1661, 1669-1671, 1780-1785, 1927-1932, 1940, 2060-2061, 2066-2073, 2091-2092, 2106, 2108-2114, 2139-2149, 2247, 2286, 2292, 2405-2406, 2408-2410, 2509, 2550-2551, 2668-2669, 2735-2738, 2768-2794, 2822-2824, 2832, 2879, 2905, 2928, 2944, 2949, 2951-2957, 3019, 3127, 3185-3186, 3195, 3198-3199 |
| src/evaluatorq/redteam/runtime/\_\_init\_\_.py                           |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/runtime/jobs.py                                   |       99 |        4 |     96% |88, 154, 204-205 |
| src/evaluatorq/redteam/tracing.py                                        |       51 |        6 |     88% |103-105, 142-144 |
| src/evaluatorq/redteam/ui/\_\_init\_\_.py                                |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/ui/colors.py                                      |        2 |        2 |      0% |       3-5 |
| src/evaluatorq/redteam/ui/dashboard.py                                   |     1330 |     1330 |      0% |    8-2653 |
| src/evaluatorq/redteam/utils.py                                          |        9 |        1 |     89% |        27 |
| src/evaluatorq/redteam/vulnerability\_registry.py                        |       73 |        4 |     95% |186, 300, 313, 327 |
| src/evaluatorq/send\_results.py                                          |       59 |        0 |    100% |           |
| src/evaluatorq/simulation/\_\_init\_\_.py                                |       23 |        1 |     96% |       325 |
| src/evaluatorq/simulation/\_config.py                                    |       77 |        0 |    100% |           |
| src/evaluatorq/simulation/\_datapoint\_io.py                             |       51 |        2 |     96% |    68, 92 |
| src/evaluatorq/simulation/\_usage.py                                     |       11 |        0 |    100% |           |
| src/evaluatorq/simulation/adapters.py                                    |       30 |        7 |     77% | 25, 75-81 |
| src/evaluatorq/simulation/agents/\_\_init\_\_.py                         |        4 |        0 |    100% |           |
| src/evaluatorq/simulation/agents/base.py                                 |      213 |       12 |     94% |190, 195, 314-316, 320-321, 539, 560, 632, 645, 723 |
| src/evaluatorq/simulation/agents/judge.py                                |      297 |       14 |     95% |208, 229-235, 247, 260, 266, 274, 441-446, 465-471, 856-861 |
| src/evaluatorq/simulation/agents/user\_simulator.py                      |       36 |       10 |     72% |83, 93-100, 108-114 |
| src/evaluatorq/simulation/api.py                                         |      635 |       54 |     91% |606, 869, 1159, 1219, 1292, 1356, 1422, 1450, 1553-1555, 1650, 1678, 1694-1695, 1765, 1768, 1856, 1882, 1934-1935, 1945, 1949, 1954, 2019, 2026-2043, 2119, 2196-2202, 2205-2208, 2295-2296, 2402-2405, 2412-2415 |
| src/evaluatorq/simulation/cli.py                                         |      666 |      103 |     85% |96-103, 112-113, 142, 146, 148, 157, 161, 166, 169-171, 175, 185, 212, 222, 238-239, 250-270, 275, 709, 746, 748, 777-778, 780-781, 802, 840, 843, 1085, 1136-1137, 1139-1140, 1152, 1369-1370, 1372-1373, 1375, 1524, 1544-1546, 1553, 1577-1588, 1591-1592, 1702-1703, 1719, 1750-1751, 1795, 1843, 1854-1855, 1857, 1910-1911, 1925-1929, 1956, 2046-2051, 2142-2143, 2221 |
| src/evaluatorq/simulation/convert.py                                     |       45 |        0 |    100% |           |
| src/evaluatorq/simulation/evaluators/\_\_init\_\_.py                     |        3 |        0 |    100% |           |
| src/evaluatorq/simulation/evaluators/scorers.py                          |      127 |        3 |     98% |135, 141, 179 |
| src/evaluatorq/simulation/exceptions.py                                  |        8 |        0 |    100% |           |
| src/evaluatorq/simulation/experiments.py                                 |       45 |        0 |    100% |           |
| src/evaluatorq/simulation/generators/\_\_init\_\_.py                     |        5 |        0 |    100% |           |
| src/evaluatorq/simulation/generators/datapoint\_generator.py             |       94 |       44 |     53% |68-69, 92-160, 200-220 |
| src/evaluatorq/simulation/generators/first\_message\_generator.py        |       64 |        2 |     97% |   124-125 |
| src/evaluatorq/simulation/generators/persona\_generator.py               |      128 |       52 |     59% |124, 128-152, 358, 374-377, 389-419 |
| src/evaluatorq/simulation/generators/scenario\_generator.py              |      194 |       60 |     69% |171-179, 183-189, 225, 304-305, 315-316, 384-385, 392, 401-403, 462-465, 474-476, 531-534, 543-545, 558-559, 563-564, 613-616, 625-627, 639-642, 654-663 |
| src/evaluatorq/simulation/hooks.py                                       |      213 |        8 |     96% |243-245, 357, 365, 395, 399, 432, 538 |
| src/evaluatorq/simulation/metrics.py                                     |        5 |        0 |    100% |           |
| src/evaluatorq/simulation/quality/\_\_init\_\_.py                        |        2 |        0 |    100% |           |
| src/evaluatorq/simulation/quality/message\_perturbation.py               |       66 |       42 |     36% |78-83, 87-94, 98-104, 108-111, 115-121, 140-143, 152-153, 165-172 |
| src/evaluatorq/simulation/replay.py                                      |       27 |        2 |     93% |     70-71 |
| src/evaluatorq/simulation/reports/\_\_init\_\_.py                        |        6 |        0 |    100% |           |
| src/evaluatorq/simulation/reports/apply.py                               |       10 |        0 |    100% |           |
| src/evaluatorq/simulation/reports/display.py                             |       79 |        3 |     96% |156-157, 168 |
| src/evaluatorq/simulation/reports/executive\_summary.py                  |       40 |        0 |    100% |           |
| src/evaluatorq/simulation/reports/export\_html.py                        |      313 |        7 |     98% |137, 275, 311, 366, 506, 601, 638 |
| src/evaluatorq/simulation/reports/export\_md.py                          |      260 |       15 |     94% |103, 207, 232-239, 315, 379-380, 385, 449 |
| src/evaluatorq/simulation/reports/recommendations.py                     |       93 |        1 |     99% |       296 |
| src/evaluatorq/simulation/reports/sections.py                            |      275 |        7 |     97% |101, 473-478 |
| src/evaluatorq/simulation/reports/token\_usage.py                        |       37 |        5 |     86% |     66-79 |
| src/evaluatorq/simulation/runner/\_\_init\_\_.py                         |        3 |        0 |    100% |           |
| src/evaluatorq/simulation/runner/simulation.py                           |      466 |       21 |     95% |247, 640, 642, 653, 661, 921, 1041, 1192-1200, 1218-1219, 1221-1222, 1366, 1388 |
| src/evaluatorq/simulation/token\_usage.py                                |        7 |        0 |    100% |           |
| src/evaluatorq/simulation/traces.py                                      |      324 |       13 |     96% |342, 389, 398, 464, 467, 486, 584, 694, 771, 853, 863, 922, 926 |
| src/evaluatorq/simulation/tracing.py                                     |       45 |        6 |     87% |     84-90 |
| src/evaluatorq/simulation/types.py                                       |      272 |        3 |     99% |476, 557-558 |
| src/evaluatorq/simulation/ui/\_\_init\_\_.py                             |        0 |        0 |    100% |           |
| src/evaluatorq/simulation/ui/colors.py                                   |        2 |        0 |    100% |           |
| src/evaluatorq/simulation/ui/dashboard.py                                |      316 |      249 |     21% |79-80, 84, 89-92, 96, 105, 109-135, 144-207, 211-311, 359-426, 430-476, 480-494, 498-502, 519-554, 569-574, 579-584, 595-625 |
| src/evaluatorq/simulation/ui/token\_display.py                           |       28 |        0 |    100% |           |
| src/evaluatorq/simulation/utils/\_\_init\_\_.py                          |        5 |        0 |    100% |           |
| src/evaluatorq/simulation/utils/dataset\_export.py                       |       75 |       28 |     63% |54-56, 61-64, 88-90, 109-115, 130-136, 179-181, 189-214 |
| src/evaluatorq/simulation/utils/extract\_json.py                         |        3 |        0 |    100% |           |
| src/evaluatorq/simulation/utils/prompt\_builders.py                      |       65 |        6 |     91% |45, 52, 59, 64, 80, 124 |
| src/evaluatorq/simulation/utils/run\_store.py                            |      112 |        6 |     95% |72-74, 80-82 |
| src/evaluatorq/simulation/utils/structured\_output.py                    |        3 |        0 |    100% |           |
| src/evaluatorq/simulation/wrap\_agent.py                                 |       29 |        3 |     90% |84, 86, 109 |
| src/evaluatorq/table\_display.py                                         |      148 |       65 |     56% |31, 43, 68, 70, 100, 106, 147, 158, 175-209, 214-232, 242-280 |
| src/evaluatorq/tracing/\_\_init\_\_.py                                   |        4 |        0 |    100% |           |
| src/evaluatorq/tracing/context.py                                        |       34 |        2 |     94% |     50-51 |
| src/evaluatorq/tracing/setup.py                                          |      146 |       39 |     73% |113-123, 144-147, 152, 181, 187-193, 201-211, 252-260, 325-326, 347 |
| src/evaluatorq/tracing/spans.py                                          |       84 |        1 |     99% |       130 |
| src/evaluatorq/types.py                                                  |       86 |        2 |     98% |   32, 288 |
| **TOTAL**                                                                | **27632** | **4167** | **85%** |           |


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