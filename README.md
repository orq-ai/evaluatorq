# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/orq-ai/evaluatorq/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                     |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/evaluatorq/\_\_init\_\_.py                                           |       19 |        0 |    100% |           |
| src/evaluatorq/\_\_main\_\_.py                                           |        3 |        3 |      0% |       1-4 |
| src/evaluatorq/cli.py                                                    |       50 |        2 |     96% |  117, 178 |
| src/evaluatorq/common/\_\_init\_\_.py                                    |        0 |        0 |    100% |           |
| src/evaluatorq/common/apply.py                                           |      134 |        6 |     96% |207, 220-221, 227, 316-317 |
| src/evaluatorq/common/async\_utils.py                                    |       54 |        3 |     94% |60, 92, 97 |
| src/evaluatorq/common/cli\_epilog.py                                     |        6 |        0 |    100% |           |
| src/evaluatorq/common/cli\_errors.py                                     |       18 |        0 |    100% |           |
| src/evaluatorq/common/cli\_help.py                                       |        2 |        0 |    100% |           |
| src/evaluatorq/common/cli\_json.py                                       |        6 |        0 |    100% |           |
| src/evaluatorq/common/cli\_tty.py                                        |        4 |        0 |    100% |           |
| src/evaluatorq/common/cli\_width.py                                      |       10 |        2 |     80% |     19-20 |
| src/evaluatorq/common/content\_filter.py                                 |       29 |        1 |     97% |        81 |
| src/evaluatorq/common/extract\_json.py                                   |       73 |        8 |     89% |29, 37, 111, 130-131, 139-142 |
| src/evaluatorq/common/fields.py                                          |        6 |        0 |    100% |           |
| src/evaluatorq/common/hook\_compose.py                                   |       16 |        0 |    100% |           |
| src/evaluatorq/common/judge.py                                           |      134 |        4 |     97% |110-111, 205, 219 |
| src/evaluatorq/common/jury.py                                            |      299 |        8 |     97% |115, 132, 145-146, 229, 434, 769, 818 |
| src/evaluatorq/common/llm\_call.py                                       |       95 |        2 |     98% |   217-218 |
| src/evaluatorq/common/llm\_client.py                                     |       48 |        0 |    100% |           |
| src/evaluatorq/common/messages.py                                        |       16 |        0 |    100% |           |
| src/evaluatorq/common/orq\_client.py                                     |       14 |        1 |     93% |        45 |
| src/evaluatorq/common/output\_adapters.py                                |       99 |        8 |     92% |37-39, 58, 114, 123-125 |
| src/evaluatorq/common/replay.py                                          |       85 |        5 |     94% |81-82, 101, 178, 182 |
| src/evaluatorq/common/reports/\_\_init\_\_.py                            |        9 |        0 |    100% |           |
| src/evaluatorq/common/reports/console.py                                 |       27 |        0 |    100% |           |
| src/evaluatorq/common/reports/executive\_summary.py                      |       34 |        2 |     94% |     91-92 |
| src/evaluatorq/common/reports/html\_helpers.py                           |      189 |       12 |     94% |57-59, 116-118, 202, 241, 267, 377, 404, 431 |
| src/evaluatorq/common/reports/md\_helpers.py                             |       46 |        1 |     98% |       112 |
| src/evaluatorq/common/reports/palette.py                                 |        9 |        0 |    100% |           |
| src/evaluatorq/common/reports/render.py                                  |       54 |        2 |     96% |     82-86 |
| src/evaluatorq/common/reports/rich\_styles.py                            |        8 |        0 |    100% |           |
| src/evaluatorq/common/reports/vega.py                                    |      114 |        5 |     96% |58-59, 92-94 |
| src/evaluatorq/common/retry.py                                           |       47 |        2 |     96% |    78, 97 |
| src/evaluatorq/common/run\_manifest.py                                   |      142 |       11 |     92% |77-80, 109, 120, 208-211, 285-289 |
| src/evaluatorq/common/run\_store\_dir.py                                 |        8 |        0 |    100% |           |
| src/evaluatorq/common/sanitize.py                                        |       12 |        0 |    100% |           |
| src/evaluatorq/common/structured\_output.py                              |       58 |        5 |     91% |138, 142, 145, 150, 154 |
| src/evaluatorq/common/target\_call.py                                    |       96 |        1 |     99% |        88 |
| src/evaluatorq/common/template\_engine.py                                |       60 |        2 |     97% |    68, 72 |
| src/evaluatorq/common/thread\_context.py                                 |       62 |        1 |     98% |        75 |
| src/evaluatorq/common/tracing.py                                         |      303 |       28 |     91% |155-166, 192, 204-206, 213, 409, 446-448, 531-533, 570-571, 583-584 |
| src/evaluatorq/common/ui/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| src/evaluatorq/common/ui/launch.py                                       |       20 |       12 |     40% |23, 35-39, 55-79 |
| src/evaluatorq/contracts.py                                              |      495 |       17 |     97% |58, 81, 115-117, 712, 781, 783, 785, 787, 837, 842, 846, 851, 853, 969, 1221, 1525 |
| src/evaluatorq/dashboard/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| src/evaluatorq/dashboard/\_compat.py                                     |       23 |       14 |     39% | 37-51, 69 |
| src/evaluatorq/dashboard/app.py                                          |      277 |       39 |     86% |88-89, 92-93, 263-264, 284-285, 322, 325, 328, 331-333, 349, 353-355, 382, 411, 415-417, 448, 451, 459-461, 487, 498, 506-508, 522, 553, 556, 564-566 |
| src/evaluatorq/dashboard/apply\_ui.py                                    |      292 |       31 |     89% |77, 153-158, 287, 289, 473-483, 553-554, 560-562, 594-599, 603, 642-643, 651, 674, 681, 705, 710, 764, 807 |
| src/evaluatorq/dashboard/filter\_request.py                              |       13 |        1 |     92% |        42 |
| src/evaluatorq/dashboard/filters.py                                      |      169 |       13 |     92% |135, 145-146, 160, 166, 184, 191, 244-245, 299, 305, 347, 414 |
| src/evaluatorq/dashboard/launch.py                                       |       48 |       10 |     79% |53-54, 81-89, 124 |
| src/evaluatorq/dashboard/library.py                                      |      184 |       13 |     93% |138-142, 162-163, 173, 209, 259-260, 296-297, 299 |
| src/evaluatorq/dashboard/metrics.py                                      |      668 |       62 |     91% |102-103, 109-110, 122-123, 141, 343-345, 362, 370-371, 374-375, 414, 417-418, 461-462, 465-466, 486-487, 490-491, 504-505, 508-509, 512-513, 664, 728, 732, 757, 762, 770-771, 866-867, 884-885, 888-889, 956, 1072, 1086-1093, 1126-1127, 1181, 1187-1188, 1190-1191 |
| src/evaluatorq/dashboard/orq\_links.py                                   |       40 |        0 |    100% |           |
| src/evaluatorq/dashboard/orq\_workspace.py                               |       15 |        0 |    100% |           |
| src/evaluatorq/dashboard/redteam\_charts.py                              |      189 |       20 |     89% |80-82, 94, 99-104, 114, 117, 176, 178, 207, 346, 359, 405, 419-420 |
| src/evaluatorq/dashboard/redteam\_transcripts.py                         |      116 |        6 |     95% |74, 143, 145-146, 181, 279 |
| src/evaluatorq/dashboard/redteam\_views.py                               |       90 |       11 |     88% |56-57, 60, 63-65, 126-127, 134, 147-148 |
| src/evaluatorq/dashboard/report\_kit.py                                  |      230 |       10 |     96% |121, 125, 245, 279, 317, 434-435, 495-496, 540 |
| src/evaluatorq/dashboard/report\_tabs.py                                 |     1056 |       72 |     93% |62, 299, 325, 459, 502, 927-928, 932, 948, 1120-1139, 1167, 1201, 1220, 1223-1228, 1255, 1357, 1390, 1410-1440, 1445-1467, 1487, 1495, 1577, 1579, 1655, 1829, 2099, 2101, 2294, 2300, 2321, 2371, 2378, 2437 |
| src/evaluatorq/dashboard/shell.py                                        |       52 |        4 |     92% |60-61, 78-79 |
| src/evaluatorq/dashboard/sim\_compare.py                                 |      257 |       11 |     96% |156, 300, 308, 322, 335, 410, 518, 521, 587, 593, 600 |
| src/evaluatorq/dashboard/sim\_views.py                                   |      217 |       10 |     95% |67, 70-72, 96, 255, 257, 259, 572-573 |
| src/evaluatorq/dashboard/styles.py                                       |       12 |        0 |    100% |           |
| src/evaluatorq/dashboard/surfaces.py                                     |       66 |        0 |    100% |           |
| src/evaluatorq/dashboard/theme.py                                        |        2 |        0 |    100% |           |
| src/evaluatorq/dashboard/trace\_links.py                                 |       38 |        0 |    100% |           |
| src/evaluatorq/dashboard/view.py                                         |      439 |       33 |     92% |104, 148-149, 331, 542, 549, 624, 681, 878, 919-920, 950-951, 1299-1339 |
| src/evaluatorq/deployment.py                                             |       62 |       35 |     44% |76-78, 122-149, 160-187, 219-227 |
| src/evaluatorq/evaluatorq.py                                             |      149 |       49 |     67% |52, 163, 249-347, 394, 419 |
| src/evaluatorq/evaluators.py                                             |       37 |       18 |     51% |52, 73, 109-144 |
| src/evaluatorq/fetch\_data.py                                            |      139 |       17 |     88% |48, 82-84, 122, 183-184, 188-191, 251, 259, 293, 314, 323, 326-327 |
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
| src/evaluatorq/llm\_jury.py                                              |      206 |        4 |     98% |142, 268, 706, 794 |
| src/evaluatorq/openresponses/\_\_init\_\_.py                             |        4 |        0 |    100% |           |
| src/evaluatorq/openresponses/client.py                                   |        7 |        0 |    100% |           |
| src/evaluatorq/openresponses/convert\_models.py                          |      121 |        3 |     98% |   168-175 |
| src/evaluatorq/openresponses/dataset.py                                  |      159 |       15 |     91% |61, 67, 71, 78, 98, 125, 153, 161, 167, 176-178, 213, 223, 228 |
| src/evaluatorq/openresponses/input\_items.py                             |       34 |        1 |     97% |        34 |
| src/evaluatorq/openresponses/target.py                                   |      118 |        3 |     97% |139, 236, 259 |
| src/evaluatorq/openresponses/tracing.py                                  |       74 |       18 |     76% |40, 67-73, 85, 87, 99-103, 117, 138-145 |
| src/evaluatorq/openresponses/types.py                                    |       91 |        0 |    100% |           |
| src/evaluatorq/pairwise.py                                               |      218 |        5 |     98% |308, 311-312, 425, 592 |
| src/evaluatorq/pairwise\_reports/\_\_init\_\_.py                         |        3 |        0 |    100% |           |
| src/evaluatorq/pairwise\_reports/export\_html.py                         |      114 |        6 |     95% |93-98, 135, 146 |
| src/evaluatorq/pairwise\_reports/sections.py                             |       56 |        0 |    100% |           |
| src/evaluatorq/pairwise\_run.py                                          |       85 |        4 |     95% |145-149, 152 |
| src/evaluatorq/processings.py                                            |       66 |        5 |     92% |51, 87-89, 280-282 |
| src/evaluatorq/progress.py                                               |      101 |       53 |     48% |55-95, 99-104, 137, 141-148, 152-157, 166-180, 213-231 |
| src/evaluatorq/ranking.py                                                |      195 |        0 |    100% |           |
| src/evaluatorq/redteam/\_\_init\_\_.py                                   |       33 |        5 |     85% |   249-258 |
| src/evaluatorq/redteam/adaptive/\_\_init\_\_.py                          |        7 |        0 |    100% |           |
| src/evaluatorq/redteam/adaptive/agent\_context.py                        |       12 |       12 |      0% |      3-42 |
| src/evaluatorq/redteam/adaptive/attack\_generator.py                     |       54 |        2 |     96% |  214, 229 |
| src/evaluatorq/redteam/adaptive/blackbox\_classifier.py                  |      124 |        7 |     94% |299-304, 360 |
| src/evaluatorq/redteam/adaptive/capability\_classifier.py                |      108 |        6 |     94% |165, 184, 255, 258, 323, 335 |
| src/evaluatorq/redteam/adaptive/evaluator.py                             |       90 |        1 |     99% |       117 |
| src/evaluatorq/redteam/adaptive/objective\_generator.py                  |      139 |       33 |     76% |54, 142, 166, 284-307, 419-433, 476-480, 495, 545-546, 548-549, 652-686 |
| src/evaluatorq/redteam/adaptive/orchestrator.py                          |      386 |       70 |     82% |98-125, 131-138, 147-160, 166-167, 173-187, 192-196, 201, 392-393, 459-465, 614-615, 890, 980, 1000, 1038-1048 |
| src/evaluatorq/redteam/adaptive/pipeline.py                              |      220 |       31 |     86% |86, 88, 237-241, 270, 274-313, 391, 400, 637-638, 682-683, 771, 779-783 |
| src/evaluatorq/redteam/adaptive/strategy\_planner.py                     |       99 |        6 |     94% |71-85, 173-177 |
| src/evaluatorq/redteam/adaptive/strategy\_registry.py                    |      101 |        2 |     98% |  259, 263 |
| src/evaluatorq/redteam/adaptive/tool\_chaining.py                        |       78 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/\_errors.py                              |       40 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/base.py                                  |       78 |        5 |     94% |37-40, 140 |
| src/evaluatorq/redteam/backends/openai.py                                |       96 |        5 |     95% |123, 142, 198, 223, 232 |
| src/evaluatorq/redteam/backends/openresponses.py                         |       43 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/orq.py                                   |      299 |       42 |     86% |27-28, 109-124, 298-299, 459, 464, 539-549, 553-565, 595-596, 606, 610, 636, 658, 670-682 |
| src/evaluatorq/redteam/backends/registry.py                              |       53 |        6 |     89% |51, 62-66, 90, 119-120 |
| src/evaluatorq/redteam/cli.py                                            |      360 |      133 |     63% |78, 84, 98-99, 165-167, 179, 193-195, 207, 234, 458, 481, 565-566, 568-569, 576-580, 583-585, 588-590, 593-595, 645-656, 661-666, 689-788, 811-815, 829-830, 858, 861-862, 933-952 |
| src/evaluatorq/redteam/contracts.py                                      |      770 |       46 |     94% |92, 132-134, 266-267, 455, 468-469, 482, 508, 618, 626, 630, 650, 1076, 1118-1138, 1144, 1238-1240, 1249, 1335-1341, 1418-1420, 1713, 1912-1921 |
| src/evaluatorq/redteam/delivery\_method\_registry.py                     |       59 |        1 |     98% |       110 |
| src/evaluatorq/redteam/exceptions.py                                     |        5 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/\_\_init\_\_.py                        |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/owasp/\_\_init\_\_.py                  |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/owasp/agent\_evaluators.py             |       44 |        1 |     98% |       961 |
| src/evaluatorq/redteam/frameworks/owasp/evaluatorq\_bridge.py            |      220 |       61 |     72% |79, 108, 124-125, 134-144, 149, 160-168, 209, 214-216, 240-241, 294, 380-381, 401-405, 410-414, 419-455 |
| src/evaluatorq/redteam/frameworks/owasp/evaluators.py                    |       67 |       25 |     63% |138-151, 169, 188, 198-211, 220 |
| src/evaluatorq/redteam/frameworks/owasp/llm\_evaluators.py               |       36 |       19 |     47% |178-374, 548-655, 677-796, 818-936, 957-1075, 1096-1213, 1233 |
| src/evaluatorq/redteam/frameworks/owasp/models.py                        |       41 |        3 |     93% | 11-13, 73 |
| src/evaluatorq/redteam/frameworks/owasp\_asi.py                          |        8 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/owasp\_llm.py                          |        8 |        0 |    100% |           |
| src/evaluatorq/redteam/hooks.py                                          |      378 |       95 |     75% |313, 354, 424-428, 432-444, 479, 488-489, 501, 517, 581-582, 654, 724, 742-825, 829-831, 844-845, 853-856, 858-861, 866-867 |
| src/evaluatorq/redteam/judge.py                                          |        3 |        0 |    100% |           |
| src/evaluatorq/redteam/replay.py                                         |       59 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/\_\_init\_\_.py                           |        5 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/\_utils.py                                |       15 |        2 |     87% |     27-28 |
| src/evaluatorq/redteam/reports/apply.py                                  |       11 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/converters.py                             |      550 |       35 |     94% |115, 157, 200-206, 241-244, 282, 284-287, 290, 297, 303-304, 312-318, 496, 634-638, 946-948 |
| src/evaluatorq/redteam/reports/display.py                                |      129 |       38 |     71% |28, 36, 82-83, 85, 90-97, 102-110, 215-244 |
| src/evaluatorq/redteam/reports/executive\_summary.py                     |       36 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/export\_html.py                           |      720 |      111 |     85% |142, 169, 196, 296-299, 336, 338-339, 378, 384-395, 474, 497, 518-538, 543-595, 606, 617, 625-628, 648, 684, 686, 722, 745, 763-767, 769, 771, 776-782, 784-790, 792-798, 800-806, 826, 838, 865, 873, 893-902, 912, 937, 956, 962, 982, 1016, 1089, 1118, 1148, 1185, 1225-1227, 1290, 1333, 1363, 1394, 1397, 1400, 1414, 1417, 1419-1420, 1435-1436, 1503, 1505, 1550, 1555 |
| src/evaluatorq/redteam/reports/export\_md.py                             |      430 |      142 |     67% |42-44, 52, 55, 101, 138, 142-145, 148-149, 169, 184, 186, 188, 190, 192, 194, 196, 230, 234-242, 249-288, 293-336, 394, 410, 413, 451, 476, 498, 507, 528-576, 581-599, 604-611, 621, 634, 686, 707, 709, 714, 736, 738, 740-741, 753-755, 843 |
| src/evaluatorq/redteam/reports/guidance.py                               |        2 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/recommendations.py                        |       76 |        2 |     97% |   51, 186 |
| src/evaluatorq/redteam/reports/sections.py                               |      348 |       25 |     93% |81, 96, 110-111, 113-114, 329-357, 451-456, 574, 820, 989 |
| src/evaluatorq/redteam/runner.py                                         |     1095 |      127 |     88% |177-179, 216, 304, 378, 385, 627-628, 637-642, 644-645, 695-696, 726, 839-840, 919-922, 1003-1004, 1045-1049, 1083-1087, 1118, 1148-1150, 1206, 1382, 1543, 1551-1553, 1662-1667, 1883-1884, 1889-1896, 1904-1905, 1925, 1927-1933, 1958-1968, 2066, 2105, 2111, 2224-2225, 2227-2229, 2290, 2328, 2373-2374, 2447-2457, 2491-2492, 2558-2561, 2592-2618, 2646-2648, 2656, 2703, 2729, 2752, 2768, 2773, 2775-2781, 2843, 2951, 3121-3122, 3131, 3134-3135 |
| src/evaluatorq/redteam/runtime/\_\_init\_\_.py                           |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/runtime/jobs.py                                   |      122 |        9 |     93% |79, 127, 148, 191-193, 200, 243-244 |
| src/evaluatorq/redteam/tracing.py                                        |       51 |        6 |     88% |103-105, 142-144 |
| src/evaluatorq/redteam/ui/\_\_init\_\_.py                                |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/ui/colors.py                                      |        2 |        2 |      0% |       3-5 |
| src/evaluatorq/redteam/ui/dashboard.py                                   |     1345 |     1345 |      0% |    8-2688 |
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
| src/evaluatorq/simulation/api.py                                         |      559 |       53 |     91% |408, 580, 828, 882, 946, 1007, 1058, 1075, 1177-1179, 1278, 1303-1304, 1310, 1425, 1451, 1503-1504, 1514, 1518, 1523, 1587, 1594-1611, 1678, 1748-1754, 1757-1760, 1800-1801, 1803, 1905-1908, 1915-1918 |
| src/evaluatorq/simulation/cli.py                                         |      667 |      126 |     81% |92-99, 108-109, 138, 142, 144, 153, 157, 162, 165-167, 171, 181, 208, 218, 234-235, 246-264, 269, 276, 709, 746, 748, 776-777, 779-780, 801, 838, 841, 1048, 1095-1096, 1098-1099, 1111, 1326-1327, 1329-1330, 1332, 1443-1503, 1606-1607, 1623, 1654-1655, 1699, 1747, 1758-1759, 1761, 1814-1815, 1829-1833, 1860, 1950-1955, 2046-2047, 2125 |
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
| src/evaluatorq/simulation/reports/\_\_init\_\_.py                        |        6 |        0 |    100% |           |
| src/evaluatorq/simulation/reports/apply.py                               |       10 |        0 |    100% |           |
| src/evaluatorq/simulation/reports/display.py                             |       79 |        3 |     96% |142-143, 154 |
| src/evaluatorq/simulation/reports/executive\_summary.py                  |       35 |        0 |    100% |           |
| src/evaluatorq/simulation/reports/export\_html.py                        |      302 |        5 |     98% |137, 303, 358, 498, 607 |
| src/evaluatorq/simulation/reports/export\_md.py                          |      251 |       13 |     95% |103, 207, 232-239, 315, 370, 434 |
| src/evaluatorq/simulation/reports/recommendations.py                     |       67 |        2 |     97% |   61, 189 |
| src/evaluatorq/simulation/reports/sections.py                            |      258 |        1 |     99% |        94 |
| src/evaluatorq/simulation/reports/token\_usage.py                        |       26 |        0 |    100% |           |
| src/evaluatorq/simulation/runner/\_\_init\_\_.py                         |        3 |        0 |    100% |           |
| src/evaluatorq/simulation/runner/simulation.py                           |      312 |       26 |     92% |152-153, 261, 272, 418-424, 441-447, 519, 619, 743-751, 769-770, 772-773, 840, 862 |
| src/evaluatorq/simulation/token\_usage.py                                |        7 |        0 |    100% |           |
| src/evaluatorq/simulation/traces.py                                      |      220 |       13 |     94% |105, 110, 119, 185, 188, 294, 369, 401-405, 415, 461, 488, 492 |
| src/evaluatorq/simulation/tracing.py                                     |       69 |       12 |     83% |67-73, 117-123 |
| src/evaluatorq/simulation/types.py                                       |      216 |        2 |     99% |   362-363 |
| src/evaluatorq/simulation/ui/\_\_init\_\_.py                             |        0 |        0 |    100% |           |
| src/evaluatorq/simulation/ui/colors.py                                   |        2 |        2 |      0% |       7-9 |
| src/evaluatorq/simulation/ui/dashboard.py                                |      300 |      300 |      0% |    13-590 |
| src/evaluatorq/simulation/ui/token\_display.py                           |       28 |        0 |    100% |           |
| src/evaluatorq/simulation/utils/\_\_init\_\_.py                          |        5 |        0 |    100% |           |
| src/evaluatorq/simulation/utils/dataset\_export.py                       |       75 |       28 |     63% |54-56, 61-64, 88-90, 109-115, 130-136, 179-181, 189-214 |
| src/evaluatorq/simulation/utils/extract\_json.py                         |        3 |        0 |    100% |           |
| src/evaluatorq/simulation/utils/prompt\_builders.py                      |       65 |        6 |     91% |45, 52, 59, 64, 80, 124 |
| src/evaluatorq/simulation/utils/run\_store.py                            |      112 |        6 |     95% |72-74, 80-82 |
| src/evaluatorq/simulation/utils/structured\_output.py                    |        3 |        0 |    100% |           |
| src/evaluatorq/simulation/wrap\_agent.py                                 |       29 |        3 |     90% |75, 77, 98 |
| src/evaluatorq/table\_display.py                                         |      147 |       89 |     39% |31, 36-76, 100, 106, 145, 156, 173-207, 212-230, 240-278 |
| src/evaluatorq/tracing/\_\_init\_\_.py                                   |        4 |        0 |    100% |           |
| src/evaluatorq/tracing/context.py                                        |       34 |        2 |     94% |     50-51 |
| src/evaluatorq/tracing/setup.py                                          |      136 |       39 |     71% |103-113, 134-137, 142, 171, 177-183, 191-201, 235-243, 292-293, 314 |
| src/evaluatorq/tracing/spans.py                                          |      112 |       13 |     88% |138, 149-156, 203-210 |
| src/evaluatorq/types.py                                                  |       85 |        2 |     98% |   32, 230 |
| **TOTAL**                                                                | **25277** | **4551** | **82%** |           |


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