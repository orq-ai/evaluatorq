# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/orq-ai/evaluatorq/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                     |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/evaluatorq/\_\_init\_\_.py                                           |       19 |        0 |    100% |           |
| src/evaluatorq/\_\_main\_\_.py                                           |        3 |        3 |      0% |       1-4 |
| src/evaluatorq/cli.py                                                    |       50 |        2 |     96% |  120, 181 |
| src/evaluatorq/common/\_\_init\_\_.py                                    |        0 |        0 |    100% |           |
| src/evaluatorq/common/apply.py                                           |      137 |        6 |     96% |208, 221-222, 228, 329-330 |
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
| src/evaluatorq/common/llm\_call.py                                       |      168 |        1 |     99% |       393 |
| src/evaluatorq/common/llm\_client.py                                     |       48 |        0 |    100% |           |
| src/evaluatorq/common/llm\_limit.py                                      |       31 |        0 |    100% |           |
| src/evaluatorq/common/messages.py                                        |       16 |        0 |    100% |           |
| src/evaluatorq/common/model\_catalogue.py                                |      150 |        5 |     97% |223, 226, 282, 289, 320 |
| src/evaluatorq/common/orq\_client.py                                     |       14 |        1 |     93% |        45 |
| src/evaluatorq/common/output\_adapters.py                                |       99 |        8 |     92% |37-39, 58, 114, 123-125 |
| src/evaluatorq/common/parallelism.py                                     |       10 |        0 |    100% |           |
| src/evaluatorq/common/prompt\_cache.py                                   |       73 |        0 |    100% |           |
| src/evaluatorq/common/recommendations.py                                 |       16 |        0 |    100% |           |
| src/evaluatorq/common/replay.py                                          |       85 |        5 |     94% |81-82, 101, 178, 182 |
| src/evaluatorq/common/reports/\_\_init\_\_.py                            |        9 |        0 |    100% |           |
| src/evaluatorq/common/reports/console.py                                 |       27 |        0 |    100% |           |
| src/evaluatorq/common/reports/executive\_summary.py                      |       41 |        2 |     95% |    99-100 |
| src/evaluatorq/common/reports/html\_helpers.py                           |      189 |       12 |     94% |57-59, 116-118, 202, 241, 267, 377, 404, 431 |
| src/evaluatorq/common/reports/md\_helpers.py                             |       52 |        1 |     98% |       134 |
| src/evaluatorq/common/reports/palette.py                                 |        9 |        0 |    100% |           |
| src/evaluatorq/common/reports/render.py                                  |       54 |        2 |     96% |     82-86 |
| src/evaluatorq/common/reports/rich\_styles.py                            |        8 |        0 |    100% |           |
| src/evaluatorq/common/reports/vega.py                                    |      121 |        7 |     94% |58-59, 93-102 |
| src/evaluatorq/common/responses.py                                       |       37 |        8 |     78% |22-24, 72-76 |
| src/evaluatorq/common/retry.py                                           |       57 |        4 |     93% |102, 134, 136, 138 |
| src/evaluatorq/common/run\_manifest.py                                   |      142 |       11 |     92% |77-80, 109, 120, 208-211, 285-289 |
| src/evaluatorq/common/run\_store\_dir.py                                 |        8 |        0 |    100% |           |
| src/evaluatorq/common/sanitize.py                                        |       12 |        0 |    100% |           |
| src/evaluatorq/common/structured\_output.py                              |      247 |        6 |     98% |163, 367-368, 482, 545, 615 |
| src/evaluatorq/common/target\_call.py                                    |      117 |        0 |    100% |           |
| src/evaluatorq/common/template\_engine.py                                |       60 |        2 |     97% |    68, 72 |
| src/evaluatorq/common/thread\_context.py                                 |       62 |        1 |     98% |        75 |
| src/evaluatorq/common/tracing.py                                         |      304 |       27 |     91% |168-179, 217-219, 226, 422, 459-461, 537-539, 576-577, 589-590 |
| src/evaluatorq/common/ui/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| src/evaluatorq/common/ui/launch.py                                       |       20 |       12 |     40% |23, 35-39, 55-79 |
| src/evaluatorq/contracts.py                                              |      539 |       17 |     97% |58, 81, 134-136, 301, 868, 944, 948, 950, 1000, 1005, 1009, 1014, 1016, 1135, 1391, 1705 |
| src/evaluatorq/dashboard/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| src/evaluatorq/dashboard/\_compat.py                                     |       23 |       14 |     39% | 39-53, 71 |
| src/evaluatorq/dashboard/app.py                                          |      277 |       39 |     86% |84-85, 88-89, 259-260, 280-281, 318, 321, 324, 327-329, 345, 349-351, 378, 407, 411-413, 444, 447, 455-457, 483, 494, 502-504, 518, 549, 552, 560-562 |
| src/evaluatorq/dashboard/apply\_ui.py                                    |      293 |       29 |     90% |78, 154-159, 474-484, 554-555, 561-563, 595-600, 604, 643-644, 652, 675, 682, 706, 711, 765, 808 |
| src/evaluatorq/dashboard/filter\_request.py                              |       13 |        1 |     92% |        42 |
| src/evaluatorq/dashboard/filters.py                                      |      169 |       13 |     92% |135, 145-146, 160, 166, 184, 191, 244-245, 299, 305, 347, 414 |
| src/evaluatorq/dashboard/launch.py                                       |       48 |       10 |     79% |53-54, 81-89, 124 |
| src/evaluatorq/dashboard/library.py                                      |      184 |       13 |     93% |138-142, 162-163, 173, 209, 259-260, 296-297, 299 |
| src/evaluatorq/dashboard/metrics.py                                      |      682 |       59 |     91% |102-103, 109-110, 141, 341-343, 360, 368-369, 372-373, 412, 415-416, 461-462, 465-466, 486-487, 490-491, 504-505, 508-509, 512-513, 672, 736, 740, 765, 770, 778-779, 874-875, 892-893, 896-897, 964, 1094-1101, 1134-1135, 1200, 1206-1207, 1209-1210 |
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
| src/evaluatorq/deployment.py                                             |       70 |       17 |     76% |78-80, 126-153, 163, 181, 187, 245-253 |
| src/evaluatorq/evaluatorq.py                                             |      182 |        6 |     97% |63, 73, 214, 313, 319, 503 |
| src/evaluatorq/evaluators.py                                             |       37 |       18 |     51% |54, 75, 113-148 |
| src/evaluatorq/fetch\_data.py                                            |      139 |       17 |     88% |48, 82-84, 122, 183-184, 188-191, 251, 259, 293, 314, 323, 326-327 |
| src/evaluatorq/integrations/\_\_init\_\_.py                              |        6 |        3 |     50% |     39-41 |
| src/evaluatorq/integrations/callable\_integration/\_\_init\_\_.py        |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/callable\_integration/target.py              |       55 |        0 |    100% |           |
| src/evaluatorq/integrations/crewai\_integration/\_\_init\_\_.py          |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/crewai\_integration/target.py                |       66 |        8 |     88% |   135-146 |
| src/evaluatorq/integrations/langchain\_integration/\_\_init\_\_.py       |        4 |        0 |    100% |           |
| src/evaluatorq/integrations/langchain\_integration/convert.py            |      159 |       49 |     69% |120-126, 138, 175-202, 220, 222-223, 228, 242, 244, 291-293, 299, 307-311, 321, 327, 339-349, 365, 376, 397, 400-402 |
| src/evaluatorq/integrations/langchain\_integration/types.py              |        4 |        0 |    100% |           |
| src/evaluatorq/integrations/langchain\_integration/wrap\_agent.py        |       75 |       21 |     72% |30-42, 173, 176-194 |
| src/evaluatorq/integrations/langgraph\_integration/\_\_init\_\_.py       |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/langgraph\_integration/target.py             |      172 |       15 |     91% |58, 84, 104-105, 113-117, 207, 226-233, 282 |
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
| src/evaluatorq/openresponses/input\_items.py                             |       50 |        1 |     98% |        67 |
| src/evaluatorq/openresponses/target.py                                   |      143 |        2 |     99% |  311, 341 |
| src/evaluatorq/openresponses/tracing.py                                  |       78 |       19 |     76% |40, 67-73, 85, 87, 99-103, 121, 123, 144-151 |
| src/evaluatorq/openresponses/types.py                                    |       91 |        0 |    100% |           |
| src/evaluatorq/pairwise.py                                               |      306 |        5 |     98% |581, 584-585, 702, 917 |
| src/evaluatorq/pairwise\_reports/\_\_init\_\_.py                         |        3 |        0 |    100% |           |
| src/evaluatorq/pairwise\_reports/export\_html.py                         |      114 |        4 |     96% |     93-98 |
| src/evaluatorq/pairwise\_reports/sections.py                             |       56 |        0 |    100% |           |
| src/evaluatorq/pairwise\_run.py                                          |       85 |        4 |     95% |145-149, 152 |
| src/evaluatorq/processings.py                                            |       77 |        4 |     95% |155, 205, 304-306 |
| src/evaluatorq/progress.py                                               |      113 |       47 |     58% |57-97, 101-106, 139, 143-150, 154-159, 168-182, 262-265 |
| src/evaluatorq/ranking.py                                                |      195 |        0 |    100% |           |
| src/evaluatorq/redteam/\_\_init\_\_.py                                   |       35 |        5 |     86% |   255-264 |
| src/evaluatorq/redteam/adaptive/\_\_init\_\_.py                          |        7 |        0 |    100% |           |
| src/evaluatorq/redteam/adaptive/agent\_context.py                        |       12 |       12 |      0% |      3-42 |
| src/evaluatorq/redteam/adaptive/attack\_generator.py                     |       54 |        2 |     96% |  220, 235 |
| src/evaluatorq/redteam/adaptive/blackbox\_classifier.py                  |      131 |       11 |     92% |263-268, 324-329, 389 |
| src/evaluatorq/redteam/adaptive/capability\_classifier.py                |      107 |        4 |     96% |254, 257, 326, 338 |
| src/evaluatorq/redteam/adaptive/evaluator.py                             |       87 |        1 |     99% |       119 |
| src/evaluatorq/redteam/adaptive/objective\_generator.py                  |      139 |       21 |     85% |54, 166, 305-306, 423-437, 480-484, 499, 549-550, 552-553, 656-690 |
| src/evaluatorq/redteam/adaptive/orchestrator.py                          |      389 |       70 |     82% |99-126, 132-139, 148-161, 167-168, 174-188, 193-197, 202, 393-394, 459-465, 614-615, 907, 997, 1017, 1055-1065 |
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
| src/evaluatorq/redteam/backends/orq.py                                   |      302 |       42 |     86% |27-28, 110-125, 313-314, 480, 485, 560-570, 574-586, 616-617, 627, 631, 667, 690, 702-713 |
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
| src/evaluatorq/redteam/reports/export\_html.py                           |      725 |      107 |     85% |147, 174, 201, 301-304, 343-344, 383, 389-400, 479, 502, 524, 528, 551, 598, 611, 622, 630-633, 653, 689, 691, 727, 750, 768-772, 774, 776, 781-787, 789-795, 797-803, 805-811, 828-856, 866-893, 901, 926, 945, 951, 971, 1014, 1087, 1116, 1146, 1183, 1223-1225, 1288, 1331, 1361, 1392, 1395, 1398, 1412, 1415, 1417-1418, 1433-1434, 1501, 1503, 1548, 1553 |
| src/evaluatorq/redteam/reports/export\_md.py                             |      432 |      150 |     65% |44-46, 54, 57, 103, 140, 144-147, 150-151, 171, 186, 188, 190, 192, 194, 196, 198, 232, 236-244, 251-290, 295-338, 396, 412, 415, 453, 474-499, 506, 527-575, 580-598, 603-610, 620, 639, 691, 712, 714, 719, 741, 743, 745-746, 758-760, 848 |
| src/evaluatorq/redteam/reports/guidance.py                               |        2 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/recommendations.py                        |      184 |        7 |     96% |76, 221, 223, 251, 302, 305, 416 |
| src/evaluatorq/redteam/reports/sections.py                               |      349 |       16 |     95% |81, 96, 110-111, 113-114, 464-469, 587, 738, 751, 833, 995 |
| src/evaluatorq/redteam/runner.py                                         |     1132 |      121 |     89% |181-183, 220, 308, 382, 389, 684-689, 691-692, 744-745, 775, 796, 888-889, 960-963, 1045-1046, 1095-1099, 1135-1139, 1181, 1211-1213, 1463, 1624, 1632-1634, 1743-1748, 1890-1895, 1903, 2023-2024, 2029-2036, 2054-2055, 2069, 2071-2077, 2102-2112, 2210, 2249, 2255, 2368-2369, 2371-2373, 2472, 2513-2514, 2631-2632, 2698-2701, 2731-2757, 2785-2787, 2795, 2842, 2868, 2891, 2907, 2912, 2914-2920, 2982, 3090, 3148-3149, 3158, 3161-3162 |
| src/evaluatorq/redteam/runtime/\_\_init\_\_.py                           |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/runtime/jobs.py                                   |       96 |        4 |     96% |83, 142, 192-193 |
| src/evaluatorq/redteam/tracing.py                                        |       51 |        6 |     88% |103-105, 142-144 |
| src/evaluatorq/redteam/ui/\_\_init\_\_.py                                |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/ui/colors.py                                      |        2 |        2 |      0% |       3-5 |
| src/evaluatorq/redteam/ui/dashboard.py                                   |     1330 |     1330 |      0% |    8-2659 |
| src/evaluatorq/redteam/utils.py                                          |        9 |        1 |     89% |        27 |
| src/evaluatorq/redteam/vulnerability\_registry.py                        |       73 |        4 |     95% |186, 300, 313, 327 |
| src/evaluatorq/send\_results.py                                          |       59 |        0 |    100% |           |
| src/evaluatorq/simulation/\_\_init\_\_.py                                |       23 |        1 |     96% |       325 |
| src/evaluatorq/simulation/\_config.py                                    |       62 |        0 |    100% |           |
| src/evaluatorq/simulation/\_datapoint\_io.py                             |       51 |        2 |     96% |    68, 92 |
| src/evaluatorq/simulation/\_usage.py                                     |       11 |        0 |    100% |           |
| src/evaluatorq/simulation/adapters.py                                    |       24 |        7 |     71% | 25, 57-63 |
| src/evaluatorq/simulation/agents/\_\_init\_\_.py                         |        4 |        0 |    100% |           |
| src/evaluatorq/simulation/agents/base.py                                 |      204 |       15 |     93% |170, 174, 176, 178, 180, 254-256, 260-261, 478, 499, 570, 583, 661 |
| src/evaluatorq/simulation/agents/judge.py                                |      299 |       17 |     94% |208, 229-235, 247, 260, 266, 274, 441-446, 465-471, 666-668, 846-851 |
| src/evaluatorq/simulation/agents/user\_simulator.py                      |       36 |       10 |     72% |74, 84-91, 99-105 |
| src/evaluatorq/simulation/api.py                                         |      622 |       54 |     91% |577, 837, 1122, 1176, 1240, 1301, 1366, 1394, 1497-1499, 1594, 1622, 1637-1638, 1708, 1711, 1799, 1825, 1877-1878, 1888, 1892, 1897, 1961, 1968-1985, 2059, 2135-2141, 2144-2147, 2234-2235, 2339-2342, 2349-2352 |
| src/evaluatorq/simulation/cli.py                                         |      665 |      103 |     85% |95-102, 111-112, 141, 145, 147, 156, 160, 165, 168-170, 174, 184, 211, 221, 237-238, 249-269, 274, 708, 745, 747, 776-777, 779-780, 801, 839, 842, 1084, 1135-1136, 1138-1139, 1151, 1368-1369, 1371-1372, 1374, 1523, 1543-1545, 1552, 1576-1587, 1590-1591, 1701-1702, 1718, 1749-1750, 1794, 1842, 1853-1854, 1856, 1909-1910, 1924-1928, 1955, 2045-2050, 2141-2142, 2220 |
| src/evaluatorq/simulation/convert.py                                     |       44 |        0 |    100% |           |
| src/evaluatorq/simulation/evaluators/\_\_init\_\_.py                     |        3 |        0 |    100% |           |
| src/evaluatorq/simulation/evaluators/scorers.py                          |      122 |        3 |     98% |119, 125, 163 |
| src/evaluatorq/simulation/exceptions.py                                  |        8 |        0 |    100% |           |
| src/evaluatorq/simulation/experiments.py                                 |       44 |        0 |    100% |           |
| src/evaluatorq/simulation/generators/\_\_init\_\_.py                     |        5 |        0 |    100% |           |
| src/evaluatorq/simulation/generators/datapoint\_generator.py             |       91 |       44 |     52% |63-64, 87-155, 195-215 |
| src/evaluatorq/simulation/generators/first\_message\_generator.py        |       59 |        2 |     97% |   101-102 |
| src/evaluatorq/simulation/generators/persona\_generator.py               |      127 |       52 |     59% |116, 120-144, 350, 366-369, 381-411 |
| src/evaluatorq/simulation/generators/scenario\_generator.py              |      194 |       60 |     69% |174-182, 186-192, 218, 297-298, 308-309, 377-378, 385, 394-396, 455-458, 467-469, 524-527, 536-538, 551-552, 556-557, 606-609, 618-620, 632-635, 647-656 |
| src/evaluatorq/simulation/hooks.py                                       |      213 |        8 |     96% |243-245, 357, 365, 395, 399, 432, 538 |
| src/evaluatorq/simulation/metrics.py                                     |        5 |        0 |    100% |           |
| src/evaluatorq/simulation/quality/\_\_init\_\_.py                        |        2 |        0 |    100% |           |
| src/evaluatorq/simulation/quality/message\_perturbation.py               |       66 |       42 |     36% |78-83, 87-94, 98-104, 108-111, 115-121, 140-143, 152-153, 165-172 |
| src/evaluatorq/simulation/replay.py                                      |       27 |        2 |     93% |     70-71 |
| src/evaluatorq/simulation/reports/\_\_init\_\_.py                        |        6 |        0 |    100% |           |
| src/evaluatorq/simulation/reports/apply.py                               |       10 |        0 |    100% |           |
| src/evaluatorq/simulation/reports/display.py                             |       79 |        3 |     96% |156-157, 168 |
| src/evaluatorq/simulation/reports/executive\_summary.py                  |       39 |        0 |    100% |           |
| src/evaluatorq/simulation/reports/export\_html.py                        |      313 |        7 |     98% |137, 275, 311, 366, 506, 601, 638 |
| src/evaluatorq/simulation/reports/export\_md.py                          |      260 |       15 |     94% |103, 207, 232-239, 315, 379-380, 385, 449 |
| src/evaluatorq/simulation/reports/recommendations.py                     |       93 |        1 |     99% |       286 |
| src/evaluatorq/simulation/reports/sections.py                            |      275 |        7 |     97% |101, 473-478 |
| src/evaluatorq/simulation/reports/token\_usage.py                        |       37 |        5 |     86% |     66-79 |
| src/evaluatorq/simulation/runner/\_\_init\_\_.py                         |        3 |        0 |    100% |           |
| src/evaluatorq/simulation/runner/simulation.py                           |      450 |       20 |     96% |243, 635, 637, 648, 874, 997, 1148-1156, 1174-1175, 1177-1178, 1322, 1344 |
| src/evaluatorq/simulation/token\_usage.py                                |        7 |        0 |    100% |           |
| src/evaluatorq/simulation/traces.py                                      |      316 |       14 |     96% |334, 381, 390, 456, 459, 478, 576, 676, 753, 826, 836, 861, 895, 899 |
| src/evaluatorq/simulation/tracing.py                                     |       78 |       12 |     85% |95-101, 145-151 |
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
| src/evaluatorq/simulation/wrap\_agent.py                                 |       29 |        3 |     90% |77, 79, 100 |
| src/evaluatorq/table\_display.py                                         |      148 |       89 |     40% |31, 36-76, 100, 106, 147, 158, 175-209, 214-232, 242-280 |
| src/evaluatorq/tracing/\_\_init\_\_.py                                   |        4 |        0 |    100% |           |
| src/evaluatorq/tracing/context.py                                        |       34 |        2 |     94% |     50-51 |
| src/evaluatorq/tracing/setup.py                                          |      136 |       39 |     71% |103-113, 134-137, 142, 171, 177-183, 191-201, 235-243, 292-293, 314 |
| src/evaluatorq/tracing/spans.py                                          |       84 |        1 |     99% |       130 |
| src/evaluatorq/types.py                                                  |       86 |        2 |     98% |   32, 274 |
| **TOTAL**                                                                | **27329** | **4198** | **85%** |           |


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