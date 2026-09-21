# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/orq-ai/evaluatorq/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                     |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/evaluatorq/\_\_init\_\_.py                                           |       26 |        2 |     92% |     30-31 |
| src/evaluatorq/\_\_main\_\_.py                                           |        3 |        3 |      0% |       1-4 |
| src/evaluatorq/cli.py                                                    |       50 |        2 |     96% |  121, 184 |
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
| src/evaluatorq/common/env\_config.py                                     |       59 |        0 |    100% |           |
| src/evaluatorq/common/extract\_json.py                                   |       79 |       10 |     87% |32, 40, 67-68, 121, 140-141, 149-152 |
| src/evaluatorq/common/fields.py                                          |        6 |        0 |    100% |           |
| src/evaluatorq/common/hook\_compose.py                                   |       16 |        0 |    100% |           |
| src/evaluatorq/common/judge.py                                           |      315 |        8 |     97% |315-316, 410, 424, 598, 602, 645, 986 |
| src/evaluatorq/common/jury.py                                            |      310 |        7 |     98% |142, 155-156, 239, 497, 833, 894 |
| src/evaluatorq/common/llm\_call.py                                       |      194 |        3 |     98% |406, 418, 512 |
| src/evaluatorq/common/llm\_client.py                                     |       48 |        0 |    100% |           |
| src/evaluatorq/common/llm\_limit.py                                      |       31 |        0 |    100% |           |
| src/evaluatorq/common/messages.py                                        |       16 |        0 |    100% |           |
| src/evaluatorq/common/model\_catalogue.py                                |      195 |        3 |     98% |252, 255, 364 |
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
| src/evaluatorq/common/structured\_output.py                              |      290 |        9 |     97% |164, 264, 321, 343, 528-529, 643, 706, 776 |
| src/evaluatorq/common/target\_call.py                                    |      117 |        0 |    100% |           |
| src/evaluatorq/common/template\_engine.py                                |       77 |        2 |     97% |    60, 64 |
| src/evaluatorq/common/thread\_context.py                                 |       62 |        1 |     98% |        75 |
| src/evaluatorq/common/tracing.py                                         |      317 |       27 |     91% |166-177, 215-217, 224, 427, 493-495, 574-576, 613-614, 643-644 |
| src/evaluatorq/contracts.py                                              |      560 |       18 |     97% |58, 81, 134-136, 332, 904, 1001, 1019, 1023, 1025, 1075, 1080, 1084, 1089, 1091, 1210, 1466, 1788 |
| src/evaluatorq/dashboard/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| src/evaluatorq/dashboard/\_compat.py                                     |       23 |       14 |     39% | 39-53, 71 |
| src/evaluatorq/dashboard/app.py                                          |      277 |       39 |     86% |84-85, 88-89, 259-260, 280-281, 318, 321, 324, 327-329, 345, 349-351, 378, 407, 411-413, 444, 447, 455-457, 483, 494, 502-504, 518, 549, 552, 560-562 |
| src/evaluatorq/dashboard/apply\_ui.py                                    |      293 |       29 |     90% |78, 154-159, 474-484, 554-555, 561-563, 595-600, 604, 643-644, 652, 675, 682, 706, 711, 765, 808 |
| src/evaluatorq/dashboard/filter\_request.py                              |       13 |        1 |     92% |        42 |
| src/evaluatorq/dashboard/filters.py                                      |      168 |       13 |     92% |132, 142-143, 157, 163, 181, 188, 241-242, 296, 302, 344, 411 |
| src/evaluatorq/dashboard/launch.py                                       |       48 |       10 |     79% |53-54, 81-89, 124 |
| src/evaluatorq/dashboard/library.py                                      |      184 |       13 |     93% |138-142, 162-163, 173, 209, 259-260, 296-297, 299 |
| src/evaluatorq/dashboard/metrics.py                                      |      733 |       53 |     93% |132, 135-136, 152-153, 159-160, 391-393, 410, 418-419, 422-423, 491-492, 495-496, 546-547, 550-551, 609-610, 847, 851, 876, 881, 889-890, 985-986, 1003-1004, 1007-1008, 1075, 1205-1212, 1245-1246, 1311, 1317-1318, 1320-1321 |
| src/evaluatorq/dashboard/orq\_links.py                                   |       40 |        0 |    100% |           |
| src/evaluatorq/dashboard/orq\_workspace.py                               |       15 |        0 |    100% |           |
| src/evaluatorq/dashboard/redteam\_charts.py                              |      189 |       17 |     91% |79-81, 93, 98-101, 111, 114, 173, 175, 363, 376, 422, 436-437 |
| src/evaluatorq/dashboard/redteam\_transcripts.py                         |      116 |        6 |     95% |74, 143, 145-146, 181, 279 |
| src/evaluatorq/dashboard/redteam\_views.py                               |       88 |        8 |     91% |56-57, 60, 63-65, 151-152 |
| src/evaluatorq/dashboard/report\_kit.py                                  |      230 |       10 |     96% |121, 125, 245, 279, 317, 434-435, 495-496, 540 |
| src/evaluatorq/dashboard/report\_tabs.py                                 |     1062 |       79 |     93% |61, 298, 324, 458, 501, 926-927, 931, 947, 1119-1138, 1166, 1200, 1219, 1222-1227, 1254, 1356, 1389, 1409-1439, 1444-1466, 1486, 1494, 1576, 1578, 1844, 2114, 2116, 2309, 2315, 2336, 2374-2390, 2448 |
| src/evaluatorq/dashboard/shell.py                                        |       52 |        4 |     92% |60-61, 78-79 |
| src/evaluatorq/dashboard/sim\_compare.py                                 |      270 |        8 |     97% |156, 308, 342, 551, 554, 620, 626, 633 |
| src/evaluatorq/dashboard/sim\_views.py                                   |      226 |       10 |     96% |64, 67-69, 93, 252, 254, 256, 606-607 |
| src/evaluatorq/dashboard/styles.py                                       |       12 |        0 |    100% |           |
| src/evaluatorq/dashboard/surfaces.py                                     |       66 |        0 |    100% |           |
| src/evaluatorq/dashboard/theme.py                                        |        2 |        0 |    100% |           |
| src/evaluatorq/dashboard/trace\_links.py                                 |       38 |        0 |    100% |           |
| src/evaluatorq/dashboard/view.py                                         |      439 |       33 |     92% |104, 148-149, 331, 542, 549, 624, 681, 878, 919-920, 950-951, 1299-1339 |
| src/evaluatorq/deployment.py                                             |       72 |        9 |     88% |79-81, 134, 170, 188, 194, 252-260 |
| src/evaluatorq/evaluatorq.py                                             |      200 |        5 |     98% |80, 90, 234, 335, 540 |
| src/evaluatorq/evaluators.py                                             |       37 |       18 |     51% |54, 75, 113-148 |
| src/evaluatorq/fetch\_data.py                                            |      139 |       17 |     88% |48, 82-84, 122, 183-184, 188-191, 251, 259, 293, 314, 323, 326-327 |
| src/evaluatorq/integrations/\_\_init\_\_.py                              |        6 |        3 |     50% |     39-41 |
| src/evaluatorq/integrations/callable\_integration/\_\_init\_\_.py        |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/callable\_integration/target.py              |       55 |        0 |    100% |           |
| src/evaluatorq/integrations/crewai\_integration/\_\_init\_\_.py          |        2 |        0 |    100% |           |
| src/evaluatorq/integrations/crewai\_integration/target.py                |       66 |        8 |     88% |   135-146 |
| src/evaluatorq/integrations/langchain\_integration/\_\_init\_\_.py       |        4 |        0 |    100% |           |
| src/evaluatorq/integrations/langchain\_integration/convert.py            |      172 |       46 |     73% |111-117, 129, 174-185, 195-204, 280, 282-283, 288, 302, 304, 351-353, 367-371, 381, 399-409, 425, 436, 457, 460-462 |
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
| src/evaluatorq/jury\_presets.py                                          |       91 |        9 |     90% |88, 124-130, 195 |
| src/evaluatorq/llm\_jury.py                                              |      355 |        4 |     99% |403, 610, 1324, 1454 |
| src/evaluatorq/openresponses/\_\_init\_\_.py                             |       10 |        1 |     90% |        81 |
| src/evaluatorq/openresponses/client.py                                   |        9 |        0 |    100% |           |
| src/evaluatorq/openresponses/convert\_models.py                          |      121 |        3 |     98% |   168-175 |
| src/evaluatorq/openresponses/dataset.py                                  |      159 |       15 |     91% |61, 67, 71, 78, 98, 125, 153, 161, 167, 176-178, 213, 223, 228 |
| src/evaluatorq/openresponses/input\_items.py                             |       57 |        1 |     98% |        94 |
| src/evaluatorq/openresponses/otel\_messages.py                           |      117 |       25 |     79% |56-57, 63-66, 83, 115, 120-121, 136-137, 156, 161-162, 172-183 |
| src/evaluatorq/openresponses/target.py                                   |      143 |        1 |     99% |       341 |
| src/evaluatorq/openresponses/tracing.py                                  |       56 |        7 |     88% |76, 78, 128-143 |
| src/evaluatorq/openresponses/types.py                                    |       91 |        0 |    100% |           |
| src/evaluatorq/pairwise.py                                               |      327 |        5 |     98% |459, 462-463, 746, 1022 |
| src/evaluatorq/pairwise\_reports/\_\_init\_\_.py                         |        3 |        0 |    100% |           |
| src/evaluatorq/pairwise\_reports/export\_html.py                         |      165 |        4 |     98% |     94-99 |
| src/evaluatorq/pairwise\_reports/sections.py                             |       96 |        2 |     98% |   213-214 |
| src/evaluatorq/pairwise\_run.py                                          |       85 |        4 |     95% |145-149, 152 |
| src/evaluatorq/processings.py                                            |       95 |        2 |     98% |  245, 346 |
| src/evaluatorq/progress.py                                               |      113 |       47 |     58% |57-97, 101-106, 139, 143-150, 154-159, 168-182, 262-265 |
| src/evaluatorq/ranking.py                                                |      201 |        0 |    100% |           |
| src/evaluatorq/redteam/\_\_init\_\_.py                                   |       35 |        5 |     86% |   255-264 |
| src/evaluatorq/redteam/adaptive/\_\_init\_\_.py                          |        7 |        0 |    100% |           |
| src/evaluatorq/redteam/adaptive/agent\_context.py                        |       12 |       12 |      0% |      3-42 |
| src/evaluatorq/redteam/adaptive/attack\_generator.py                     |       54 |        2 |     96% |  220, 235 |
| src/evaluatorq/redteam/adaptive/blackbox\_classifier.py                  |      131 |       11 |     92% |263-268, 324-329, 389 |
| src/evaluatorq/redteam/adaptive/capability\_classifier.py                |      107 |        4 |     96% |254, 257, 326, 338 |
| src/evaluatorq/redteam/adaptive/evaluator.py                             |       87 |        1 |     99% |       119 |
| src/evaluatorq/redteam/adaptive/objective\_generator.py                  |      142 |       21 |     85% |55, 176, 315-316, 433-447, 490-494, 509, 559-560, 562-563, 666-700 |
| src/evaluatorq/redteam/adaptive/orchestrator.py                          |      449 |       72 |     84% |101-128, 134-141, 150-163, 169-170, 176-190, 195-199, 204, 315-316, 354, 473-474, 539-545, 696-707, 1004, 1089, 1304 |
| src/evaluatorq/redteam/adaptive/pipeline.py                              |      210 |       29 |     86% |84, 235-239, 268, 272-311, 390, 627-628, 672-673, 758, 766-770 |
| src/evaluatorq/redteam/adaptive/strategy\_planner.py                     |       99 |        3 |     97% |   177-181 |
| src/evaluatorq/redteam/adaptive/strategy\_registry.py                    |      101 |        2 |     98% |  259, 263 |
| src/evaluatorq/redteam/adaptive/tool\_chaining.py                        |       80 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/\_errors.py                              |       23 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/\_retry.py                               |       12 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/base.py                                  |       78 |        5 |     94% |37-40, 140 |
| src/evaluatorq/redteam/backends/openai.py                                |      107 |        4 |     96% |173, 192, 257, 305 |
| src/evaluatorq/redteam/backends/openresponses.py                         |       44 |        0 |    100% |           |
| src/evaluatorq/redteam/backends/orq.py                                   |      305 |       42 |     86% |27-28, 115-130, 325-326, 492, 497, 572-582, 586-598, 628-629, 639, 643, 679, 702, 714-725 |
| src/evaluatorq/redteam/backends/registry.py                              |       53 |        5 |     91% |61-65, 89, 142-143 |
| src/evaluatorq/redteam/cli.py                                            |      351 |       56 |     84% |80, 86, 100-101, 167-169, 181, 195-197, 209, 236, 297, 632, 697-698, 700-701, 708-712, 715-717, 720-722, 725-727, 779, 783, 887, 890-891, 939-943, 956-957, 1005-1024 |
| src/evaluatorq/redteam/contracts.py                                      |      824 |       44 |     95% |96, 136-138, 306, 485, 498-499, 512, 538, 690, 694, 714, 1316, 1358-1378, 1384, 1478-1480, 1489, 1575-1581, 1658-1660, 1986, 2185-2194 |
| src/evaluatorq/redteam/delivery\_method\_registry.py                     |       59 |        1 |     98% |       106 |
| src/evaluatorq/redteam/exceptions.py                                     |        5 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/\_\_init\_\_.py                        |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/owasp/\_\_init\_\_.py                  |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/owasp/agent\_evaluators.py             |       44 |        1 |     98% |       963 |
| src/evaluatorq/redteam/frameworks/owasp/evaluatorq\_bridge.py            |      224 |       56 |     75% |86, 115, 131-132, 141-151, 156, 169, 175, 230, 235-237, 270-271, 430-431, 451-455, 460-464, 469-507 |
| src/evaluatorq/redteam/frameworks/owasp/evaluators.py                    |       67 |       25 |     63% |138-151, 169, 188, 198-211, 220 |
| src/evaluatorq/redteam/frameworks/owasp/llm\_evaluators.py               |       36 |       19 |     47% |178-374, 548-655, 677-796, 818-936, 957-1075, 1096-1213, 1233 |
| src/evaluatorq/redteam/frameworks/owasp/models.py                        |       41 |        3 |     93% | 11-13, 73 |
| src/evaluatorq/redteam/frameworks/owasp\_asi.py                          |        8 |        0 |    100% |           |
| src/evaluatorq/redteam/frameworks/owasp\_llm.py                          |        8 |        0 |    100% |           |
| src/evaluatorq/redteam/hooks.py                                          |      381 |       95 |     75% |313, 354, 374-377, 450-454, 458-470, 505, 514-515, 527, 543, 666, 736, 754-837, 841-843, 856-857, 865-868, 870-873, 878-879 |
| src/evaluatorq/redteam/judge.py                                          |        3 |        0 |    100% |           |
| src/evaluatorq/redteam/replay.py                                         |       59 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/\_\_init\_\_.py                           |        5 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/\_utils.py                                |       15 |        2 |     87% |     27-28 |
| src/evaluatorq/redteam/reports/apply.py                                  |       11 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/converters.py                             |      606 |       31 |     95% |157, 203, 219-225, 254-260, 301-304, 342, 344-347, 350, 357, 363-364, 374, 376, 1063-1065 |
| src/evaluatorq/redteam/reports/display.py                                |      146 |       17 |     88% |32, 40, 73-74, 81-88, 93-101 |
| src/evaluatorq/redteam/reports/executive\_summary.py                     |       36 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/export\_html.py                           |      725 |      107 |     85% |145, 171, 197, 297-300, 339-340, 379, 385-396, 475, 498, 520, 524, 547, 594, 607, 618, 626-629, 649, 685, 687, 723, 746, 764-768, 770, 772, 777-783, 785-791, 793-799, 801-807, 824-852, 862-889, 897, 922, 941, 947, 967, 1010, 1083, 1112, 1142, 1179, 1219-1221, 1284, 1327, 1357, 1388, 1391, 1394, 1408, 1411, 1413-1414, 1429-1430, 1497, 1499, 1544, 1549 |
| src/evaluatorq/redteam/reports/export\_md.py                             |      432 |      150 |     65% |44-46, 54, 57, 103, 140, 144-147, 150-151, 171, 186, 188, 190, 192, 194, 196, 198, 232, 236-244, 251-290, 295-338, 396, 412, 415, 453, 474-499, 506, 527-575, 580-598, 603-610, 620, 639, 691, 712, 714, 719, 741, 743, 745-746, 758-760, 848 |
| src/evaluatorq/redteam/reports/guidance.py                               |        2 |        0 |    100% |           |
| src/evaluatorq/redteam/reports/recommendations.py                        |      184 |        7 |     96% |76, 221, 223, 251, 302, 305, 417 |
| src/evaluatorq/redteam/reports/sections.py                               |      351 |       16 |     95% |70, 85, 99-100, 102-103, 440-445, 582, 732, 745, 827, 989 |
| src/evaluatorq/redteam/runner.py                                         |     1468 |      109 |     93% |185-187, 224, 312, 386, 393, 693-698, 700-701, 753-754, 784, 806, 898-899, 973, 1055-1056, 1105-1109, 1139-1143, 1185, 1255-1257, 1507, 1699, 1706-1708, 1817-1822, 1964-1969, 1977, 2022-2023, 2028-2035, 2083, 2085-2091, 2116-2126, 2220, 2263, 2432-2433, 2435-2437, 2542, 2605-2606, 3063-3064, 3144, 3259-3260, 3427-3430, 3497-3523, 3624, 3711, 3814, 3875-3879, 3883-3884, 3893, 3896-3897 |
| src/evaluatorq/redteam/runtime/\_\_init\_\_.py                           |        0 |        0 |    100% |           |
| src/evaluatorq/redteam/runtime/jobs.py                                   |       99 |        4 |     96% |88, 154, 205-206 |
| src/evaluatorq/redteam/tracing.py                                        |       51 |        6 |     88% |103-105, 142-144 |
| src/evaluatorq/redteam/utils.py                                          |        9 |        1 |     89% |        27 |
| src/evaluatorq/redteam/vulnerability\_registry.py                        |       73 |        4 |     95% |186, 300, 313, 327 |
| src/evaluatorq/send\_results.py                                          |       59 |        0 |    100% |           |
| src/evaluatorq/simulation/\_\_init\_\_.py                                |       23 |        1 |     96% |       325 |
| src/evaluatorq/simulation/\_config.py                                    |       88 |        0 |    100% |           |
| src/evaluatorq/simulation/\_datapoint\_io.py                             |       51 |        2 |     96% |    68, 92 |
| src/evaluatorq/simulation/\_usage.py                                     |       11 |        0 |    100% |           |
| src/evaluatorq/simulation/adapters.py                                    |       30 |        7 |     77% | 25, 75-81 |
| src/evaluatorq/simulation/agents/\_\_init\_\_.py                         |        4 |        0 |    100% |           |
| src/evaluatorq/simulation/agents/base.py                                 |      198 |       12 |     94% |171, 176, 295-297, 301-302, 520, 541, 613, 626, 704 |
| src/evaluatorq/simulation/agents/judge.py                                |      297 |       14 |     95% |208, 229-235, 247, 260, 266, 274, 441-446, 465-471, 856-861 |
| src/evaluatorq/simulation/agents/user\_simulator.py                      |       36 |       10 |     72% |83, 93-100, 108-114 |
| src/evaluatorq/simulation/api.py                                         |      716 |       54 |     92% |607, 870, 1165, 1225, 1298, 1362, 1428, 1456, 1559-1561, 1675, 1692, 1720, 1736-1737, 1807, 1810, 1898, 1924, 1976-1977, 1987, 1991, 1996, 2061, 2068-2085, 2161, 2238-2244, 2247-2250, 2518-2521, 2528-2531, 2707 |
| src/evaluatorq/simulation/cli.py                                         |      651 |      103 |     84% |97-104, 113-114, 143, 147, 149, 158, 162, 167, 170-172, 176, 186, 213, 223, 239-240, 251-271, 276, 530, 567, 569, 827-828, 830-831, 852, 890, 893, 1135, 1186-1187, 1189-1190, 1202, 1419-1420, 1422-1423, 1425, 1574, 1594-1596, 1603, 1627-1638, 1641-1642, 1752-1753, 1769, 1800-1801, 1845, 1893, 1904-1905, 1907, 1961, 2010-2011, 2024-2028, 2097-2102, 2136-2137, 2215 |
| src/evaluatorq/simulation/convert.py                                     |       52 |        0 |    100% |           |
| src/evaluatorq/simulation/evaluators/\_\_init\_\_.py                     |        3 |        0 |    100% |           |
| src/evaluatorq/simulation/evaluators/scorers.py                          |      143 |        3 |     98% |149, 155, 193 |
| src/evaluatorq/simulation/exceptions.py                                  |        8 |        0 |    100% |           |
| src/evaluatorq/simulation/experiments.py                                 |       45 |        0 |    100% |           |
| src/evaluatorq/simulation/generators/\_\_init\_\_.py                     |        5 |        0 |    100% |           |
| src/evaluatorq/simulation/generators/datapoint\_generator.py             |       94 |       44 |     53% |68-69, 92-160, 200-220 |
| src/evaluatorq/simulation/generators/first\_message\_generator.py        |       64 |        2 |     97% |   124-125 |
| src/evaluatorq/simulation/generators/persona\_generator.py               |      128 |       52 |     59% |124, 128-152, 358, 374-377, 389-419 |
| src/evaluatorq/simulation/generators/scenario\_generator.py              |      194 |       60 |     69% |171-179, 183-189, 225, 304-305, 315-316, 384-385, 392, 401-403, 462-465, 474-476, 531-534, 543-545, 558-559, 563-564, 613-616, 625-627, 639-642, 654-663 |
| src/evaluatorq/simulation/hooks.py                                       |      225 |        8 |     96% |264-266, 389, 397, 429, 433, 466, 576 |
| src/evaluatorq/simulation/metrics.py                                     |        5 |        0 |    100% |           |
| src/evaluatorq/simulation/quality/\_\_init\_\_.py                        |        2 |        0 |    100% |           |
| src/evaluatorq/simulation/quality/message\_perturbation.py               |       66 |       42 |     36% |78-83, 87-94, 98-104, 108-111, 115-121, 140-143, 152-153, 165-172 |
| src/evaluatorq/simulation/replay.py                                      |       27 |        2 |     93% |     70-71 |
| src/evaluatorq/simulation/reports/\_\_init\_\_.py                        |        6 |        0 |    100% |           |
| src/evaluatorq/simulation/reports/apply.py                               |       10 |        0 |    100% |           |
| src/evaluatorq/simulation/reports/display.py                             |       79 |        3 |     96% |156-157, 168 |
| src/evaluatorq/simulation/reports/executive\_summary.py                  |       40 |        0 |    100% |           |
| src/evaluatorq/simulation/reports/export\_html.py                        |      318 |        7 |     98% |137, 275, 311, 366, 506, 609, 646 |
| src/evaluatorq/simulation/reports/export\_md.py                          |      265 |       15 |     94% |103, 207, 246-253, 329, 393-394, 399, 463 |
| src/evaluatorq/simulation/reports/recommendations.py                     |       93 |        1 |     99% |       296 |
| src/evaluatorq/simulation/reports/sections.py                            |      287 |        7 |     98% |109, 500-505 |
| src/evaluatorq/simulation/reports/token\_usage.py                        |       37 |        5 |     86% |     66-79 |
| src/evaluatorq/simulation/runner/\_\_init\_\_.py                         |        3 |        0 |    100% |           |
| src/evaluatorq/simulation/runner/simulation.py                           |      466 |       21 |     95% |247, 640, 642, 653, 661, 921, 1041, 1192-1200, 1218-1219, 1221-1222, 1366, 1388 |
| src/evaluatorq/simulation/token\_usage.py                                |        7 |        0 |    100% |           |
| src/evaluatorq/simulation/traces.py                                      |      324 |       13 |     96% |342, 389, 398, 464, 467, 486, 584, 694, 771, 853, 863, 922, 926 |
| src/evaluatorq/simulation/tracing.py                                     |       45 |        6 |     87% |     84-90 |
| src/evaluatorq/simulation/types.py                                       |      273 |        3 |     99% |476, 564-565 |
| src/evaluatorq/simulation/utils/\_\_init\_\_.py                          |        5 |        0 |    100% |           |
| src/evaluatorq/simulation/utils/dataset\_export.py                       |       75 |       20 |     73% |54-56, 61-64, 88-90, 109-115, 130-136, 179-181, 194, 196 |
| src/evaluatorq/simulation/utils/extract\_json.py                         |        3 |        0 |    100% |           |
| src/evaluatorq/simulation/utils/prompt\_builders.py                      |       65 |        6 |     91% |45, 52, 59, 64, 80, 124 |
| src/evaluatorq/simulation/utils/run\_store.py                            |      112 |        6 |     95% |72-74, 80-82 |
| src/evaluatorq/simulation/utils/structured\_output.py                    |        3 |        0 |    100% |           |
| src/evaluatorq/simulation/wrap\_agent.py                                 |       29 |        3 |     90% |84, 86, 109 |
| src/evaluatorq/table\_display.py                                         |      154 |       65 |     58% |57, 70, 78, 90, 115, 117, 147, 153, 185-219, 224-242, 252-290 |
| src/evaluatorq/tracing/\_\_init\_\_.py                                   |        4 |        0 |    100% |           |
| src/evaluatorq/tracing/context.py                                        |       34 |        2 |     94% |     50-51 |
| src/evaluatorq/tracing/setup.py                                          |      140 |       12 |     91% |97, 184-187, 253, 255-258, 323-324, 345 |
| src/evaluatorq/tracing/spans.py                                          |       84 |        1 |     99% |       130 |
| src/evaluatorq/types.py                                                  |       88 |        2 |     98% |   32, 292 |
| **TOTAL**                                                                | **27157** | **2431** | **91%** |           |


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