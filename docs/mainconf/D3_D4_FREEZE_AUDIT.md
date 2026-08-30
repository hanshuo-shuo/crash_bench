# D3 source selection and D4 protocol/split freeze audit

Status: `D4_SPLIT_FROZEN__TEST_NOT_AUTHORIZED`

Quest array `5205759_[0-5]` completed all 126 predeclared fresh reset
candidates at commit `acb3d2d28dc29c4bc502b1db2966df118b6be2f8`. Task 0 had
57/63 nominal successes and task 2 had 61/63. After millimeter-scale scene
deduplication, the first 50 successful candidates per task were selected, giving
100 unique physical sources.

The D4 composite protocol SHA-256 is
`e2a5264a802ad58fc955a3bc898821f3880c58023ccbd47bef0bfe15012c5997`.
It pins the benchmark, source-sampling, split, utility, deployable-option,
diagnostic-option, and mechanism-screen configs.

Deterministic physical-source splitting produced, per task: 12 train, 6
calibration, 6 development, 16 future statewise test, and 10 independent future
sequential sources. Test identities are frozen, but no test authorization exists
and no test outcome has been read. Formal nominal candidates enter future test
roles only through their exact predeclared attempt IDs; historical and D2 screen
sources remain blocked.
