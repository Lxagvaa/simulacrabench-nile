# Results log

Sandbox = invented data from make_sandbox.py: checks that the pipeline runs, not real performance.
Dev leaderboard = noised (Laplace, eps=10) and rounded to 0.01.

| Date | Version | Where | UNICEF | World Bank | UNHCR | Mean | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-01 | baseline marginal | sandbox P1 | 0.162 | 0.239 | 0.429 | 0.277 | official baseline |
| 2026-10-01 | v0.1 | sandbox P1, 1 CPU | 0.501 | 0.449 | 0.560 | 0.503 | 1-CPU box: MLP skipped on WB/UNHCR by budget guard |
| | v0.1 | Codabench dev | | | | | not submitted; would fail if real data has off-schema values, numeric options or NaN in GIVEN |
| 2026-10-02 | v0.2 | sandbox P1, 18-core Mac, no torch | 0.5007 | 0.4517 | 0.5615 | 0.5046 | robustness + LatentClass + GBDT + calibration; sandbox can't reward the model changes |
| 2026-10-02 | v0.2 | official score.py P1 | 0.50 | 0.45 | 0.56 | 0.503 | PASS, 56 / 97 / 176 s |
| 2026-10-02 | v0.2 | Codabench dev (sub 956709) | 0.52 | 0.49 | 0.76 | 0.59 | first real score; top teams 0.65 (0.52 / 0.50 / 0.93) |
| 2026-10-02 | v0.2 | sandbox unhcr_quirk | | | 0.411 | | quirky UNHCR: round-1 writes "0" for skipped in 68 multi-select items + 10% gate flips |
| 2026-10-02 | v0.3 | sandbox unhcr_quirk | | | 0.501 | | gate model conditioned on parent x best GIVEN column (wave) + Direct variants: +0.09 |
| 2026-10-02 | v0.3 | sandbox P1 | 0.5006 | 0.4517 | 0.5617 | 0.5047 | no regression; official score.py PASS 64 / 130 / 232 s |
| 2026-10-03 | v0.3 | Codabench dev (sub 957974) | 0.52 | 0.49 | 0.90 | 0.64 | +0.05 from the UNHCR gate fix |
| 2026-10-03 | v0.4 | sandbox, one process via main.predict (UNHCR quirk first) | 0.5006 | 0.4518 | 0.5015 | | v0.3 same harness: 0.5006 / 0.4517 / 0.5012; 627 s of 780 |
| 2026-10-03 | v0.4 | official score.py P1 | 0.50 | 0.45 | 0.56 | | PASS 83 / 220 / 338 s |
| 2026-10-04 | v0.4 | Codabench dev (sub 959636) | ? | ? | ? | 0.64 | size-weighted budget, pairwise-interaction logit, per-item T, extra MLP seed |
| 2026-10-04 | v0.5 | sandbox, one process (UNHCR quirk first) | 0.5009 | 0.4518 | 0.5012 | | EM init fix (blend >= best member), 2 gate contexts, ordinal logit; learned gate parents tried and switched off (no gain: Direct models already cover it) |
| 2026-10-04 | v0.5 | sandbox unhcr_quirk2 (10 items re-routed) | | | 0.5039 | | v0.4: 0.5041 -> learned parents add nothing |
| 2026-10-05 | v0.5 | Codabench dev | ? | ? | ? | 0.64 | + deadline guard (capped L-BFGS, per-fit deadline). Official P1 UNHCR 1369 s was the Mac sleeping, not the code |
| 2026-10-05 | v0.5 | sandbox unhcr_qshift (rows sorted by wave, roles by position: DEV all wave 2) | | | 0.4186 | | |
| 2026-10-05 | v0.6 | sandbox unhcr_qshift | | | 0.4385 | | +0.020: shift detection (TV on GIVEN) -> per-wave Stratified models + importance-weighted blend/T; latent_k8_by_wave gets 0.55 weight. No shift detected on normal sandboxes (max TV 0.08), UNICEF identical 0.5009 |
| 2026-10-06 | v0.6 | Codabench dev | ? | ? | ? | 0.64 | no visible change: real files probably not sorted by round (or no shift above threshold) | |
| 2026-10-05 | v0.7 | sandbox UNICEF, Mac MPS, network blocked | 0.5009 | | | | TabPFN v2 kept with a 3000 s budget: OOF -0.7904, blend weight 0.06; dropped by the scheduler at the dev budget on MPS (29.8 s/fold for 12 items) |
| 2026-10-08 | v0.7 | Codabench dev (sub 967753) | ? | ? | ? | 0.64 | installed and finished: no build error, not worse. Unknown whether TabPFN ran or was dropped by the scheduler or a runtime error | probe: does tabpfn install/run on the hosted image? zip dist/submission_v0.7.zip |
| 2026-10-08 | v0.8 | sandbox unhcr_top2 (s3_23 exactly two ticks) | | | 0.5599 | | v0.7 same set: 0.5599. Family constraint learns a=2.0, b=0.0 for top-two (OOF +7.4 nats over ~83k cells, ~1e-4/cell): harmless, ~no gain |
| 2026-10-09 | v0.8 | Codabench dev | ? | ? | ? | 0.64 | CANARY PASSED: TabPFN runs on the hosted H100 (a failure would have dropped to ~0.55) | TabPFN CANARY on UNICEF: ~0.55 means TabPFN fails on hosted, >=0.64 means it runs. Canary is off in the source again |
| 2026-10-09 | v0.9 | sandbox unhcr_quirk | | | 0.5015 | | v0.8: 0.5012. Subtree calibration of gate parents (s3_11 x 73 children: T=1.05, OOF +302 nats; s6_5 +98; s6_10 +54) + leverage-first item choice for TabPFN/GBDT |
| | v0.9 | Codabench dev | | | | | canary OFF |
| 2026-10-09 | v0.10 | sandbox unhcr_panel (round 2 = re-interviewed round-1 households, 70% answers repeat) | | | 0.5540 | | v0.9: 0.5534. Profile matching WITH cross-round matching: 0.6258 (+0.072) -- not shipped, awaiting organisers' answer |
| 2026-10-09 | v0.10 | sandbox normal, one process | 0.5009 | 0.4519 | 0.5621 | | no regression; total 723 s of 780 (CPU, no GPU) |
| 2026-10-10 | v0.10 | Codabench dev (sub 973916) | 0.52 | 0.49 | 0.91 | 0.64 | REPLACED v0.3 on the leaderboard (higher unrounded mean): UNHCR 0.90 -> 0.91, first visible gain since v0.3. Rank 49 | safe ProfileMatch (same-round twins only) |
| 2026-10-10 | v0.11 | sandbox wb_ord (41 numeric items regenerated as smooth binned latents) | | 0.3732 | | | v0.10: 0.3731. Ordinal smoothing picked on 19/41 items (OOF +12..23 nats each): ~no test gain. Fixed "Not answered" being treated as the top ordinal level. TabPFN/GBDT item cap: 40 in dev, all items when budget >= 1000 s (final) |
| 2026-10-10 | v0.11 | sandbox normal, one process | 0.5009 | 0.4519 | 0.5621 | | no regression, 728 s |
| 2026-10-10 | v0.12 | sandbox unhcr_panel, DEV budget | | | 0.5858 | | v0.11: 0.5540 (+0.032). Fast parallel GBDT (188 items/fold in ~4 s instead of 82 s for 40: thread oversubscription removed) on ALL items + Direct(GBDT) + deep GBDT. Big-budget plain GBDT alone: 0.5746, best single model, 55% weight. Standard GIVEN-only models, no matching logic |
| 2026-10-10 | v0.12 | sandbox normal, one process | 0.5009 | 0.4519 | 0.5621 | | no regression, faster: 570 s (v0.11: 728 s); unhcr_quirk 0.5015 (same) |
| 2026-10-11 | v0.12 | Codabench dev (sub 976310) | 0.52 | 0.49 | 0.92 | 0.64 | UNHCR +0.01 (0.91 -> 0.92): fast GBDT on all items works on real data. Rank 46 |
| 2026-10-11 | v0.13 | sandbox unhcr_panel, DEV budget | | | 0.5910 | | v0.12: 0.5858 (+0.005). + Direct(gbdt_deep), gbdt_big (63 leaves), tabpfn_all (all items, scheduler-gated) |
| 2026-10-11 | v0.13 | sandbox normal, one process | 0.5009 | 0.4519 | 0.5621 | | no regression, 604 s |
| 2026-10-12 | v0.13 | Codabench dev | ? | ? | ? | 0.64 | no visible change (leaderboard keeps v0.12: 0.52 / 0.49 / 0.92) | |
| 2026-10-12 | v0.14 | single-model holdout (log score, cond.) | | | | | panel: rf -0.6262 best, gbdt_deep -0.6329, gbdt -0.6385, tabpfn1 -0.6468, mlp -0.6544. quirk: tabpfn1 -0.6372 best, mlp -0.6595, gbdt -0.6632, rf -0.6757 |
| 2026-10-12 | v0.14 | blend, budget 3000 s, Mac MPS | | | quirk 0.5020 / panel 0.5980 | | v0.13 same setting: 0.5015 / 0.5932. + multi-output RF + TabPFN n=1 on all items + duplicate-TabPFN fix. Modest |
| 2026-10-12 | -- | pooling test (OOF, 2-fold over rows) | | | | | log-linear vs linear pool: UNICEF -0.0001, WB +0.0009, quirk +0.0008, panel -0.0014 skill: blend is not the bottleneck, idea dropped |
| 2026-10-12 | v0.14 | sandbox normal, one process | 0.5009 | 0.4518 | 0.5621 | | no regression, 593 s |
| | v0.14 | Codabench dev | | | | | |
| 2026-10-12 | v0.15 | single-model holdout wb_ord | | | | | mlp -1.4589, factor_d5 -1.4767, factor_d3 -1.4902, latent16 -1.4939, latent8 -1.4967 |
| 2026-10-12 | v0.15 | sandbox normal, one process, torch env | 0.5010 | 0.4553 | 0.5633 | | WB +0.0036 (first WB sandbox move since v0.1), UNHCR +0.0012. FactorVAE d=5/d=3 (continuous latent traits, GIVEN-dependent prior); 37% blend weight on UNICEF. Redundant TabPFN removed + skip slower TabPFN siblings once one is dropped: 731 s (was 774) |
| 2026-10-13 | v0.15 | Codabench dev | ? | ? | ? | 0.64 | finished (no memory/time error on hosted with RF, all-item TabPFN, factor models); did not beat v0.12 | |

## UNHCR gap in absolute terms (2026-10-09)

0.90 -> 0.93 is 8.1 nats per respondent (U = 1.262 nats/cell, 213 items). That is too large for better
modelling of the same GIVEN inputs; the leaders likely use information we don't: near-identical
profiles (UNHCR is a panel with the household link removed) and/or respondent_id/row order. Asked the
organisers before using either across rounds.

## Why UNHCR is 0.76 vs leaders' 0.93

The UNHCR schema description (starter_kit/data/unhcr.json, dataset.description) says gates are NOT
satisfied by the delivered rows: gate/data agreement median 0.83 (0.29-0.98), the three s3_11
branches overlap, and in five multi-select families round 1 wrote "0" for skipped respondents while
round 2 wrote the sentinel. v0.2's gate model only looked at the parent, so it could not tell the
rounds apart. v0.3 conditions P(asked) on the most informative GIVEN column too (wave).

## Organiser clarification 2026-10-07 (starter kit f980ef5, FAQ 12)

Rows were shuffled under a fixed seed BEFORE roles and respondent_id were assigned: row position and
respondent_id carry no information; DEV/TEST are random samples (UNICEF stratified by country, UNHCR
balanced on round and country of asylum). Random CV on TRAIN is appropriate. So: the v0.6 file-order idea
was wrong (harmless, never triggers), and ID features are pointless. The UNHCR gap must come from GIVEN,
most plausibly near-identical profiles (same household in both rounds), which flexible models pick up.

## Final-phase simulation 2026-10-12 (v0.13, Mac MPS + 18 CPU cores, network blocked)

notes/runs/final_sim_v13.log. Phase 2 detected on all three (TRAIN+DEV visible, TEST scored, 5 folds).
Total 2769 s of 3300 (hard 3600); peak RSS 7.6 GB of 16 GB. UNHCR 1417 s (TabPFN dropped: 288 s/fold on
MPS), World Bank 826 s, UNICEF 527 s (TabPFN kept on both). Wasted ~10 min on a duplicate TabPFN
(tabpfn_all identical to tabpfn when the cap covers all items) -> fixed: skipped when cap >= #items.
Hosted: 8 CPU cores (GBDT/numpy slower than here) but an H100 (TabPFN/nets faster). Margin ~800 s.

## Final-phase memory/robustness fixes 2026-10-12 (v0.15 rebuilt)

- Peak memory of the final-phase run was 20-22 GB (limit 16): a 300-tree multi-output RF stores a
  distribution for every item in every leaf (12.4 GB alone). Now 100 trees, leaf >= 5, half rows each:
  4.1 GB alone and BETTER holdout (-0.6383 vs -0.6477). Full final sim: 5.2 GB peak.
- TabPFN kept one fitted network per item; now one shared classifier refitted per item (1.5 GB for all
  189 UNHCR items).
- A refit cut short by the deadline returned barely trained models (UNICEF 0.49 -> 0.34 after a Mac
  sleep). Now every fold model also predicts the scored rows; their average replaces the refit when it
  would not fit / overran. Tested: forced fallback 0.5006 vs refit 0.5009.
- v0.14 zip renamed submission_v14_UNSAFE_memory.zip (300-tree RF + per-item TabPFN).

## Cross-question stacking 2026-10-13 (tried, removed)

Per-item logistic correction on [GIVEN, 20 PCs of the blend's predictions for all items], offset = blend,
accepted per item only on held-out-half gain. panel: 2/213 items, 0.5919 -> 0.5919; quirk: 14/213 items
(+192 nats held-out on OOF) but test 0.5018 -> 0.5017; World Bank: 0/76. The OOF gain does not carry over
to the scored rows. Removed from the pipeline (module simbench/stacking.py kept, unused). v0.16 = v0.15.

