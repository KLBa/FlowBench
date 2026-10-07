# Changelog

## 2026-10-07 commits from 3b4863f to HEAD

* Add GitHub action test
* PCFA: Add flag `optim_strong_wolfe` to avoid loss spikes.

## 2026-05-27 commits from 818794e to 3b4863f

* Add attack_combis
* Add and rename logged metrics.
  * Added metrics: Basically compare all available flow fields against all available flow fields and don't select pairs to compare.
  * Rename metrics: The standard names are confusing. I hope my mechanic naming scheme serves better...
    cf. table at the end of the README for overview of metric names.

## 2026-05-26 commits from 5413267 to 818794e

* Restructure attacks to be callable classes.

  * Every attack must inherit BaseAttack which in turn inherits  LightningModule (so we can use the familiar `forward`.)

  * New attacks can be registered with the `@register_attack` annotation.

  * CLI change: common corruptions have been renamed:
    * `3dcc` -> `cc3d` (because the number at the start is not allowed in Python identifiers.)
    * `common_corruptions` -> `cc2d` (for consistency with `cc3d`)

  Progress: ([code created & code ran] [code checked against previous version for single-sample mini run])

  * [x] [-] Set up framework to work with attack objects
  * [x] [x] fgsm
  * [x] [x] bim
  * [x] [x] pgd
  * [x] [x] cospgd
  * [-] [ ] ffgsm (doesn't exist)
  * [x] [x] apgd
  * [-] [ ] fab (Didn't even work before? Computing grads doesn't seem to happen.)
  * [x] [x] pcfa
  * [x] [x] weather (only works for the dataset `sintel-trainval` after downloading additional data)
  * [x] [x] 3dcc -> cc3d
  * [x] [x] common_corruptions -> cc2d
  * [x] [x] none (must be explicitely passed now!)

## 2026-05-05 commits from ef0e551 to 5413267

* Update FlowBench to use ptlflow-0.4.2
* FlowBench now uses the current version of ptlflow as independent module.
  (That is, you can `pip install ptlflow` and it should still work.)
