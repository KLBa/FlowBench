# Changelog

## 2026-05-26 commits from 5413267 to 818794e

* Restructure attacks to be callable classes.

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
  * [x] [x] weather (only works for sintel-train)
  * [x] [x] 3dcc -> cc3d
  * [x] [x] common_corruptions -> cc2d
  * [x] [ ] none

## 2026-05-05 commits from ef0e551 to 5413267

* Update FlowBench to use ptlflow-0.4.2
* FlowBench now uses the current version of ptlflow as independent module.
  (That is, you can `pip install ptlflow` and it should still work.)
