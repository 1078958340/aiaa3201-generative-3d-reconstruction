# Part 2 results

Six official runs use the same i3dgs method and split within each scene. Full uses all training inputs. Sparse uses the fixed temporally sampled training subset. The metric tables use all held-out views, with the same view list for both models.

| Scene | Mode | Input images | Saved keyframes | Gaussian nodes | ATE RMSE ↓ (reference units) | Core reconstruction (s) | Runner wall time (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DL3DV-2 | Full | 267 | 19 | 37,492 | 0.00557 | 13.26 | 78 |
| DL3DV-2 | Sparse | 11 | 8 | 29,368 | 0.00814 | 3.29 | 74 |
| Re10k-1 | Full | 154 | 6 | 23,869 | 0.00578 | 4.96 | 39 |
| Re10k-1 | Sparse | 6 | 3 | 18,215 | 0.00776 | 1.07 | 37 |
| Waymo-405841 | Full | 174 | 63 | 275,060 | 1.48093 | 30.38 | 81 |
| Waymo-405841 | Sparse | 20 | 5 | 78,731 | 0.05724 | 4.44 | 45 |

[Exact summary, including image metrics and run IDs](summary.csv) · [Example renders and main comparison](../../README.md#part-2--immediate-3d-gaussian-splatting-from-rgb)

## Rendering and coverage

- **DL3DV-2.** Sparse loses 3.04 dB of average PSNR. The gap in registered keyframes between frames 92 and 212 coincides with lower quality over those test views. At frame 153, the foreground planter and building outline distort visibly. [Full scores](DL3DV-2/full-per-view.csv) · [Sparse scores](DL3DV-2/sparse-per-view.csv)
- **Re10k-1.** Sparse saves three keyframes and loses 5.43 dB of average PSNR. In the frame 50 view, parts of the nearby floor and side walls have limited coverage. [Full scores](Re10k-1/full-per-view.csv) · [Sparse scores](Re10k-1/sparse-per-view.csv)
- **Waymo-405841.** Sparse retains frames 1, 11, 21, 31, and 41. The 15 later inputs fail registration checks under the upstream thresholds (eight with too few inliers and seven with too few triangulated points). The first six held-out views have mean PSNR of 16.98 dB (Full) and 16.92 dB (Sparse). For the remaining 19 test views the respective means are 16.09 and 6.92 dB. All 15 black Sparse renders count toward the 25-view evaluation. [Full scores](Waymo-405841/full-per-view.csv) · [Sparse scores](Waymo-405841/sparse-per-view.csv)

## Pose and timing

Each run fits a similarity transform from its registered camera centers to the course reference poses. Its ATE is the center RMSE on those registered frames, in the scene's reference units. Full and Sparse have different registered sets. The four common Waymo keyframes are all near the beginning of the route. Waymo Sparse's small fitted ATE describes that short segment, while the held-out views expose the missing route coverage. Re10k-1 Sparse has three near-collinear aligned centers, which weakly constrain the rotation.

Core reconstruction time covers the frame loop including geometry and pose initialization. Runner wall time includes module startup, just-in-time compilation, output writing and model validation after input preparation. Each number comes from one run on a shared RTX A6000.

## Reproduction details

- The runner retains 4 bootstrap camera slots, a 1,000-point bootstrap budget, `--num_iterations 30`, and the upstream frame-registration thresholds. The viewer is disabled during the timed reconstruction. Tests and reference poses are supplied to the separate evaluator.
- MegaLoc feature aggregation casts both einsum inputs to FP32 to handle the mixed types in the selected runtime. The bootstrap adapter masks vacant solver slots when fewer views are selected than configured. Re10k-1 Sparse reached initialization with 3 of 4 slots. Both compatibility adjustments are used in the published runner for all scenes.
- Archived results come from the course GPU runs. The repository version was adapted for portable paths and checkpoint retrieval. Input manifests and archived results were checked locally. The complete GPU setup and six runs have yet to be repeated from a fresh copy of this repository.
