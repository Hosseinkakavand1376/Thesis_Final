# Wavelength Selection for ML & Deep Learning in Hyperspectral Leaf Classification

**MSc Thesis** — Artificial Intelligence Engineering, Politecnico di Torino, July 2026  
**Author:** Hossein Kakavand  
**Supervisors:** Prof. Renato Ferrero · PhD Nicola Dilillo

---

## Abstract

Hyperspectral Imaging (HSI) captures hundreds of contiguous narrow spectral bands, enabling highly detailed analysis of agricultural targets. However, this spectral richness introduces the **Hughes Phenomenon** (curse of dimensionality), and existing literature largely ignores a critical flaw in evaluation: **Spatial Data Leakage**.

This thesis makes three core contributions:

1. **Leakage-Free Checkerboard Evaluation Protocol** — A block-based spatial splitting strategy that prevents adjacent training/testing pixels from sharing information, fixing up to **+15% inflated accuracy** reported in prior work.
2. **Non-Linear Extension of CCARS** — The Competitive Calibration Adaptive Reweighted Sampling (CCARS) framework is extended to SVM-RBF, achieving **75% band reduction** (204 → 50 bands) on Salinas with only **<0.6% accuracy loss**.
3. **FCovSel for Deep Learning** — The Forward Covariate Selection (FCovSel) method outperforms PLS-based methods for 3D-CNN classification, reaching **55.12% OA** on Indian Pines with just **19 bands**, while all other methods (including full-spectrum with 200 bands) fail to exceed 19%.

---

## Key Results at a Glance

### Phase 1 — Traditional ML (SVM-RBF)

| Dataset | Method | # Bands | OA (%) | Band Reduction |
|---------|--------|---------|--------|----------------|
| Salinas | Full Spectrum | 204 | 93.18 | — |
| Salinas | **CCARS-SVM-RBF** | **50** | **92.63** | **75.5%** |

### Phase 2 — Deep Learning (Adaptive 3D-CNN)

| Dataset | Method | # Bands | OA (%) |
|---------|--------|---------|--------|
| Indian Pines | **FCovSel** | **19** | **55.12** |
| Indian Pines | CARS (WST) | 50+ | <19 |
| Indian Pines | Full Spectrum | 200 | <19 |

> **Jaccard Similarity:** FCovSel selects **completely disjoint bands** (Jaccard = 0.00) from all PLS-based WST methods, proving the two approaches capture fundamentally different spectral information.

---

## Datasets

Both datasets are publicly available from the [Hyperspectral Remote Sensing Scenes](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) repository (University of the Basque Country).

| Dataset | Sensor | Size | Bands (raw) | Bands (clean) | Classes |
|---------|--------|------|-------------|----------------|---------|
| **Indian Pines** | AVIRIS | 145×145 px | 220 | **200** | 16 |
| **Salinas Valley** | AVIRIS | 512×217 px | 224 | **204** | 16 |

> The datasets (`.mat` files) are **not included** in this repository due to file size. Download them separately and place in a `dataset/` folder.

---

## Methods Compared

### Band Selection Methods

| Method | Category | Criterion |
|--------|----------|-----------|
| **CCARS** | Hybrid Wrapper | Adaptive PLS coefficients + Monte Carlo |
| CARS | Wrapper | PLS regression coefficients |
| BOSS | Wrapper | Bootstrap PLS coefficients |
| GA-iPLS | Wrapper | PLS cross-validation error (interval) |
| GA-iPLS-BOSS | Hybrid | Two-stage: GA intervals + BOSS |
| **FCovSel** | Filter | Covariance with target (classifier-free) |

### Preprocessing Pipelines

| Pipeline | Description |
|----------|-------------|
| SG-MSC | Savitzky-Golay smooth + Multiplicative Scatter Correction |
| SG-SNV | Savitzky-Golay smooth + Standard Normal Variate |
| SG1-MSC | SG first derivative + MSC |
| SG1-SNV | SG first derivative + SNV |

### Classifiers

- **Phase 1:** SVM-RBF, Random Forest, PLS-DA
- **Phase 2:** Adaptive 3D-CNN (kernel size scales with input band count)

---

## Leakage-Free Evaluation

The thesis introduces a **Checkerboard Spatial Block Split**:

- Indian Pines: 8×8 pixel blocks alternating between train/test
- Salinas: 16×16 pixel blocks alternating between train/test

This strategy ensures that train and test pixels never share horizontal or vertical boundaries, eliminating spatial autocorrelation leakage.

> **Impact:** Tested on the same models, random pixel-wise splitting overestimates accuracy by up to **15 percentage points** compared to the checkerboard protocol.

![Checkerboard Split](Figures/checkerboard_split.png)

---

## Selected Figures

<table>
  <tr>
    <td><img src="Figures/band_selection_IP.png" alt="Band Selection Indian Pines" width="350"/><br><em>Band selection comparison — Indian Pines</em></td>
    <td><img src="Figures/band_selection_Salinas.png" alt="Band Selection Salinas" width="350"/><br><em>Band selection comparison — Salinas</em></td>
  </tr>
  <tr>
    <td><img src="Figures/confusion_matrix_IP.png" alt="Confusion Matrix Indian Pines" width="350"/><br><em>3D-CNN confusion matrix — Indian Pines (FCovSel, 19 bands)</em></td>
    <td><img src="Figures/spectra_comparison.png" alt="Spectra Comparison" width="350"/><br><em>Mean reflectance spectra — both datasets</em></td>
  </tr>
</table>

---


## Acknowledgements

This work builds upon the CCARS framework introduced by:

- Dilillo et al. (2023). *Competitive Calibration Adaptive Reweighted Sampling for Hyperspectral Data.*
- Dilillo et al. (2025). *Enhancing CCARS for Non-Linear Classification.*

