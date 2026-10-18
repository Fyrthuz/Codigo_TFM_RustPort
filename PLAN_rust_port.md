# Plan de port a Rust (`tch-rs`) — Codigo_TFM

> Documento de planificación del port del proyecto Python/PyTorch a Rust usando
> `tch-rs` (bindings de libtorch). Este directorio es una copia independiente del
> repositorio original (`~/Desktop/Codigo_TFM`); **el original no se modifica**.

## Objetivo

Reimplementar en Rust la funcionalidad completa del TFM — *Medical Image
Segmentation with Uncertainty Estimation* — manteniendo **equivalencia
estadística** con el original (mismo Dice/IoU/ECE/p-valores dentro del ruido de
30 muestras MC/TTA/ruido, y el mismo split por paciente con `seed=42`). **No** se
busca bit-exactitud con numpy/torch.

Alcance acordado (completo):

- Inferencia UNet 2D y UniVerSeg few-shot.
- Métodos de incertidumbre: Normal, MC Dropout, TTA, Noisy, Fusion, CRF.
- Métricas (IoU, Dice, NLL, NLL_fg, ECE, Brier, accuracy, precision, recall,
  certainty).
- Estadística a nivel de paciente (bootstrap por clusters + Wilcoxon).
- Entrenamiento (`train_unet`).
- Generación de figuras (`make_figures` → `docs/*.png`).

## Decisiones cerradas

1. **`tch-rs`** como motor DL (libtorch 2.5+, CUDA 12.x).
2. **Ambos modelos reimplementados en Rust** (grafo a mano en `tch-rs::nn`), con
   **carga de pesos 1:1 vía safetensors**:
   - UNet desde `unet_model.pth`.
   - UniVerSeg desde el release oficial
     `universeg_v1_nf64_ss64_STA.pt` (licencia OpenRAIL++-M, solo investigación).
   - Esto habilita MC Dropout (inyección de máscaras en `forward`) para los dos
     modelos, ya que `tch-rs` no expone *forward hooks* sobre módulos opacos.
3. **MC Dropout** reproduce la *semántica* del original (máscara Bernoulli
   independiente `p=0.01` sobre la salida de cada capa, 30 muestras), no la
   colocación bit-exacta de hooks → equivalencia estadística.
4. **Paridad de resultados** validada contra los CSVs generados por Python
   (`results/visualizations/*.csv`, `results_foundation_universeg/...`).

## Estado del fork

- No es posible un fork de GitHub (la cuenta `Fyrthuz` ya es dueña del repo).
- Se creó un **clon local independiente** en `~/Desktop/Codigo_TFM_rust`
  (`origin = git@github.com:Fyrthuz/Codigo_TFM.git`).
- El código Python original permanece intacto en `~/Desktop/Codigo_TFM`.

## Fiabilidad del port de UniVerSeg (1:1)

Se revisó la arquitectura real (`JJGO/UniverSeg` → `universeg/model.py` y
`universeg/nn/*`). Es reimplementable de forma exacta:

| Componente | Qué hace | Equivalente `tch-rs` |
|---|---|---|
| `CrossConv2d` | Conv2d sobre canales concatenados; meshgrid Sx×Sy vía repeat/cat/reshape | `unsqueeze` + `repeat`/`expand` + `cat` + `reshape` + `conv2d` |
| `vmap` / `Vmap` | reshape `(B,S,…)→(B·S,…)`, aplicar módulo, reshape | `reshape` + llamada + `reshape` |
| `ConvOp` | `Conv2d` + `LeakyReLU` (o identidad) | `nn::conv2d` + `relu` con pendiente negativa |
| `UniverSeg` | MaxPool2d + upsample bilineal + 4 CrossBlock enc + 3 dec + conv 1×1 | estándar en `tch-rs` |

Pesos (`state_dict` con claves deterministas
`enc_blocks.{i}.cross.cross_conv.weight`, `enc_blocks.{i}.target.vmapped.conv.weight`,
…). Exportables por nombre a safetensors y cargables en un grafo Rust con los
mismos nombres de parámetros.

## Estructura del workspace Cargo

```
Codigo_TFM_rust/
├── Cargo.toml                       # workspace (members = crates/*)
├── crates/
│   ├── core/                        # SIN tch — compila/testea sin libtorch/CUDA
│   │   └── src/
│   │       ├── metrics.rs           # ioU, dice, classwise_ece, nll, nll_fg, brier, certainty
│   │       ├── fusion.rs            # weighted_average_with_uncertainty + otsu/percentile/mean_std
│   │       ├── crf.rs               # dense CRF mean-field (opencv: sobel/gaussian/bilateral/resize)
│   │       ├── dataset.rs           # escaneo TIFF, extract_patient_id, split_by_patient, índices JSON
│   │       ├── config.rs            # serde_yaml → structs de configuración (PipelineConfig y sub)
│   │       ├── statistics.rs        # bootstrap por paciente + Wilcoxon signed-rank exacto
│   │       └── io.rs                # CSV (metrics_summary/detailed/statistical) + PNG (save_image)
│   └── app/                         # CON tch
│       └── src/
│           ├── models/
│           │   ├── unet.rs          # UNet + carga safetensors
│           │   ├── universeg.rs     # CrossConv2d, Vmap, CrossBlock, UniverSeg + carga safetensors
│           │   └── mod.rs
│           ├── uncertainty/
│           │   ├── mc_dropout.rs    # máscaras inyectadas en forward (p, num_samples)
│           │   ├── tta.rs           # RandomImageTransformer (affine) + 9 combos resize-safe
│           │   └── noise.rs         # NoisyInference (gaussiano σ, clamp 0..1)
│           ├── pipelines/
│           │   ├── base.rs          # orquestación de métodos + guardado por muestra
│           │   ├── unet.rs          # carga test_indices.json, checkpoint
│           │   └── foundation.rs    # support/test, canal G (T1c)
│           ├── train.rs             # Adam, BCE+Dice, augmentation, early stopping
│           ├── viz.rs               # plotters: barras, box+jitter, forest plot, paneles cualitativos
│           └── bin/
│               ├── run_unet.rs
│               ├── run_foundation.rs
│               ├── train_unet.rs
│               ├── statistics.rs
│               └── make_figures.rs
├── scripts/
│   ├── export_safetensors.py        # UNet + UniVerSeg → .safetensors (1 vez, con .venv original)
│   ├── export_torchscript.py        # traza ambos → .pt (referencia para tests numéricos)
│   └── download_data.sh
├── configs/                         # YAML copiados (pipeline_2d, foundation_universeg, *_4pct)
└── tests/                           # ports de los 68 tests (prioridad: metrics, fusion, crf, config, dataset, statistics)
```

## Dependencias (crates)

| Crate | Uso |
|---|---|
| `tch` | tensor + `nn` + optimizadores (libtorch 2.5+, CUDA) |
| `ndarray`, `ndarray-stats` | arrays, `mean`/`std`/`percentile` |
| `image` (+ `image-tiff`) | carga TIFF/PNG, `save_image` |
| `opencv` | CRF: `Sobel`, `GaussianBlur`, `bilateralFilter`, `cvtColor`, `resize` |
| `serde` + `serde_yaml` | configuración YAML |
| `csv` + `serde` | CSVs de resultados |
| `safetensors` | carga de pesos de modelos |
| `plotters` | figuras (`docs/*.png`) |
| `clap` | CLI de los binarios |
| `rand` (+ seedable) | RNG reproducible (bootstrap, ruido, MC, TTA) |
| `anyhow` | manejo de errores |

## Fases

### Fase 0 — Fork y andamiaje ✅
- [x] Clon del repo a `~/Desktop/Codigo_TFM_rust`.
- [ ] `Cargo.toml` workspace + `crates/core` y `crates/app` vacíos.
- [ ] `.gitignore` ampliado (`target/`, `MRI/`, `*.pth`, `*.pt`, `*.safetensors`,
      `results*/`).
- [ ] Instalar/verificar libtorch (CUDA) y compilar `cargo build` de smoke-test.

### Fase 1 — Port de `core` (sin `tch`) + tests
Port 1:1 de lógica pura y sus tests:
- [ ] `metrics.rs` (IoU, Dice, `classwise_ece`, NLL, NLL_fg, Brier, certainty).
- [ ] `fusion.rs` (weighting inverse/exponential/powerlaw + Otsu manual/percentile/mean_std).
- [ ] `crf.rs` (kernels Gaussiano + bilateral, edge-gate, 3 iteraciones).
- [ ] `dataset.rs` (escaneo de `.tif`/`_mask.tif`, `split_by_patient`, JSON índices).
- [ ] `config.rs` (parseo YAML con `serde`).
- [ ] `statistics.rs` (bootstrap por clusters 10.000 iter + **Wilcoxon exacto**).
- [ ] `io.rs` (CSV largo/detallado/resumen + `save_image`).
- [ ] Port de `test_metrics.py`, `test_fusion.py`, `test_crf.py`, `test_config.py`,
      `test_datasets.py`, `test_statistics.py` → `#[test]`.
- [ ] **Verificación cruzada**: mismas entradas → mismos valores (métricas, CRF,
      fusión, p-valores) que las salidas Python.

### Fase 2 — Export de pesos (un paso, en el `.venv` original)
- [ ] `scripts/export_safetensors.py` (UNet → `unet_model.safetensors`;
      UniVerSeg → `universeg.safetensors` con claves de `state_dict`).
- [ ] `scripts/export_torchscript.py` (traza UNet y UniVerSeg → `.pt` como
      referencia para los tests numéricos de la Fase 3).

### Fase 3 — Modelos en `tch-rs`
- [ ] `models/unet.rs` + carga safetensors.
- [ ] `models/universeg.rs` (`CrossConv2d`, `Vmap`, `CrossBlock`, `UniverSeg`) +
      carga safetensors.
- [ ] **Test numérico**: forward Rust vs TorchScript trazado (diff máx < 1e-5)
      sobre entradas aleatorias y sobre imágenes reales del dataset.

### Fase 4 — Incertidumbre + pipelines
- [ ] `uncertainty/mc_dropout.rs` (máscaras en `forward`, `p=0.01`, 30 muestras).
- [ ] `uncertainty/tta.rs` (affine flip+scale+intensity para UNet; 9 transformaciones
      fotométricas resize-safe para UniVerSeg).
- [ ] `uncertainty/noise.rs` (gaussiano `σ=0.01` UNet / `0.1` UniVerSeg).
- [ ] `pipelines/base.rs` con la orquestación y estructura de salida idéntica a
      `src/pipelines/base.py` (directorios `original/`, `mc_dropout/`, `tta/`,
      `noisy/`, `fusion/`, `refined/`, `visualizations/`).
- [ ] `bin/run_unet.rs` y `bin/run_foundation.rs` (canal G/T1c en foundation).
- [ ] Port de `test_models.py`, `test_mc_dropout.py`, `test_tta.py`,
      `test_noise.py`, `test_foundation_models.py`.

### Fase 5 — Entrenamiento
- [ ] `train.rs`: Adam, loss combinada BCE+Dice, augmentation
      (flip/affine/color-jitter) con semilla compartida imagen/máscara,
      early stopping, `split_by_patient(seed=42)`.

### Fase 6 — Figuras (`plotters`)
- [ ] `viz.rs`: `metrics_summary.png` (barras), `enhanced_metrics_comparison.png`,
      `box_plot_comparison.png` (box+jitter), paneles cualitativos y forest plot
      de `statistical_tests.csv` (equivalentes a `docs/*.png`).

### Fase 7 — Verificación end-to-end
- [ ] Ejecutar `run_unet` y `run_foundation` sobre los datos reales.
- [ ] Comparar `detailed_metrics.csv` y `statistical_tests.csv` contra los del
      repo original (`results/`, `results_foundation_universeg/`): Dice/IoU/ECE
      dentro del ruido de muestreo; p-valores del mismo orden.
- [ ] Reproducir protocolo 4% (`protocol_subset` + configs `*_4pct.yaml`).

## Riesgos y mitigación

| Riesgo | Mitigación |
|---|---|
| Bugs de reshape en `CrossConv2d`/`vmap` | Test numérico vs TorchScript (Fase 3) |
| Wilcoxon no coincide con scipy | Implementación exacta (n≤25) + validación contra `statistical_tests.csv` |
| Bilateral/Sobel de OpenCV en Rust | crate `opencv` (misma implementación que `cv2`) |
| `plotters` (box+jitter/forest) tedioso | Fase 6, aislado en `viz.rs`; no bloquea las fases 1–5 |
| libtorch/CUDA no instalado | `crates/core` compila/testea sin `tch`; Fase 0 verifica CUDA |

## Notas

- Los pesos de UniVerSeg están bajo OpenRAIL++-M (solo investigación); correcto
  para el TFM.
- `unet_model.pth` está versionado en git (17 MB) y `MRI/` también; se mantienen
  fuera del build Rust pero disponibles para las verificaciones.
- Los binarios usan `clap` para replicar los argumentos de los entry points
  Python (`--config`, `--checkpoint`, `--test-indices`, `--context-size`, etc.).
