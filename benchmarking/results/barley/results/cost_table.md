| Method | Params (M) | GFLOPs @512 | Inference GPU (ms/img) | Inference CPU (s/img) | Peak GPU mem (GB) |
|---|---|---|---|---|---|
| BSA (SDA-UNet, ours) | 3.34 | 24.8 | 50.5 ± 3.0 | 0.25 ± 0.01 | 0.25 |
| DeepLabV3+ (ResNet-50) | 17.83 | 95.2 | 54.4 ± 1.9 | 0.49 ± 0.01 | 0.40 |
| FCN-ResNet-50 | 23.60 | 40.4 | 63.2 ± 22.8 | 0.35 ± 0.01 | 0.19 |
| PSPNet (ResNet-50) | 16.20 | 47.8 | 48.2 ± 1.5 | 0.52 ± 0.01 | 0.63 |
| Random Forest | n/a | n/a | n/a | n/a | n/a |
| SLIC + RF | n/a | n/a | n/a | n/a | n/a |
| Linear SVM | n/a | n/a | n/a | n/a | n/a |

Host: 12th Gen Intel(R) Core(TM) i7-12700KF, 31.8 GB RAM, Windows-10-10.0.26200-SP0