# Tabela padronizada de resultados (§26)

## cfp_ff_bbox

| Informação | Resultado |
|---|---|
| Modalidade | Face |
| Dataset | cfp_ff |
| CNN | Inception-ResNet-v1 (FaceNet) |
| Pré-treinamento | vggface2 (pesos congelados) |
| Nº de parâmetros | 23.48 M |
| Dimensão do embedding | 512 |
| Matching | similaridade de cosseno (L2) |
| Threshold | 0.3748 (eer) |
| EER | 1.00% |
| AUC | 0.9992 |
| FMR | 1.00% ± 0.61% |
| FNMR | 1.03% ± 0.70% |
| TAR@FAR=1% | 99.09% |
| FTA | 0.19% |
| Tempo de inferência | 18.527 ms/imagem (batch 1, cuda) |
| Tamanho do modelo | 89.9 MB |
| Hardware | NVIDIA GeForce GTX 1050 Ti / AMD Ryzen 7 1700 Eight-Core Processor |

## cfp_fp_aligned

| Informação | Resultado |
|---|---|
| Modalidade | Face |
| Dataset | cfp_fp |
| CNN | Inception-ResNet-v1 (FaceNet) |
| Pré-treinamento | vggface2 (pesos congelados) |
| Nº de parâmetros | 23.48 M |
| Dimensão do embedding | 512 |
| Matching | similaridade de cosseno (L2) |
| Threshold | 0.2514 (eer) |
| EER | 6.16% |
| AUC | 0.9825 |
| FMR | 6.20% ± 1.81% |
| FNMR | 6.14% ± 2.33% |
| TAR@FAR=1% | 85.97% |
| FTA | 1.56% |
| Tempo de inferência | 18.527 ms/imagem (batch 1, cuda) |
| Tamanho do modelo | 89.9 MB |
| Hardware | NVIDIA GeForce GTX 1050 Ti / AMD Ryzen 7 1700 Eight-Core Processor |

## cfp_fp_bbox

| Informação | Resultado |
|---|---|
| Modalidade | Face |
| Dataset | cfp_fp |
| CNN | Inception-ResNet-v1 (FaceNet) |
| Pré-treinamento | vggface2 (pesos congelados) |
| Nº de parâmetros | 23.48 M |
| Dimensão do embedding | 512 |
| Matching | similaridade de cosseno (L2) |
| Threshold | 0.2557 (eer) |
| EER | 5.09% |
| AUC | 0.9840 |
| FMR | 5.06% ± 1.32% |
| FNMR | 5.06% ± 1.82% |
| TAR@FAR=1% | 89.80% |
| FTA | 1.56% |
| Tempo de inferência | 18.527 ms/imagem (batch 1, cuda) |
| Tamanho do modelo | 89.9 MB |
| Hardware | NVIDIA GeForce GTX 1050 Ti / AMD Ryzen 7 1700 Eight-Core Processor |

## cross_cfp_fp_to_lfw

| Informação | Resultado |
|---|---|
| Modalidade | Face |
| Dataset | lfw |
| CNN | Inception-ResNet-v1 (FaceNet) |
| Pré-treinamento | vggface2 (pesos congelados) |
| Nº de parâmetros | 23.48 M |
| Dimensão do embedding | 512 |
| Matching | similaridade de cosseno (L2) |
| Threshold | 0.2561 (eer) |
| EER | 0.57% |
| AUC | 0.9995 |
| FMR | 6.73% |
| FNMR | 0.17% |
| TAR@FAR=1% | 99.60% |
| FTA | 0.01% |
| Tempo de inferência | 18.527 ms/imagem (batch 1, cuda) |
| Tamanho do modelo | 89.9 MB |
| Hardware | NVIDIA GeForce GTX 1050 Ti / AMD Ryzen 7 1700 Eight-Core Processor |

## lfw_aligned

| Informação | Resultado |
|---|---|
| Modalidade | Face |
| Dataset | lfw |
| CNN | Inception-ResNet-v1 (FaceNet) |
| Pré-treinamento | vggface2 (pesos congelados) |
| Nº de parâmetros | 23.48 M |
| Dimensão do embedding | 512 |
| Matching | similaridade de cosseno (L2) |
| Threshold | 0.4074 (eer) |
| EER | 0.88% |
| AUC | 0.9994 |
| FMR | 0.93% ± 0.41% |
| FNMR | 0.90% ± 0.90% |
| TAR@FAR=1% | 99.23% |
| FTA | 0.01% |
| Tempo de inferência | 18.527 ms/imagem (batch 1, cuda) |
| Tamanho do modelo | 89.9 MB |
| Hardware | NVIDIA GeForce GTX 1050 Ti / AMD Ryzen 7 1700 Eight-Core Processor |

## lfw_bbox

| Informação | Resultado |
|---|---|
| Modalidade | Face |
| Dataset | lfw |
| CNN | Inception-ResNet-v1 (FaceNet) |
| Pré-treinamento | vggface2 (pesos congelados) |
| Nº de parâmetros | 23.48 M |
| Dimensão do embedding | 512 |
| Matching | similaridade de cosseno (L2) |
| Threshold | 0.4098 (eer) |
| EER | 0.57% |
| AUC | 0.9995 |
| FMR | 0.60% ± 0.49% |
| FNMR | 0.63% ± 0.74% |
| TAR@FAR=1% | 99.60% |
| FTA | 0.01% |
| Tempo de inferência | 18.527 ms/imagem (batch 1, cuda) |
| Tamanho do modelo | 89.9 MB |
| Hardware | NVIDIA GeForce GTX 1050 Ti / AMD Ryzen 7 1700 Eight-Core Processor |
