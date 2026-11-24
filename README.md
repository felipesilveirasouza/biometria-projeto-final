# Verificação facial 1:1 com Inception-ResNet-v1 (FaceNet) em CFP-FP e LFW

Projeto final da disciplina de Biometria (UTFPR). Sistema de verificação
biométrica 1:1: cada amostra vira um embedding gerado por uma CNN, e duas
amostras são comparadas por similaridade de cosseno com um threshold
definido só na validação.

| Item | Valor |
|---|---|
| Modalidade | Face |
| Base principal | CFP-FP (Celebrities in Frontal-Profile, cenário frontal × perfil) |
| Segunda base | LFW (Labeled Faces in the Wild, View 2) |
| CNN | Inception-ResNet-v1 (FaceNet), família Inception |
| Origem dos pesos | [facenet-pytorch](https://github.com/timesler/facenet-pytorch), convertidos de [davidsandberg/facenet](https://github.com/davidsandberg/facenet) |
| Dataset de pré-treinamento | VGGFace2 (`model.pretrained` no `config.yaml`) |
| Camadas congeladas / treinadas | todas congeladas / nenhuma (sem fine-tuning) |
| Função de perda (pré-treino) | softmax de classificação de identidade (modelo `20180402-114759`, sem center loss nem triplet; LFW reportado 0,9965 ± 0,0025 no repositório original) |
| Dimensão do embedding | 512, normalizado em L2 |
| Nº de parâmetros / tamanho | gerados em `results/efficiency.json` |
| Matching | similaridade de cosseno (maior = mais parecido) |
| Threshold | ponto de EER nos folds de validação (`threshold.rule`), congelado antes do teste |
| Seeds | 42 (amostragens e posição das oclusões). Sem treino, então não há repetição com 3 seeds |

## Pipeline

```
imagem bruta
 → controle de qualidade (abre? MTCNN acha rosto com p ≥ 0,90? senão conta FTA)
 → detecção e localização (MTCNN: bbox + 5 landmarks)
 → seleção do rosto (maior área penalizada pela distância ao centro)
 → recorte pela bbox (margem 20 px) e redimensionamento para 160×160
 → normalização de intensidade (x − 127,5) / 128
 → Inception-ResNet-v1 → embedding 512-D (L2)
 → cosseno entre os dois embeddings → score
 → threshold (da validação, congelado) → genuíno / impostor
```

**Por que recorte pela bbox, e não alinhamento.** O modelo foi pré-treinado com
recortes da MTCNN pela bbox (FaceNet original). Por isso essa é a configuração
principal (`preprocess.align: false`). O alinhamento por 5 landmarks para o
template do ArcFace é avaliado como ablação (`--align`) e piora o resultado nas
duas bases.

**Seleção do rosto.** Com vários rostos na imagem, escolhe-se o de maior área
penalizada pela distância ao centro, o mesmo critério de `align_dataset_mtcnn.py`
do FaceNet original. Escolher só o maior pega pessoas ao fundo em parte do LFW.

**Alinhamento em perfil (ablação).** O template de 5 pontos é frontal. Em imagens de
perfil a MTCNN costuma posicionar landmarks fora do rosto, e o alinhamento
gera recortes girados e quase vazios. Por isso os landmarks passam por uma
verificação de plausibilidade antes de alinhar: todos dentro da imagem e da
bbox, nariz entre os dois olhos e rotação ≤ `max_align_rotation` (30°). Se
algum critério falhar, usa-se o recorte pela bbox, e o motivo fica em
`cache/crops/<variante>/<base>/quality.csv` (coluna `method`). Numa amostra
de 150 imagens por grupo, 95% das frontais do CFP e 99% das imagens do LFW
foram alinhadas por landmarks, contra só 23% dos perfis do CFP.

Quando a detecção falha, `preprocess.fta_policy` decide: `fallback` (padrão)
usa um recorte central e segue, contando a falha no FTA; `reject` não gera
template e o par recebe score −1 (sempre rejeitado).

## Protocolo experimental

As duas bases têm protocolo oficial de 10 folds, e ele é usado (§12):

- **CFP-FP**: `Protocol/Split/FP/01..10`, 350 pares genuínos + 350 impostores por fold (frontal × perfil).
- **LFW**: `pairs.txt` da View 2, 300 + 300 pares por fold.

Para cada fold *k*, o threshold é escolhido nos outros 9 folds (validação),
congelado e aplicado ao fold *k* (teste). FMR, FNMR e acurácia são reportados
como média ± desvio-padrão dos 10 folds. EER, AUC, ROC e TAR@FAR=1% não
dependem do threshold e usam todos os pares. `make_protocols.py` verifica se
os folds são disjuntos por identidade e grava o resultado em
`protocols/<base>/summary.json`.

Como o modelo é usado congelado, não existe conjunto de treino. Por isso os
arquivos `train.csv`/`validation.csv`/`test.csv` do enunciado são substituídos
pela coluna `fold` de `protocols/<base>/pairs.csv`.

| Experimento | O que é | Comando | Saída |
|---|---|---|---|
| A | CFP-FP, protocolo oficial | `python evaluate.py --dataset cfp_fp` | `results/cfp_fp_bbox/` |
| B | LFW, protocolo oficial | `python evaluate.py --dataset lfw` | `results/lfw_bbox/` |
| C | Cross-dataset: threshold global do CFP-FP congelado e aplicado ao LFW, sem ajuste | `python evaluate.py --dataset lfw --frozen-from cfp_fp_bbox --tag cross_cfp_fp_to_lfw` | `results/cross_cfp_fp_to_lfw/` |
| D | Robustez: blur (σ = 0; 1,5; 3; 5) e oclusão (0; 15; 30; 45% do lado) na 2ª amostra | `python robustness.py` | `results/robustness/` |
| Ablação | Alinhamento por 5 landmarks em vez do recorte pela bbox, nas duas bases | ver `run_all.sh` | `results/cfp_fp_aligned/`, `results/lfw_aligned/` |
| Erros | 3 falsos matches + 3 falsos non-matches | `python errors.py --tag cfp_fp_bbox` | `errors.csv`, `errors.png` |

## Como obter os dados

As bases **não** estão no repositório (§22). Baixe das fontes oficiais e
organize assim:

```
data/
├── cfp-dataset/          # http://www.cfpw.io/ (CFP, Sengupta et al., WACV 2016)
│   ├── Data/Images/001/frontal/01.jpg …
│   └── Protocol/{Pair_list_F.txt, Pair_list_P.txt, Split/FP/01..10/{same,diff}.txt}
└── lfw/                  # http://vis-www.cs.umass.edu/lfw/ (lfw.tgz, versão original)
    ├── pairs.txt
    └── Aaron_Eckhart/Aaron_Eckhart_0001.jpg …
```

Os caminhos podem ser trocados em `config.yaml` (`paths`).

## Ambiente

Python **3.12**. O `facenet-pytorch` 2.6.0 exige torch 2.2, que não tem wheel para Python 3.13 ou mais novo.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Os pesos da Inception-ResNet-v1 (cerca de 100 MB) são baixados automaticamente na primeira execução.

## Reproduzir tudo

```bash
./run_all.sh
```

Isso gera `results/` (`scores.csv`, `metrics.json`, `roc.png`,
`distributions.png` de cada experimento, `efficiency.json`, `robustness/`,
`errors.png`, `pipeline_*.png`, `tabela_padronizada.md`).

## Estrutura

```
config.yaml             parâmetros de todos os experimentos
common.py               protocolos, métricas (EER, FMR, FNMR, AUC, TAR@FAR) e gráficos
make_protocols.py       protocols/<base>/ a partir dos protocolos oficiais
preprocess.py           qualidade, detecção, alinhamento, recorte e degradações
model.py                Inception-ResNet-v1, normalização, nº de parâmetros, hardware
extract_embeddings.py   embeddings + tempo de inferência
evaluate.py             scores, threshold, métricas e gráficos
robustness.py           experimento D
errors.py               análise de erros
figures.py              figura das etapas do pipeline
report.py               tabela padronizada (§26)
run_all.sh              tudo, na ordem
```

## Limitações conhecidas

- O VGGFace2 é composto por celebridades, assim como o LFW e o CFP. Pode haver
  identidades em comum entre o pré-treino e as bases de teste; isso precisa
  ser verificado e discutido no artigo.
- O tempo de inferência medido é só o da CNN; a detecção com a MTCNN não entra.

## Versão avaliada

Repositório: _preencher_ · Versão: tag `final-v1.0` (ou hash do commit)
