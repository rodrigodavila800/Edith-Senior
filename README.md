# Edith Senior Laravel

Projeto de treinamento e avaliação de um modelo autoregressivo focado no ecossistema Laravel, com pipeline resiliente a quedas de energia, benchmark por competências e gate automático de aprovação.

## Visão geral

A Edith foi treinada para completar código e padrões Laravel (CRUD, Auth, API, validação, testes, filas, eventos, rotas e segurança), usando corpus curado do ecossistema Laravel.

Pipeline principal:

1. Curadoria do dataset
2. Treino com checkpoint/resume
3. Benchmark multiprompt por categoria
4. Histórico e gráficos de evolução
5. Gate de promoção (Senior Ready)

## Estrutura principal

- `marco3_tokenizer.py`: treinamento do tokenizer BPE
- `marco4_model.py`: arquitetura do modelo
- `marco5_train.py`: treino (com resume/checkpoint, validação e melhor modelo)
- `curadoria_dataset.py`: curadoria e balanceamento do corpus
- `batismo_edith.py`: benchmark multiprompt por categoria
- `plot_benchmark_history.py`: relatório e gráfico da evolução
- `gate_senior.py`: decisão automática APROVADA/TREINAR MAIS

Artefatos:

- `edith_laravel_senior.pt`: modelo final salvo
- `edith_laravel_senior_best.pt`: melhor modelo por validação
- `edith_laravel_senior_ckpt.pt`: checkpoint completo de treino
- `benchmark_history.jsonl`: histórico de benchmarks
- `benchmark_evolucao.png`: gráfico completo de evolução
- `benchmark_top5.png`: gráfico focado em categorias-chave

## Setup rápido

Ativar ambiente virtual:

```bash
source /home/rodrigo/Edith_Senior_Laravel/venv/bin/activate
```

## 1) Curadoria do dataset

Exemplo de curadoria elite:

```bash
python curadoria_dataset.py \
  --elite-only \
  --min-weight 1.2 \
  --target-files 15000 \
  --manifest curated_manifest.elite.txt \
  --stats curated_stats.elite.json
```

## 2) Treino resiliente (resume + checkpoints)

Preset ultra-frio (recomendado para GPU fraca/instável):

```bash
RESUME=1 \
DATA_MANIFEST=curated_manifest.elite.txt \
TRAIN_STEPS=20000 \
CHECKPOINT_EVERY=5 \
EVAL_INTERVAL=250 \
GRAD_ACCUM=4 \
BASE_LR=1e-4 \
MIN_LR=8e-6 \
WARMUP_STEPS=400 \
MAX_GRAD_NORM=0.8 \
STEP_SLEEP=0.08 \
python marco5_train.py
```

Modo blindado em CPU (se a máquina continuar apagando):

```bash
FORCE_CPU=1 RESUME=1 DATA_MANIFEST=curated_manifest.elite.txt TRAIN_STEPS=20000 python marco5_train.py
```

## 3) Benchmark de competências

Rodar benchmark completo e salvar no histórico:

```bash
python batismo_edith.py \
  --model edith_laravel_senior_best.pt \
  --prompt-file prompts_laravel_mastery.txt \
  --samples 2 \
  --tokens 220 \
  --temperature 0.65 \
  --top-k 30 \
  --save-history \
  --history-file benchmark_history.jsonl \
  --run-name pos_20k
```

## 4) Relatórios e gráficos

Resumo da evolução:

```bash
python plot_benchmark_history.py --history-file benchmark_history.jsonl
```

Gerar gráfico PNG:

```bash
python plot_benchmark_history.py --history-file benchmark_history.jsonl --png benchmark_evolucao.png
```

## 5) Gate de aprovação (Senior Ready)

Critério padrão:

- score global >= 3.5
- todas as categorias >= 2.5
- por 2 rodadas consecutivas

Comando:

```bash
python gate_senior.py --history-file benchmark_history.jsonl --verbose
```

Saídas possíveis:

- `APROVADA`
- `TREINAR MAIS`

## Estado atual do projeto

- Treino concluído em 20k passos
- Melhor `val_loss` registrado em torno de 2.36
- Gate já atingiu estado `APROVADA` em rodadas consecutivas recentes

## Próximos passos sugeridos

- Criar script de inferência para uso diário (prompt -> completion)
- Adicionar benchmark de regressão por arquivo de fixtures
- Exportar relatório final (JSON + PNG) para apresentação

## Fase 2 (especialização contínua)

Arquivos adicionados para automação da próxima fase:

- `phase2_repos.txt`: lista curada de 20 repositórios para ampliar o corpus
- `sync_phase2_repos.sh`: clona/atualiza automaticamente os repositórios da Fase 2 em `data/`
- `run_phase2_cycle.sh`: executa pipeline completo (curadoria -> treino -> benchmark -> gate -> gráficos)

### Sincronizar novos datasets

```bash
./sync_phase2_repos.sh
```

### Rodar ciclo completo da Fase 2

```bash
./run_phase2_cycle.sh
```

Opcionalmente você pode passar um identificador de rodada:

```bash
./run_phase2_cycle.sh fase2_lote_a
```
