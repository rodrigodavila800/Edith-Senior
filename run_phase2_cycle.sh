#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python do venv não encontrado em ${PYTHON_BIN}"
  exit 1
fi

RUN_TAG="${1:-phase2_$(date +%Y%m%d_%H%M%S)}"
MANIFEST="${ROOT_DIR}/curated_manifest.phase2.txt"
STATS="${ROOT_DIR}/curated_stats.phase2.json"
HISTORY="${ROOT_DIR}/benchmark_history.jsonl"

pushd "${ROOT_DIR}" >/dev/null

# 1) Curadoria com viés elite + novos dados
echo "[1/4] Curadoria Phase 2..."
"${PYTHON_BIN}" curadoria_dataset.py \
  --data-dir data \
  --elite-only \
  --min-weight 1.2 \
  --target-files 18000 \
  --mix-code 0.62 \
  --mix-docs 0.23 \
  --manifest "${MANIFEST}" \
  --stats "${STATS}"

# 2) Treino incremental (continua do checkpoint)
echo "[2/4] Treino Phase 2..."
RESUME=1 \
DATA_MANIFEST="${MANIFEST}" \
TRAIN_STEPS=28000 \
CHECKPOINT_EVERY=5 \
EVAL_INTERVAL=200 \
GRAD_ACCUM=4 \
BASE_LR=6e-5 \
MIN_LR=5e-6 \
WARMUP_STEPS=200 \
MAX_GRAD_NORM=0.8 \
STEP_SLEEP=0.08 \
"${PYTHON_BIN}" marco5_train.py

# 3) Benchmark mastery
echo "[3/4] Benchmark Phase 2..."
"${PYTHON_BIN}" batismo_edith.py \
  --model edith_laravel_senior_best.pt \
  --prompt-file prompts_laravel_mastery.txt \
  --samples 2 \
  --tokens 220 \
  --temperature 0.62 \
  --top-k 28 \
  --save-history \
  --history-file "${HISTORY}" \
  --run-name "${RUN_TAG}"

# 4) Gate de promoção
echo "[4/4] Gate Senior..."
"${PYTHON_BIN}" gate_senior.py \
  --history-file "${HISTORY}" \
  --verbose || true

# Gráficos atualizados
"${PYTHON_BIN}" plot_benchmark_history.py --history-file "${HISTORY}" --png benchmark_evolucao.png

if [[ -f "${ROOT_DIR}/benchmark_top5.png" ]]; then
  rm -f "${ROOT_DIR}/benchmark_top5.png"
fi

"${PYTHON_BIN}" - <<'PY'
import json
import matplotlib.pyplot as plt

history='benchmark_history.jsonl'
out='benchmark_top5.png'
focus=['store','api','security','validation','tests']
rows=[]
with open(history,'r',encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if line:
            rows.append(json.loads(line))

x=[]
g=[]
series={k:[] for k in focus}
for i,row in enumerate(rows, start=1):
    avg=row.get('global',{}).get('avg_score')
    if avg is None:
        continue
    x.append(i)
    g.append(float(avg))
    cat=row.get('category_avg',{})
    for k in focus:
        series[k].append(float(cat.get(k, float('nan'))))

plt.figure(figsize=(11,6))
plt.plot(x,g,label='global',linewidth=2,color='#1f77b4')
for k,v in series.items():
    plt.plot(x,v,'--',label=k,alpha=0.85)
plt.ylim(0,6)
plt.xlabel('Rodada')
plt.ylabel('Score médio')
plt.title('Edith - Evolução (Top 5 Categorias)')
plt.grid(alpha=0.25)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(out,dpi=150)
PY

echo "Fase 2 concluída. Run tag: ${RUN_TAG}"

echo "Arquivos-chave:"
echo "- ${MANIFEST}"
echo "- ${STATS}"
echo "- ${HISTORY}"
echo "- benchmark_evolucao.png"
echo "- benchmark_top5.png"

popd >/dev/null
