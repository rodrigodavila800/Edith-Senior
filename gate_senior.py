import argparse
import json
import os
import sys


def read_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Histórico não encontrado: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def run_passes(row, min_global, min_category):
    global_avg = row.get("global", {}).get("avg_score")
    category_avg = row.get("category_avg", {})

    if global_avg is None:
        return False, "sem global.avg_score"

    if float(global_avg) < min_global:
        return False, f"global {float(global_avg):.2f} < {min_global:.2f}"

    if not category_avg:
        return False, "sem category_avg"

    low_categories = [name for name, value in category_avg.items() if float(value) < min_category]
    if low_categories:
        top = ", ".join(low_categories[:6])
        return False, f"categorias abaixo do mínimo ({min_category:.2f}): {top}"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description="Gate de promoção da Edith para nível Senior")
    parser.add_argument("--history-file", default="benchmark_history.jsonl", help="Arquivo JSONL de histórico")
    parser.add_argument("--min-global", type=float, default=3.5, help="Score global mínimo")
    parser.add_argument("--min-category", type=float, default=2.5, help="Score mínimo por categoria")
    parser.add_argument("--consecutive", type=int, default=2, help="Quantidade de rodadas consecutivas necessárias")
    parser.add_argument("--verbose", action="store_true", help="Mostra diagnóstico de cada rodada analisada")
    args = parser.parse_args()

    if args.consecutive <= 0:
        raise ValueError("--consecutive precisa ser maior que zero")

    rows = read_jsonl(args.history_file)
    if not rows:
        print("TREINAR MAIS")
        print("Motivo: histórico vazio")
        sys.exit(1)

    window = rows[-args.consecutive:]

    if len(window) < args.consecutive:
        print("TREINAR MAIS")
        print(f"Motivo: existem apenas {len(window)} rodada(s), necessário {args.consecutive}")
        sys.exit(1)

    all_ok = True
    for idx, row in enumerate(window, start=1):
        ok, reason = run_passes(row, args.min_global, args.min_category)
        all_ok = all_ok and ok
        if args.verbose:
            ts = row.get("timestamp", "sem_timestamp")
            name = row.get("run_name") or "sem_nome"
            g = row.get("global", {}).get("avg_score")
            g_str = f"{float(g):.2f}" if g is not None else "N/A"
            print(f"Rodada {idx}: {ts} | {name} | global={g_str} | {'OK' if ok else 'FALHOU'} | {reason}")

    if all_ok:
        print("APROVADA")
        print(
            f"Critério atingido em {args.consecutive} rodada(s) consecutiva(s): "
            f"global >= {args.min_global:.2f} e todas categorias >= {args.min_category:.2f}"
        )
        sys.exit(0)

    print("TREINAR MAIS")
    print(
        f"Critério não atingido nas últimas {args.consecutive} rodada(s): "
        f"global >= {args.min_global:.2f} e categorias >= {args.min_category:.2f}"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
