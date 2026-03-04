import argparse
import importlib
import json
import os


SPARK_CHARS = "▁▂▃▄▅▆▇█"


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


def sparkline(values):
    if not values:
        return ""
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return SPARK_CHARS[0] * len(values)
    out = []
    for v in values:
        idx = int((v - vmin) / (vmax - vmin) * (len(SPARK_CHARS) - 1))
        out.append(SPARK_CHARS[max(0, min(idx, len(SPARK_CHARS) - 1))])
    return "".join(out)


def build_series(rows):
    global_series = []
    by_category = {}

    for idx, row in enumerate(rows, start=1):
        g = row.get("global", {})
        score = g.get("avg_score")
        if score is None:
            continue

        global_series.append((idx, float(score), row.get("timestamp"), row.get("run_name")))

        cat = row.get("category_avg", {})
        for name, value in cat.items():
            by_category.setdefault(name, []).append((idx, float(value)))

    return global_series, by_category


def print_summary(global_series, by_category):
    if not global_series:
        print("Sem dados de score no histórico.")
        return

    global_values = [x[1] for x in global_series]
    print("=" * 80)
    print("📈 EVOLUÇÃO GLOBAL")
    print("=" * 80)
    print(f"Rodadas válidas: {len(global_values)}")
    print(f"Último score: {global_values[-1]:.2f}/6")
    print(f"Melhor score: {max(global_values):.2f}/6")
    print(f"Sparkline: {sparkline(global_values)}")

    print("\n" + "=" * 80)
    print("📚 EVOLUÇÃO POR CATEGORIA")
    print("=" * 80)
    for category, points in sorted(by_category.items()):
        values = [x[1] for x in points]
        if not values:
            continue
        print(f"- {category:12s} último={values[-1]:.2f}/6 melhor={max(values):.2f}/6  {sparkline(values)}")


def maybe_plot_png(global_series, by_category, output_png):
    if not output_png:
        return

    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception:
        print("⚠️ matplotlib não disponível; gráfico PNG não foi gerado.")
        return

    if not global_series:
        print("⚠️ Sem série global para plotar.")
        return

    x_global = [x[0] for x in global_series]
    y_global = [x[1] for x in global_series]

    plt.figure(figsize=(11, 6))
    plt.plot(x_global, y_global, label="global", linewidth=2)

    for category, points in sorted(by_category.items()):
        xs = [x[0] for x in points]
        ys = [x[1] for x in points]
        plt.plot(xs, ys, linestyle="--", alpha=0.6, label=category)

    plt.ylim(0, 6)
    plt.xlabel("Rodada")
    plt.ylabel("Score médio")
    plt.title("Evolução do Benchmark da Edith")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png, dpi=140)
    plt.close()
    print(f"🖼️ Gráfico salvo em: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Plota e resume evolução histórica do benchmark da Edith")
    parser.add_argument("--history-file", default="benchmark_history.jsonl", help="Arquivo JSONL de histórico")
    parser.add_argument("--png", default="", help="Caminho opcional para salvar gráfico PNG")
    args = parser.parse_args()

    rows = read_jsonl(args.history_file)
    global_series, by_category = build_series(rows)
    print_summary(global_series, by_category)
    maybe_plot_png(global_series, by_category, args.png)


if __name__ == "__main__":
    main()
