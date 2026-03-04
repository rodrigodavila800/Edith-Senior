import argparse
import hashlib
import json
import os
import random
from pathlib import Path


DEFAULT_EXTENSIONS = {".php", ".md", ".json"}
DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "vendor",
    "node_modules",
    "dist",
    "build",
    "storage",
    "bootstrap/cache",
    "coverage",
    "tmp",
    "temp",
    "__pycache__",
}

CODE_EXTENSIONS = {".php"}
DOC_EXTENSIONS = {".md"}
TEST_HINTS = {"/tests/", "test", "pest", "phpunit"}
ELITE_REPO_HINTS = {
    "/framework/",
    "/laravel/",
    "/docs/",
    "/horizon/",
    "/octane/",
    "/sanctum/",
    "/passport/",
    "/fortify/",
    "/scout/",
    "/telescope/",
    "/pint/",
    "/jetstream/",
    "/breeze/",
    "/reverb/",
}


def normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def file_bucket(path: str) -> str:
    p = path.lower()
    if any(h in p for h in TEST_HINTS):
        return "tests"
    ext = Path(path).suffix.lower()
    if ext in DOC_EXTENSIONS:
        return "docs"
    if ext in CODE_EXTENSIONS:
        return "code"
    return "other"


def repo_weight(path: str) -> float:
    p = path.lower()
    # Peso maior para bases oficiais e pacote de runtime
    high = [
        "/framework/",
        "/laravel/",
        "/docs/",
        "/horizon/",
        "/octane/",
        "/sanctum/",
        "/passport/",
        "/fortify/",
        "/scout/",
        "/telescope/",
        "/pint/",
        "/jetstream/",
        "/breeze/",
        "/reverb/",
    ]
    medium = [
        "/livewire",
        "/inertia",
        "/spark",
        "/vapor",
        "/lumen",
        "/sail",
        "/homestead",
    ]

    if any(k in p for k in high):
        return 1.5
    if any(k in p for k in medium):
        return 1.2
    return 1.0


def should_skip_dir(rel_dir: str) -> bool:
    rel = rel_dir.replace("\\", "/")
    if rel.startswith("./"):
        rel = rel[2:]
    parts = rel.split("/") if rel else []
    for part in parts:
        if part in DEFAULT_EXCLUDE_DIRS:
            return True
    # match path-like excludes such as bootstrap/cache
    for excl in DEFAULT_EXCLUDE_DIRS:
        if "/" in excl and excl in rel:
            return True
    return False


def collect_candidates(data_dir: str, extensions: set[str], max_chars: int, min_weight: float = 0.0, elite_only: bool = False) -> list[dict]:
    candidates = []

    for root, dirs, files in os.walk(data_dir):
        rel_root = os.path.relpath(root, data_dir)
        if should_skip_dir(rel_root):
            dirs[:] = []
            continue

        dirs[:] = [d for d in dirs if not should_skip_dir(os.path.join(rel_root, d))]

        for file_name in files:
            ext = Path(file_name).suffix.lower()
            if ext not in extensions:
                continue

            abs_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(abs_path, data_dir).replace("\\", "/")
            rel_with_slash = f"/{rel_path.lower()}"

            if elite_only and not any(h in rel_with_slash for h in ELITE_REPO_HINTS):
                continue

            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read(max_chars + 1)
                if not text.strip():
                    continue
            except Exception:
                continue

            normalized = normalize_text(text[:max_chars])
            if len(normalized) < 40:
                continue

            digest = hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()
            bucket = file_bucket(rel_path)
            weight = repo_weight(f"/{rel_path}")
            if weight < min_weight:
                continue

            candidates.append(
                {
                    "path": rel_path,
                    "bucket": bucket,
                    "weight": weight,
                    "digest": digest,
                    "chars": len(normalized),
                }
            )

    return candidates


def deduplicate(candidates: list[dict]) -> list[dict]:
    seen = {}
    for item in candidates:
        digest = item["digest"]
        current = seen.get(digest)
        if current is None or item["weight"] > current["weight"]:
            seen[digest] = item
    return list(seen.values())


def weighted_sample(items: list[dict], k: int, rng: random.Random) -> list[dict]:
    if k <= 0 or not items:
        return []
    if k >= len(items):
        return items[:]

    picked = []
    pool = items[:]
    for _ in range(k):
        weights = [max(0.01, x["weight"]) for x in pool]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        idx = 0
        for i, w in enumerate(weights):
            acc += w
            if acc >= r:
                idx = i
                break
        picked.append(pool.pop(idx))
        if not pool:
            break
    return picked


def curate(candidates: list[dict], target_files: int, mix_code: float, mix_docs: float, seed: int):
    rng = random.Random(seed)

    code = [x for x in candidates if x["bucket"] == "code"]
    docs = [x for x in candidates if x["bucket"] == "docs"]
    tests = [x for x in candidates if x["bucket"] == "tests"]
    other = [x for x in candidates if x["bucket"] == "other"]

    n_code = int(target_files * mix_code)
    n_docs = int(target_files * mix_docs)
    n_tests = target_files - n_code - n_docs

    selected = []
    selected.extend(weighted_sample(code, min(n_code, len(code)), rng))
    selected.extend(weighted_sample(docs, min(n_docs, len(docs)), rng))

    remaining_tests = max(0, n_tests)
    tests_and_other = tests + other
    selected.extend(weighted_sample(tests_and_other, min(remaining_tests, len(tests_and_other)), rng))

    # completa caso falte por falta de bucket
    if len(selected) < target_files:
        picked_paths = {x["path"] for x in selected}
        leftover = [x for x in candidates if x["path"] not in picked_paths]
        selected.extend(weighted_sample(leftover, min(target_files - len(selected), len(leftover)), rng))

    rng.shuffle(selected)
    return selected


def summarize(items: list[dict]) -> dict:
    by_bucket = {"code": 0, "docs": 0, "tests": 0, "other": 0}
    total_chars = 0
    for item in items:
        by_bucket[item["bucket"]] = by_bucket.get(item["bucket"], 0) + 1
        total_chars += item.get("chars", 0)
    return {
        "total_files": len(items),
        "by_bucket": by_bucket,
        "total_chars": total_chars,
    }


def main():
    parser = argparse.ArgumentParser(description="Curadoria automática de dataset Laravel")
    parser.add_argument("--data-dir", default="data", help="Diretório raiz do corpus")
    parser.add_argument("--manifest", default="curated_manifest.txt", help="Arquivo de saída com paths relativos")
    parser.add_argument("--stats", default="curated_stats.json", help="Arquivo JSON com estatísticas")
    parser.add_argument("--target-files", type=int, default=12000, help="Quantidade alvo de arquivos")
    parser.add_argument("--mix-code", type=float, default=0.60, help="Proporção de código")
    parser.add_argument("--mix-docs", type=float, default=0.25, help="Proporção de docs")
    parser.add_argument("--seed", type=int, default=1337, help="Seed aleatória")
    parser.add_argument("--max-chars", type=int, default=120000, help="Máximo de caracteres lidos por arquivo")
    parser.add_argument("--min-weight", type=float, default=0.0, help="Peso mínimo de repositório")
    parser.add_argument("--elite-only", action="store_true", help="Mantém apenas repositórios elite do ecossistema Laravel")
    args = parser.parse_args()

    if args.mix_code < 0 or args.mix_docs < 0 or args.mix_code + args.mix_docs > 1:
        raise ValueError("Mistura inválida. Garanta mix_code + mix_docs <= 1")

    print("🔎 Coletando candidatos...")
    candidates = collect_candidates(
        args.data_dir,
        DEFAULT_EXTENSIONS,
        max_chars=args.max_chars,
        min_weight=args.min_weight,
        elite_only=args.elite_only,
    )
    print(f"📦 Candidatos brutos: {len(candidates)}")

    deduped = deduplicate(candidates)
    print(f"🧹 Após deduplicação: {len(deduped)}")

    curated = curate(
        deduped,
        target_files=args.target_files,
        mix_code=args.mix_code,
        mix_docs=args.mix_docs,
        seed=args.seed,
    )

    with open(args.manifest, "w", encoding="utf-8") as f:
        for item in curated:
            f.write(item["path"] + "\n")

    stats = {
        "input_summary": summarize(deduped),
        "curated_summary": summarize(curated),
        "config": {
            "data_dir": args.data_dir,
            "target_files": args.target_files,
            "mix_code": args.mix_code,
            "mix_docs": args.mix_docs,
            "mix_tests_and_other": 1.0 - args.mix_code - args.mix_docs,
            "seed": args.seed,
            "max_chars": args.max_chars,
            "min_weight": args.min_weight,
            "elite_only": args.elite_only,
        },
    }

    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"✅ Manifesto salvo em: {args.manifest}")
    print(f"📊 Stats salvas em: {args.stats}")
    print(f"🎯 Arquivos selecionados: {len(curated)}")


if __name__ == "__main__":
    main()
