import argparse
import json
import os
import re
from datetime import datetime
import torch
from tokenizers import ByteLevelBPETokenizer

from marco4_model import EdithLaravelModel, device, block_size


def load_tokenizer():
    vocab_path = "edith-laravel-grand/vocab.json"
    merges_path = "edith-laravel-grand/merges.txt"
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        raise FileNotFoundError("Tokenizer não encontrado em edith-laravel-grand/. Rode o treino/tokenizer antes.")
    return ByteLevelBPETokenizer(vocab_path, merges_path)


def load_model(vocab_size, model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    model = EdithLaravelModel(vocab_size).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def token_ids_for_whitespace(tokenizer):
    ids = set()
    for s in ["\n", "\n\n", "\r", "\t", " "]:
        try:
            ids.update(tokenizer.encode(s).ids)
        except Exception:
            continue
    return ids


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=220, temperature=0.8, top_k=40, repetition_penalty=1.10, anti_blank=True):
    input_ids = tokenizer.encode(prompt).ids
    if len(input_ids) == 0:
        raise ValueError("Prompt vazio após tokenização.")

    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    whitespace_ids = token_ids_for_whitespace(tokenizer)
    newline_ids = set(tokenizer.encode("\n").ids)
    consecutive_newlines = 0

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        if repetition_penalty > 1.0:
            seen = set(idx[0].tolist())
            for token_id in seen:
                value = logits[0, token_id]
                if value < 0:
                    logits[0, token_id] = value * repetition_penalty
                else:
                    logits[0, token_id] = value / repetition_penalty

        if anti_blank and consecutive_newlines >= 2 and newline_ids:
            for token_id in newline_ids:
                logits[0, token_id] -= 6.0

        if top_k > 0:
            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_values[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

        next_id = int(next_token.item())
        if next_id in newline_ids:
            consecutive_newlines += 1
        elif next_id in whitespace_ids:
            pass
        else:
            consecutive_newlines = 0

    output_ids = idx[0].tolist()
    return tokenizer.decode(output_ids)


def score_laravel_store(text):
    t = text.lower()

    checks = {
        "assinatura_store": "function store" in t,
        "request_tipado": "request $request" in t,
        "validacao": ("$request->validate" in t) or ("validator::make" in t),
        "persistencia": any(x in t for x in ["::create(", "->save()", "->create("]),
        "retorno_http": any(x in t for x in ["return redirect", "return response", "return back(", "return to_route("]),
        "nao_expoe_senha": "password" not in t or "bcrypt" in t or "hash::make" in t,
    }

    score = sum(1 for ok in checks.values() if ok)
    return score, checks


def score_generic_laravel(text):
    t = text.lower()
    checks = {
        "tem_php": "<?php" in t or "function " in t,
        "usa_laravel": any(x in t for x in ["request", "model", "controller", "route", "eloquent", "validator", "middleware"]),
        "estrutura_basica": ("{" in text and "}" in text),
        "retorno_resposta": any(x in t for x in ["return", "response", "redirect", "to_route"]),
        "baixo_ruido": len(re.findall(r"\n{3,}", text)) <= 2,
        "tamanho_util": len(text.strip()) >= 120,
    }
    score = sum(1 for ok in checks.values() if ok)
    return score, checks


def score_auth_laravel(text):
    t = text.lower()
    checks = {
        "usa_auth": any(x in t for x in ["auth", "sanctum", "gate", "policy", "can(", "authorize("]),
        "middleware_guard": "middleware" in t,
        "retorno": "return" in t,
        "estrutura": ("{" in text and "}" in text),
        "baixo_ruido": len(re.findall(r"\n{3,}", text)) <= 2,
        "tamanho_util": len(text.strip()) >= 120,
    }
    return sum(1 for ok in checks.values() if ok), checks


def score_api_laravel(text):
    t = text.lower()
    checks = {
        "resource_json": any(x in t for x in ["json", "resource", "response()->json", "apiresource"]),
        "status_code": any(x in t for x in ["201", "200", "422", "status("]),
        "validacao": any(x in t for x in ["validate", "rules", "validator"]),
        "retorno": "return" in t,
        "baixo_ruido": len(re.findall(r"\n{3,}", text)) <= 2,
        "tamanho_util": len(text.strip()) >= 120,
    }
    return sum(1 for ok in checks.values() if ok), checks


def score_tests_laravel(text):
    t = text.lower()
    checks = {
        "framework_teste": any(x in t for x in ["test", "assert", "it(", "expect(", "phpunit", "pest"]),
        "assert_http": any(x in t for x in ["assertstatus", "assertok", "assertredirect", "assertjson"]),
        "arrange_act_assert": ("->" in text) or ("given" in t and "when" in t),
        "estrutura": ("{" in text and "}" in text),
        "baixo_ruido": len(re.findall(r"\n{3,}", text)) <= 2,
        "tamanho_util": len(text.strip()) >= 120,
    }
    return sum(1 for ok in checks.values() if ok), checks


def score_queue_events_laravel(text):
    t = text.lower()
    checks = {
        "fila_ou_evento": any(x in t for x in ["dispatch", "queue", "job", "event", "listener", "handle("]),
        "assincrono": any(x in t for x in ["implements shouldqueue", "onqueue", "delay", "bus::"]),
        "estrutura": ("{" in text and "}" in text),
        "retorno_ou_handle": any(x in t for x in ["return", "handle("]),
        "baixo_ruido": len(re.findall(r"\n{3,}", text)) <= 2,
        "tamanho_util": len(text.strip()) >= 120,
    }
    return sum(1 for ok in checks.values() if ok), checks


def choose_scoring(prompt, generated, category="generic"):
    p = prompt.lower()
    c = (category or "generic").lower()

    if c == "store" or "function store" in p:
        return score_laravel_store(generated), "Laravel Store"
    if c in {"auth", "policy", "security"}:
        return score_auth_laravel(generated), "Laravel Auth"
    if c in {"api", "resource"}:
        return score_api_laravel(generated), "Laravel API"
    if c in {"test", "tests", "testing"}:
        return score_tests_laravel(generated), "Laravel Tests"
    if c in {"queue", "event", "events"}:
        return score_queue_events_laravel(generated), "Laravel Queue/Event"
    if "function store" in p:
        return score_laravel_store(generated), "Laravel Store"
    return score_generic_laravel(generated), "Laravel Generic"


def load_prompts(prompt_file):
    prompts = []
    if not prompt_file:
        return prompts
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Arquivo de prompts não encontrado: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "|" in line:
                    category, prompt = line.split("|", 1)
                    prompts.append({"category": category.strip(), "prompt": prompt.strip()})
                else:
                    prompts.append({"category": "generic", "prompt": line})
    return prompts


def print_report(sample_idx, generated, prompt, category="generic"):
    (score, checks), metric_name = choose_scoring(prompt, generated, category=category)
    rendered = re.sub(r"\n{4,}", "\n\n[...linhas vazias suprimidas...]\n\n", generated)

    print("\n" + "=" * 80)
    print(f"🧪 AMOSTRA {sample_idx}")
    print("=" * 80)
    print(rendered)
    print("-" * 80)
    print(f"📊 Score {metric_name}: {score}/6")
    for item, ok in checks.items():
        print(f"  - {item}: {'✅' if ok else '❌'}")

    prompt_lower = prompt.lower()
    prompt_pos = generated.lower().find(prompt_lower[: min(len(prompt_lower), 24)])
    if prompt_pos == -1:
        print("⚠️ Observação: saída pode ter se desviado do prompt inicial.")


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Batismo da Edith: geração + avaliação básica de saída Laravel")
    parser.add_argument("--model", default="edith_laravel_senior.pt", help="Caminho do modelo .pt")
    parser.add_argument("--prompt", default="public function store(Request $request) {", help="Prompt inicial")
    parser.add_argument("--samples", type=int, default=3, help="Número de amostras")
    parser.add_argument("--tokens", type=int, default=220, help="Tokens gerados por amostra")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperatura")
    parser.add_argument("--top-k", type=int, default=40, dest="top_k", help="Top-k sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.10, dest="repetition_penalty", help="Penaliza repetição de tokens")
    parser.add_argument("--no-anti-blank", action="store_true", help="Desativa proteção contra excesso de linhas em branco")
    parser.add_argument("--prompt-file", default="", help="Arquivo de prompts (1 por linha) para benchmark")
    parser.add_argument("--json-report", default="", help="Salva relatório completo desta execução em JSON")
    parser.add_argument("--save-history", action="store_true", help="Adiciona esta execução no histórico JSONL")
    parser.add_argument("--history-file", default="benchmark_history.jsonl", help="Arquivo JSONL de histórico")
    parser.add_argument("--run-name", default="", help="Nome opcional para identificar a rodada")
    args = parser.parse_args()

    print(f"🚀 Dispositivo: {device}")
    print(f"🧠 Modelo: {args.model}")

    tokenizer = load_tokenizer()
    model = load_model(tokenizer.get_vocab_size(), args.model)

    prompts = [{"category": "store", "prompt": args.prompt}]
    file_prompts = load_prompts(args.prompt_file)
    if file_prompts:
        prompts = file_prompts

    global_scores = []
    scores_by_category = {}
    prompt_results = []

    for prompt_idx, prompt_cfg in enumerate(prompts, start=1):
        prompt = prompt_cfg["prompt"]
        category = prompt_cfg.get("category", "generic")
        if len(prompts) > 1:
            print("\n" + "#" * 80)
            print(f"🎯 PROMPT {prompt_idx}/{len(prompts)} | Categoria: {category}")
            print(prompt)
            print("#" * 80)

        for i in range(1, args.samples + 1):
            out = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                anti_blank=not args.no_anti_blank,
            )
            print_report(i, out, prompt, category=category)
            (score, _), _metric_name = choose_scoring(prompt, out, category=category)
            global_scores.append(score)
            scores_by_category.setdefault(category, []).append(score)
            prompt_results.append(
                {
                    "category": category,
                    "prompt": prompt,
                    "sample": i,
                    "score": score,
                }
            )

    avg = None
    if global_scores:
        avg = sum(global_scores) / len(global_scores)
        print("\n" + "=" * 80)
        print(f"🏁 SCORE MÉDIO GLOBAL: {avg:.2f}/6 em {len(global_scores)} amostras")
        print("=" * 80)

    category_avg = {}
    if scores_by_category:
        print("\n" + "=" * 80)
        print("📚 SCORE MÉDIO POR CATEGORIA")
        print("=" * 80)
        for category, values in scores_by_category.items():
            avg_cat = sum(values) / len(values)
            category_avg[category] = avg_cat
            print(f"- {category}: {avg_cat:.2f}/6 em {len(values)} amostras")

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": args.run_name or None,
        "model": args.model,
        "device": str(device),
        "params": {
            "samples": args.samples,
            "tokens": args.tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "anti_blank": not args.no_anti_blank,
            "prompt_file": args.prompt_file or None,
        },
        "global": {
            "avg_score": avg,
            "count": len(global_scores),
        },
        "category_avg": category_avg,
        "prompt_results": prompt_results,
    }

    if args.json_report:
        with open(args.json_report, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"🧾 Relatório JSON salvo em: {args.json_report}")

    if args.save_history:
        append_jsonl(args.history_file, payload)
        print(f"📘 Histórico atualizado em: {args.history_file}")


if __name__ == "__main__":
    main()
