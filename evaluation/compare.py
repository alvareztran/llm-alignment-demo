import os
import sys
import argparse
from datetime import datetime
from html import escape
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.generate import generate_response
from models.load_model import MODEL_NAME, load_tokenizer
from ppo.reward import compute_reward

MODEL_PATHS = {
    "base": MODEL_NAME,
    "dpo": "./outputs/dpo_model",
    "ppo": "./outputs/ppo_model",
}

MODEL_DISPLAY = {
    "base": {
        "title": "Base Model",
        "subtitle": "Original model before additional alignment",
        "accent": "#2563eb",
    },
    "dpo": {
        "title": "DPO Model",
        "subtitle": "Trained directly from chosen / rejected pairs",
        "accent": "#16a34a",
    },
    "ppo": {
        "title": "PPO Model",
        "subtitle": "Optimized with a reward signal",
        "accent": "#dc2626",
    },
}


def model_exists(path: str) -> bool:
    return os.path.isdir(path) or path == MODEL_NAME


def load_model_for_compare(path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    model.to(device)
    return model


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def compare_models(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    tokenizer = load_tokenizer()
    results = {"prompt": prompt}

    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(prompt)

    for name, path in MODEL_PATHS.items():
        if not model_exists(path):
            output = f"[Skipped: model not found at {path}]"
        else:
            model = load_model_for_compare(path)
            output = generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results[name] = output

        print("\n" + "=" * 80)
        print(f"{name.upper()} MODEL")
        print("=" * 80)
        print(output)

    return results


def save_result(result, path="./outputs/compare_result.txt"):
    ensure_parent_dir(path)

    with open(path, "w", encoding="utf-8") as f:
        f.write("===== PROMPT =====\n")
        f.write(result["prompt"] + "\n\n")

        f.write("===== BASE =====\n")
        f.write(f"Reward score: {compute_reward(result['prompt'], result['base']):.4f}\n")
        f.write(result["base"] + "\n\n")

        f.write("===== DPO =====\n")
        f.write(f"Reward score: {compute_reward(result['prompt'], result['dpo']):.4f}\n")
        f.write(result["dpo"] + "\n\n")

        f.write("===== PPO =====\n")
        f.write(f"Reward score: {compute_reward(result['prompt'], result['ppo']):.4f}\n")
        f.write(result["ppo"] + "\n\n")

    print(f"\nSaved to {path}")


def response_stats(prompt: str, text: str):
    clean_text = text.strip()
    return {
        "chars": len(clean_text),
        "words": len(clean_text.split()),
        "reward_score": compute_reward(prompt, clean_text),
    }


def render_response_card(model_key: str, prompt: str, response: str) -> str:
    info = MODEL_DISPLAY[model_key]
    stats = response_stats(prompt, response)

    return f"""
        <article class="model-card" style="--accent: {info['accent']}">
            <header class="card-header">
                <div>
                    <h2>{escape(info['title'])}</h2>
                    <p>{escape(info['subtitle'])}</p>
                </div>
                <span class="model-badge">{escape(model_key.upper())}</span>
            </header>

            <div class="stats">
                <span>{stats['words']} words</span>
                <span>{stats['chars']} chars</span>
                <span>reward {stats['reward_score']:.2f}</span>
            </div>

            <pre class="response-text">{escape(response)}</pre>
        </article>
    """


def save_html_report(result, path="./outputs/compare_report.html"):
    ensure_parent_dir(path)

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cards = "\n".join(
        render_response_card(model_key, result["prompt"], result.get(model_key, ""))
        for model_key in MODEL_PATHS
    )

    html = f"""<!doctype html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Alignment Demo - Model Comparison</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f6f7fb;
            --panel: #ffffff;
            --text: #172033;
            --muted: #667085;
            --line: #dde3ee;
            --shadow: 0 14px 34px rgba(15, 23, 42, 0.10);
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.55;
        }}

        .page {{
            width: min(1320px, calc(100% - 32px));
            margin: 0 auto;
            padding: 32px 0 44px;
        }}

        .topbar {{
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 18px;
        }}

        h1 {{
            margin: 0;
            font-size: 30px;
            line-height: 1.15;
            letter-spacing: 0;
        }}

        .meta {{
            margin: 8px 0 0;
            color: var(--muted);
            font-size: 14px;
        }}

        .tag {{
            display: inline-flex;
            align-items: center;
            min-height: 32px;
            padding: 6px 10px;
            border: 1px solid var(--line);
            border-radius: 6px;
            background: var(--panel);
            color: var(--muted);
            font-size: 13px;
            white-space: nowrap;
        }}

        .prompt-panel {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: var(--shadow);
            padding: 18px;
            margin-bottom: 18px;
        }}

        .prompt-panel h2 {{
            margin: 0 0 10px;
            font-size: 16px;
        }}

        .prompt-text,
        .response-text {{
            margin: 0;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            font-family: Consolas, "Courier New", monospace;
        }}

        .prompt-text {{
            color: #243045;
            font-size: 14px;
        }}

        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
            align-items: stretch;
        }}

        .model-card {{
            display: flex;
            min-height: 520px;
            flex-direction: column;
            background: var(--panel);
            border: 1px solid var(--line);
            border-top: 5px solid var(--accent);
            border-radius: 8px;
            box-shadow: var(--shadow);
            overflow: hidden;
        }}

        .card-header {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            padding: 16px 16px 12px;
            border-bottom: 1px solid var(--line);
        }}

        .card-header h2 {{
            margin: 0;
            font-size: 18px;
            line-height: 1.25;
        }}

        .card-header p {{
            margin: 4px 0 0;
            color: var(--muted);
            font-size: 13px;
        }}

        .model-badge {{
            flex: 0 0 auto;
            padding: 4px 8px;
            border-radius: 6px;
            background: color-mix(in srgb, var(--accent) 12%, white);
            color: var(--accent);
            font-size: 12px;
            font-weight: 700;
        }}

        .stats {{
            display: flex;
            gap: 8px;
            padding: 12px 16px;
            border-bottom: 1px solid var(--line);
            color: var(--muted);
            font-size: 13px;
        }}

        .stats span {{
            padding: 4px 8px;
            border: 1px solid var(--line);
            border-radius: 6px;
            background: #f9fafc;
        }}

        .response-text {{
            flex: 1;
            padding: 16px;
            color: #1d2939;
            font-size: 14px;
            overflow: auto;
        }}

        @media (max-width: 980px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}

            .model-card {{
                min-height: auto;
            }}

            .topbar {{
                align-items: flex-start;
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <main class="page">
        <section class="topbar">
            <div>
                <h1>Alignment Demo: Model Comparison</h1>
                <p class="meta">Generated at {escape(created_at)}</p>
            </div>
            <div class="tag">Base vs DPO vs PPO</div>
        </section>

        <section class="prompt-panel">
            <h2>Prompt</h2>
            <pre class="prompt-text">{escape(result['prompt'])}</pre>
        </section>

        <section class="comparison-grid">
            {cards}
        </section>
    </main>
</body>
</html>
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved HTML report to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Base, DPO, and PPO models with a custom prompt."
    )
    parser.add_argument(
        "prompt_text",
        nargs="*",
        help="Prompt text. Example: Human: Explain AI. Assistant:",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help="Prompt text. Overrides positional prompt_text.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./outputs/compare_result.txt",
        help="Path to save comparison result.",
    )
    parser.add_argument(
        "--html-output",
        default="./outputs/compare_report.html",
        help="Path to save HTML comparison report.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Do not save the HTML comparison report.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling value.",
    )
    return parser.parse_args()


def get_prompt_from_args(args) -> str:
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_text:
        prompt = " ".join(args.prompt_text)
    else:
        print("Enter prompt for comparison.")
        print("Tip: include 'Human:' and 'Assistant:' if you want chat-style input.")
        prompt = input("Prompt: ")

    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    if "Assistant:" not in prompt:
        prompt = f"Human: {prompt}\nAssistant:"

    return prompt


if __name__ == "__main__":
    args = parse_args()
    user_prompt = get_prompt_from_args(args)

    result = compare_models(
        user_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    save_result(result, args.output)
    if not args.no_html:
        save_html_report(result, args.html_output)
