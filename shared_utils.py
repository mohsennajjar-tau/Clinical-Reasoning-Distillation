#!/usr/bin/env python3
"""
shared_utils.py — Local GPU version (RTX 4090)

SETUP INSTRUCTIONS:
===================

1. Create your project folder:
   mkdir -p ~/kd_project/data
   mkdir -p ~/kd_project/runs

2. Copy your Google Drive data files to ~/kd_project/data/:
   - clinical_pharm_prompts_10000.jsonl
   - clinical_pharm_teacher_train_6000.jsonl
   - clinical_pharm_splits_random_8k_1k_1k_seed42.json
   - student_inference_200.json  (if you have it, for baseline comparison)
   - Any judge_eval__*__gemini.jsonl files you want to compare against

3. Create a conda environment:
   conda create -n kd python=3.11 -y
   conda activate kd

4. Install dependencies:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install transformers peft accelerate bitsandbytes sentencepiece
   pip install google-genai   # for judge evaluation

5. Place this file at ~/kd_project/shared_utils.py

6. Set your Gemini API key as an environment variable:
   export GEMINI_API_KEY="your-new-key-here"

   Or create a file ~/kd_project/.env with:
   GEMINI_API_KEY=your-new-key-here

7. Run experiments:
   cd ~/kd_project
   python M1_Additive_Multi_Loss.py
   python M2_Anti_Curriculum.py
   python M3_Juggler_Dynamic_Weights.py
   python M4_Token_Confidence_Routing.py
   python M_Eval_Inference_Judge.py
"""

import os, json, math, re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ----------------------------
# PATHS — EDIT THESE if your layout differs
# ----------------------------
# Auto-detect: use env var if set, otherwise data/ next to this file
BASE_DIR = os.environ.get("KD_BASE_DIR", "data")
RUNS_DIR = os.environ.get("KD_RUNS_DIR", "runs")

PROMPTS_PATH = os.path.join(BASE_DIR, "clinical_pharm_prompts_10000.jsonl")
TEACHER_6K_PATH = os.path.join(BASE_DIR, "clinical_pharm_teacher_train_6000.jsonl")
SPLIT_PATH = os.path.join(BASE_DIR, "clinical_pharm_splits_random_8k_1k_1k_seed42.json")

# ----------------------------
# TRAINING CONFIG
# ----------------------------
MAX_SEQ_LEN = 2048
EPOCHS = 2
LR = 2e-4
SEED = 42

# RTX 4090 has 24GB VRAM — can fit larger batches than Colab T4
# 4-bit Qwen 7B ≈ 5GB, + gradients/optimizer ≈ 12-14GB total
# So batch_size=2 with grad_acc=16 gives effective batch=32 (same as before)
BATCH_SIZE = 8        # ← doubled from Colab (1→2)
GRAD_ACC = 4         # ← halved from Colab (32→16) to keep effective batch = 32

# WSFT weights
W_FORMAT = 1.0
W_DECISION = 2.0
W_EXPL = 2.0

# Confidence clipping
ALPHA_MIN = 0.25
ALPHA_MAX = 3.0

# Entropy matching
LAMBDA = 0.05

STUDENTS = {
    "qwen25_1p5b_base": {"path": "Qwen/Qwen2.5-1.5B", "is_instruct": False},
    "qwen25_3b_base":   {"path": "Qwen/Qwen2.5-3B",   "is_instruct": False},
    "qwen25_7b_base":   {"path": "Qwen/Qwen2.5-7B",   "is_instruct": False},
}

# ----------------------------
# Gemini API key (for judge evaluation only — NOT needed for training)
# ----------------------------
def get_gemini_api_key() -> str:
    """
    Reads API key from (in order):
    1. GEMINI_API_KEY env var
    2. ~/kd_project/.env file
    3. Prompts user interactively
    """
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    env_file = os.path.join(os.path.dirname(BASE_DIR), ".env")
    if os.path.exists(env_file):
        for line in open(env_file):
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    return input("Enter your Gemini API key: ").strip()


# ----------------------------
# IO helpers
# ----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            out.append(json.loads(l))
    return out


def load_data():
    """Load prompts dict and teacher_rows list."""
    assert os.path.exists(PROMPTS_PATH), f"Not found: {PROMPTS_PATH}"
    assert os.path.exists(TEACHER_6K_PATH), f"Not found: {TEACHER_6K_PATH}"

    prompts = {r["id"]: r["prompt"] for r in load_jsonl(PROMPTS_PATH)}
    teacher_rows = [
        r for r in load_jsonl(TEACHER_6K_PATH)
        if (r.get("status") == "ok" and r.get("teacher_text") and r.get("split") == "train")
    ]
    print(f"✅ Loaded {len(teacher_rows)} teacher samples from: {TEACHER_6K_PATH}")
    return prompts, teacher_rows


# ----------------------------
# Section span finder
# ----------------------------
_DECISION_PATTERNS = [
    r'"decision"\s*:\s*', r"'decision'\s*:\s*", r"\bDecision\s*:\s*", r"\bDECISION\s*:\s*"
]
_EXPL_PATTERNS = [
    r'"explanation"\s*:\s*', r"'explanation'\s*:\s*", r"\bExplanation\s*:\s*", r"\bEXPLANATION\s*:\s*",
    r'"reasoning"\s*:\s*', r"'reasoning'\s*:\s*", r"\bReasoning\s*:\s*", r"\bREASONING\s*:\s*"
]


def _find_span_by_markers(text: str, start_markers: List[str], end_markers: List[str]) -> Tuple[int, int]:
    start_end = -1
    for pat in start_markers:
        m = re.search(pat, text)
        if m:
            start_end = m.end()
            break
    if start_end == -1:
        return (-1, -1)
    end_pos = len(text)
    for pat in end_markers:
        m2 = re.search(pat, text[start_end:])
        if m2:
            end_pos = min(end_pos, start_end + m2.start())
    return (start_end, end_pos)


def get_section_spans(answer_text: str) -> Dict[str, List[Tuple[int, int]]]:
    spans = {"decision": [], "explanation": []}
    d_span = _find_span_by_markers(answer_text, _DECISION_PATTERNS, _EXPL_PATTERNS)
    if d_span != (-1, -1) and d_span[0] < d_span[1]:
        spans["decision"].append(d_span)
    e_span = _find_span_by_markers(answer_text, _EXPL_PATTERNS, _DECISION_PATTERNS)
    if e_span != (-1, -1) and e_span[0] < e_span[1]:
        spans["explanation"].append(e_span)
    return spans


def in_any_span(token_s: int, token_e: int, span_list: List[Tuple[int, int]]) -> bool:
    for (s, e) in span_list:
        if token_e > s and token_s < e:
            return True
    return False


# ----------------------------
# Confidence computation
# ----------------------------
def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def compute_mean_margin(logprobs_steps: List[Dict[str, Any]]) -> Optional[float]:
    if not logprobs_steps:
        return None
    margins = []
    for step in logprobs_steps:
        ch = step.get("chosen", {})
        topk = step.get("topk", [])
        ch_lp = ch.get("logprob", None)
        ch_tok = ch.get("token", None)
        if ch_lp is None or not topk:
            continue
        alt_lps = []
        for tc in topk:
            if ch_tok is not None and tc.get("token", None) == ch_tok:
                continue
            lp = tc.get("logprob", None)
            if lp is not None:
                alt_lps.append(lp)
        if not alt_lps:
            continue
        margins.append(float(ch_lp) - float(max(alt_lps)))
    if not margins:
        return None
    return float(sum(margins) / len(margins))


def compute_confidence(row: Dict[str, Any]) -> float:
    mm = compute_mean_margin(row.get("logprobs_steps", None) or [])
    if mm is None:
        return float("nan")
    return _sigmoid(mm)


def compute_mean_confidence(teacher_rows: List[Dict[str, Any]]) -> float:
    confs = []
    for r in teacher_rows:
        c = compute_confidence(r)
        if not (c != c):
            confs.append(c)
    return float(sum(confs) / max(1, len(confs)))


def compute_alpha(row: Dict[str, Any], mean_c: float) -> float:
    c = compute_confidence(row)
    if (c != c) or mean_c <= 0:
        return 1.0
    a = c / mean_c
    return float(max(ALPHA_MIN, min(ALPHA_MAX, a)))


# ----------------------------
# Entropy helpers
# ----------------------------
def entropy_from_logprobs(logps_1d: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logps_1d, dim=0)
    return -(probs * torch.log(probs + 1e-9)).sum()


def step_char_spans_from_chosen_tokens(logprobs_steps):
    spans = []
    buf = ""
    for st in logprobs_steps or []:
        tok = (st.get("chosen") or {}).get("token", "")
        start = len(buf)
        buf += tok
        end = len(buf)
        spans.append((start, end))
    return spans


def teacher_section_entropy_mean(row, section_span):
    steps = row.get("logprobs_steps", []) or []
    if not steps or not section_span:
        return torch.tensor(0.0)
    spans = step_char_spans_from_chosen_tokens(steps)
    s0, s1 = section_span
    ent = []
    for i, st in enumerate(steps):
        a, b = spans[i]
        if not (b <= s0 or a >= s1):
            topk = st.get("topk", [])
            if topk:
                lp = torch.tensor([t["logprob"] for t in topk], dtype=torch.float32)
                ent.append(entropy_from_logprobs(lp))
    return torch.stack(ent).mean() if ent else torch.tensor(0.0)


def find_decision_span_chars(teacher_text: str):
    for p in [r"\bDecision\s*:\s*", r'"decision"\s*:\s*']:
        m = re.search(p, teacher_text)
        if m:
            for ep in [r"\bExplanation\s*:\s*", r'"explanation"\s*:\s*']:
                m2 = re.search(ep, teacher_text[m.end():])
                if m2:
                    return (m.end(), m.end() + m2.start())
            return (m.end(), len(teacher_text))
    return None


def find_expl_span_chars(teacher_text: str):
    for p in [r"\bExplanation\s*:\s*", r'"explanation"\s*:\s*']:
        m = re.search(p, teacher_text)
        if m:
            return (m.end(), len(teacher_text))
    return None


# ----------------------------
# Tokenization helpers
# ----------------------------
def build_prompt_text(tokenizer, prompt: str, is_instruct: bool) -> str:
    if is_instruct and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
    return prompt


def tokenize_and_mask(tokenizer, prompt_text: str, answer: str, max_len: int = MAX_SEQ_LEN):
    full_text = prompt_text + answer
    tok = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
    )
    input_ids = tok["input_ids"]
    offsets = tok["offset_mapping"]
    answer_start = len(prompt_text)
    labels = [-100] * len(input_ids)
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        if s >= answer_start:
            labels[i] = input_ids[i]
    return input_ids, offsets, labels, answer_start


# ----------------------------
# Flexible Collator
# ----------------------------
class FlexibleCollator:
    def __init__(self, tokenizer, extra_1d_fields=None, extra_scalar_fields=None):
        self.tokenizer = tokenizer
        self.extra_1d_fields = extra_1d_fields or []
        self.extra_scalar_fields = extra_scalar_fields or []

    def __call__(self, features):
        padded = self.tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            return_tensors="pt",
        )
        max_len = padded["input_ids"].shape[1]

        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        for i, f in enumerate(features):
            L = f["labels"].shape[0]
            labels[i, :L] = f["labels"]
        padded["labels"] = labels

        for key in self.extra_1d_fields:
            if key not in features[0]:
                continue
            dtype = features[0][key].dtype
            if dtype == torch.bool:
                out = torch.zeros((len(features), max_len), dtype=torch.long)
                for i, f in enumerate(features):
                    L = f[key].shape[0]
                    out[i, :L] = f[key].to(torch.long)
                padded[key] = out.bool()
            else:
                pad_val = 0
                out = torch.full((len(features), max_len), pad_val, dtype=dtype)
                for i, f in enumerate(features):
                    L = f[key].shape[0]
                    out[i, :L] = f[key]
                padded[key] = out

        for key in self.extra_scalar_fields:
            if key not in features[0]:
                continue
            padded[key] = torch.stack([f[key] for f in features], dim=0)

        return padded


# ----------------------------
# Model loading
# ----------------------------
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_student(model_name: str, cfg: dict):
    """Load tokenizer + LoRA model. Uses 8-bit for 7B, bf16 for smaller."""
    print(f"  Loading {model_name} from {cfg['path']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "7b" in model_name:
        print("  Using 8-bit quantization (Windows stability)")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  ✅ {model_name}: {trainable:,} trainable / {total:,} total params")

    return tokenizer, model