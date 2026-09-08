"""Phase-1 eval report: E1 routing + E2 Look + E4 RAG. E3 live 2D is skipped without keys.

Usage (repo root):
  python scripts/eval_report.py --golden tests/eval/golden --out data/eval/report.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_look import score_pair  # noqa: E402
from app.api.main import app  # noqa: E402
from app.graphs.route import keyword_route  # noqa: E402
from app.jobs.budget import snapshot as budget_snapshot  # noqa: E402
from app.rag.retriever import ingest, query as rag_query  # noqa: E402
from app.renderers.image_2d import ART_STYLES, build_prompt  # noqa: E402
from app.renderers.photo_look import LOOKS, apply_look  # noqa: E402
from app.settings import settings  # noqa: E402

GOLDEN = ROOT / "tests" / "eval" / "golden"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_alias() -> dict:
    raw = yaml.safe_load((ROOT / "tests" / "eval" / "alias.yml").read_text(encoding="utf-8")) or {}
    return {str(k): [str(x) for x in (v or [])] for k, v in raw.items()}


def _ci_scenes() -> dict[str, np.ndarray]:
    portrait = np.zeros((160, 120, 3), dtype=np.uint8)
    portrait[:] = (88, 118, 168)
    cv2.circle(portrait, (60, 55), 32, (110, 150, 200), -1)
    cv2.rectangle(portrait, (35, 90), (85, 150), (70, 90, 140), -1)

    landscape = np.zeros((120, 180, 3), dtype=np.uint8)
    landscape[:50] = (210, 160, 80)
    landscape[50:] = (60, 140, 70)
    landscape[:, :40] = (40, 90, 40)

    night = np.zeros((120, 160, 3), dtype=np.uint8)
    night[:] = (18, 12, 22)
    night[20:50, 30:70] = (40, 20, 180)
    night[70:95, 90:130] = (160, 40, 90)
    return {"portrait": portrait, "landscape": landscape, "night": night}


def score_e1(dialogues: list[dict]) -> dict:
    alias = _load_alias()
    n = len(dialogues)
    intent_ok = 0
    style_ok = 0
    style_n = 0
    modality_ok = 0
    rows = []
    for item in dialogues:
        intent, style, _ = keyword_route(item["message"], bool(item.get("has_image")))
        expect_intent = item["intent"]
        expect_style = item.get("style_id")
        ok_i = intent == expect_intent
        intent_ok += int(ok_i)
        near = alias.get(expect_style or "", [])
        if expect_intent in {"look", "image2d"}:
            style_n += 1
            ok_s = style == expect_style or style in near
            style_ok += int(ok_s)
        else:
            ok_s = True
        look_as_2d = expect_intent == "look" and intent == "image2d"
        gen_as_look = expect_intent == "image2d" and intent == "look"
        ok_m = not look_as_2d and not gen_as_look
        modality_ok += int(ok_m)
        rows.append({"id": item["id"], "ok_intent": ok_i, "ok_style": ok_s, "pred": [intent, style]})
    return {
        "n": n,
        "intent_acc": intent_ok / n if n else 0,
        "style_top1": style_ok / style_n if style_n else 1.0,
        "modality_acc": modality_ok / n if n else 0,
        "rows": rows,
    }


def score_e2() -> dict:
    scenes = _ci_scenes()
    pairs = []
    t0 = time.perf_counter()
    for src in scenes.values():
        for sid in LOOKS:
            dst = apply_look(src, sid)
            pairs.append(score_pair(sid, src, dst))
    elapsed = time.perf_counter() - t0
    n = len(pairs)
    size_ok = sum(1 for p in pairs if p["size_ok"]) / n
    ssim_mean = sum(p["ssim"] for p in pairs) / n
    direction = sum(1 for p in pairs if p["direction_hit"]) / n
    p95_s = elapsed  # batch; per-image below
    per = elapsed / n
    return {
        "n": n,
        "size_ok": size_ok,
        "ssim_mean": round(ssim_mean, 4),
        "direction_hit": round(direction, 4),
        "seconds_total": round(elapsed, 4),
        "seconds_per_look": round(per, 4),
        "p95_look_s_proxy": round(per, 4),
        "pairs": pairs,
        "source": "opencv_ci_fixtures_not_portfolio",
    }


def score_e4(items: list[dict]) -> dict:
    ingest()
    hit = 0
    cite = 0
    rows = []
    for item in items:
        hits = rag_query(item["query"], k=3)
        ids = {h.get("style_id") for h in hits}
        ok = any(x in ids for x in item["expect_any"])
        hit += int(ok)
        cite += int(bool(ids))
        rows.append({"id": item["id"], "ok": ok, "ids": sorted(x for x in ids if x)})
    n = len(items)
    return {"n": n, "recall_at_3": hit / n if n else 0, "citation_rate": cite / n if n else 0, "rows": rows}


def score_e5() -> dict:
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    img[:, :32] = (30, 70, 140)
    img[:, 32:] = (160, 90, 40)
    raw = cv2.imencode(".jpg", img)[1].tobytes()
    c = TestClient(app)
    up = c.post("/v1/assets", files={"file": ("t.jpg", raw, "image/jpeg")})
    aid = up.json()["asset_id"]
    first = c.post("/v1/chat", json={"message": "做成胶片暖调", "asset_ids": [aid]})
    tid = first.json()["thread_id"]
    style = first.json()["style_id"]
    ok = 0
    patches = ["再暗一点", "再暖一点", "少颗粒", "再亮一点", "LUT淡一点"]
    for msg in patches:
        r = c.post("/v1/chat", json={"thread_id": tid, "message": msg})
        ok += int(r.status_code == 200 and r.json().get("style_id") == style)
    return {"patch_rounds": len(patches), "style_kept": ok, "style_id": style}


def gates(e1: dict, e2: dict, negatives: list[dict]) -> dict:
    c = TestClient(app)
    g2_ok = True
    for item in negatives:
        chat = c.post("/v1/chat", json={"message": item["message"]})
        if chat.status_code != 200:
            g2_ok = False
            continue
        if chat.json().get("job_id"):
            job = c.get(f"/v1/jobs/{chat.json()['job_id']}").json()
            if job.get("modality") in {"asset.3d", "video.clip", "video.long"}:
                g2_ok = False
        if item["expect"] == "no_paid_3d_video" and chat.json().get("job_id"):
            g2_ok = False
    r3d = c.post("/v1/jobs", json={"modality": "asset.3d", "style_id": "x", "prompt": "x"})
    rvid = c.post("/v1/jobs", json={"modality": "video.long", "style_id": "x", "prompt": "x"})
    g2_ok = g2_ok and r3d.status_code == 501 and rvid.status_code == 501

    snap = budget_snapshot()
    g1 = snap["project_budget_cny"] <= 500 and snap["project_spent_cny"] <= snap["project_budget_cny"]

    look = c.post(
        "/v1/jobs",
        json={
            "modality": "image.photo_look",
            "style_id": "film_portra",
            "input_assets": [{"asset_id": c.post("/v1/assets", files={"file": ("t.jpg", cv2.imencode(".jpg", np.zeros((16, 16, 3), np.uint8))[1].tobytes(), "image/jpeg")}).json()["asset_id"], "role": "source"}],
        },
    )
    body = look.json()
    g3 = (
        look.status_code == 200
        and body.get("status") == "succeeded"
        and body.get("trace", {}).get("style_id")
        and body.get("trace", {}).get("renderer")
        and body.get("trace", {}).get("lut")
    )

    banned = "官方吉卜力授权"
    prompt, negative = build_prompt("pastoral_anime", "改成田园动画风")
    blob = f"{prompt}\n{negative}".lower()
    g4 = banned not in prompt and "ghibli" not in blob and "吉卜力" not in prompt and "吉卜力" not in negative
    from app.rag import load_packs

    for pack in load_packs():
        name = str(pack.get("name") or "")
        positive = str(pack.get("positive") or "")
        if banned in name or banned in positive:
            g4 = False
        if "ghibli" in name.lower() or "ghibli" in positive.lower():
            g4 = False
        if "吉卜力" in name or "吉卜力" in positive:
            g4 = False
    return {"G1": g1, "G2": g2_ok, "G3": bool(g3), "G4": g4, "budget": snap}


def total_score(e1: dict, e2: dict, e4: dict, e5: dict, e3_run: bool) -> dict:
    e1n = min(1.0, (e1["intent_acc"] + e1["style_top1"] + e1["modality_acc"]) / 3)
    e2n = min(1.0, 0.4 * e2["size_ok"] + 0.4 * min(1.0, e2["ssim_mean"] / 0.75) + 0.2 * e2["direction_hit"])
    e3n = 0.0
    e4n = min(1.0, 0.7 * e4["recall_at_3"] + 0.3 * e4["citation_rate"])
    e5n = e5["style_kept"] / e5["patch_rounds"] if e5["patch_rounds"] else 0
    score = 25 * e1n + 25 * e2n + 25 * e3n + 10 * e4n + 15 * e5n
    return {
        "E1": round(e1n, 4),
        "E2": round(e2n, 4),
        "E3": round(e3n, 4),
        "E4": round(e4n, 4),
        "E5": round(e5n, 4),
        "total": round(score, 2),
        "e3_skipped": not e3_run,
        "pass_bar": score >= 70 and e1n >= 0.85 and e2n >= 0.75,
        "note": "E3=0 because live 2D portfolio is not in git; do not claim 10-style generation.",
    }


def render_md(payload: dict) -> str:
    g = payload["gates"]
    t = payload["totals"]
    lines = [
        f"# StyleForge phase-1 eval ({payload['date']})",
        "",
        f"- Python / model pin: `{settings.deepseek_model}`, 2D backend `{settings.image_2d_backend}`",
        f"- Project spend snapshot: {g['budget']}",
        f"- Gates: G1={g['G1']} G2={g['G2']} G3={g['G3']} G4={g['G4']}",
        f"- Totals: {t}",
        "",
        "## Honest limits",
        "- E3 live 2D and 12 copyright photos: skipped",
        "- Embedding: n-gram hash, not BGE-small-zh",
        "- Face cue: OpenCV Haar, not MediaPipe",
        "- MCTS/GRPO/VLM: off",
        "",
        "## E1 intent/style",
        f"- intent_acc={payload['E1']['intent_acc']:.3f} style_top1={payload['E1']['style_top1']:.3f} modality={payload['E1']['modality_acc']:.3f}",
        "",
        "## E2 Look (CI fixtures)",
        f"- n={payload['E2']['n']} size_ok={payload['E2']['size_ok']} ssim_mean={payload['E2']['ssim_mean']} direction={payload['E2']['direction_hit']}",
        "",
        "## E4 RAG",
        f"- recall@3={payload['E4']['recall_at_3']:.3f}",
        "",
        "## E5 patches",
        f"- {payload['E5']}",
    ]
    return "\n".join(lines) + "\n"


def run_report(golden: Path | None = None, out: Path | None = None) -> dict:
    golden = golden or GOLDEN
    dialogues = _load_json(golden / "dialogues.json")
    qa = _load_json(golden / "qa.json")
    negatives = _load_json(golden / "negatives.json")
    e1 = score_e1(dialogues)
    e2 = score_e2()
    e4 = score_e4(qa)
    e5 = score_e5()
    g = gates(e1, e2, negatives)
    totals = total_score(e1, e2, e4, e5, e3_run=False)
    if not (g["G1"] and g["G2"] and g["G3"] and g["G4"]):
        totals["pass_bar"] = False
        totals["gate_fail"] = True
    payload = {
        "date": datetime.now(timezone.utc).isoformat(),
        "E1": e1,
        "E2": e2,
        "E4": e4,
        "E5": e5,
        "gates": g,
        "totals": totals,
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_md(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "eval" / "report.md")
    args = parser.parse_args()
    payload = run_report(args.golden, args.out)
    print(json.dumps(payload["totals"], ensure_ascii=False, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
