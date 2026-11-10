"""Component 9: Gradio demo for the HF Space. Mounted on the same server as the API.

The FE talks to the BE only through the /v1 JSON contract. PII is masked here before sending.
Run locally: python -m apps.space.app   (serves API + UI on PORT, default 7860)
"""
from __future__ import annotations

import html
import os
import uuid

import gradio as gr
import httpx

from apps.api.main import app as api_app
from core.pii import mask

PORT = int(os.environ.get("PORT", "7860"))
BACKEND = os.environ.get("BACKEND_URL", f"http://127.0.0.1:{PORT}")
ARM_LABELS = {"A0": "A0 · base", "A1": "A1 · raw SFT", "A2": "A2 · curated SFT", "A3": "A3 · curated SFT + DPO"}
FOOTER = "Research demo. Not financial or legal advice. Confirm with your bank."

CSS = """
.card {border:1px solid #d0d4da;border-radius:10px;padding:12px;min-height:150px}
.card.abstained {background:#f1f3f4}
.badge {display:inline-block;border-radius:6px;padding:1px 8px;font-size:12px;margin-right:4px}
.badge.amber {background:#fde7b0;color:#5c3c00}
.badge.grey {background:#e0e3e7;color:#3c4043}
.badge.green {background:#d4f0dc;color:#0b5b24}
.cite {font-size:12px;color:#5f6368;margin-top:6px}
"""


def card(r: dict) -> str:
    arm = r.get("arm", "?")
    head = f"<b>{ARM_LABELS.get(arm, arm)}</b><br>"
    if r.get("status") == "error":
        return f"<div class='card'>{head}<span class='badge grey'>{html.escape(r['error']['code'])}</span></div>"
    if r.get("status") == "timeout":
        return f"<div class='card'>{head}<span class='badge grey'>timeout</span></div>"
    badges = []
    if r.get("source") == "cached":
        badges.append("<span class='badge grey'>cached answer</span>")
    cc = r.get("checks", {}).get("claim_check")
    if cc == "fail":
        badges.append("<span class='badge amber'>number not found in source</span>")
    elif cc == "pass":
        badges.append("<span class='badge green'>numbers match source</span>")
    cls = "card abstained" if r.get("status") == "abstained" else "card"
    cites = "".join(
        f"<div class='cite'>[{html.escape(c['pid'])}] "
        + (f"<a href='{html.escape(c['url'])}' target='_blank'>{html.escape(c.get('title') or '')}</a>" if c.get("url") else "")
        + "</div>"
        for c in r.get("citations", []))
    note = f"<div class='cite'>{html.escape(r['note'])}</div>" if r.get("note") else ""
    return f"<div class='{cls}'>{head}{''.join(badges)}<p>{html.escape(r.get('answer', ''))}</p>{cites}{note}</div>"


def run(question: str, mode: str, request: gr.Request):
    masked, n = mask(question or "")
    headers = {"x-session-id": getattr(request, "session_hash", None) or "anon"}
    body = {"request_id": str(uuid.uuid4()), "question": masked}
    try:
        if mode == "Compare all 4 arms":
            r = httpx.post(f"{BACKEND}/v1/compare", json=body, headers=headers, timeout=60).json()
            results = r.get("results") or [r]
        else:
            r = httpx.post(f"{BACKEND}/v1/ask", json={**body, "arm": "A3"}, headers=headers, timeout=30).json()
            results = [r]
    except httpx.HTTPError as e:
        results = [{"arm": "-", "status": "error", "error": {"code": "NETWORK", "message": str(e)}}]
    if len(results) == 1 and results[0].get("status") == "error" and "error" in results[0] and "arm" not in results[0]:
        results[0]["arm"] = "-"
    info = f"Masked {n} number(s) that looked like account or card numbers." if n else ""
    grid = "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px'>"
    return grid + "".join(card(x) for x in results) + "</div>", info


def examples() -> list[list[str]]:
    try:
        ex = httpx.get(f"{BACKEND}/v1/examples", timeout=5).json()["examples"]
        return [[e["question"]] for e in ex]
    except (httpx.HTTPError, KeyError):
        return []


with gr.Blocks(title="DhanGround", css=CSS) as demo:
    gr.Markdown("## DhanGround: Saaf Data, Sahi Jawab\nBanking questions in Hindi, Hinglish or English, "
                "answered only from public RBI rules, with citations.")
    q = gr.Textbox(label="Your question", max_lines=3, max_length=500,
                   placeholder="UPI payment fail hua par paisa kat gaya, kab wapas aayega?")
    mode = gr.Radio(["Compare all 4 arms", "Best arm only (A3)"], value="Compare all 4 arms", label="Mode")
    go = gr.Button("Ask", variant="primary")
    info = gr.Markdown()
    out = gr.HTML()
    ex_box = gr.Examples(examples=[], inputs=[q], label="Try these")
    go.click(run, [q, mode], [out, info])
    q.submit(run, [q, mode], [out, info])
    gr.Markdown(f"<small>{FOOTER}</small>")
    demo.load(lambda: gr.Dataset(samples=examples()), None, ex_box.dataset)

app = gr.mount_gradio_app(api_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
