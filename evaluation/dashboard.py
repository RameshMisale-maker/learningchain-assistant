"""Sapphirus RAG Evaluation dashboard - independent of the chatbot UI.

    evaluation\\run_dashboard.bat                         (recommended: uses evaluation/.venv with RAGAS)
    evaluation\\.venv\\Scripts\\python -m streamlit run evaluation/dashboard.py

Nothing is executed when the page opens; evaluations start only from the "Run Evaluation" button.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from evaluation import config  # noqa: E402
from evaluation import dataset_manager as dm  # noqa: E402
from evaluation import hallucination_metrics as hm  # noqa: E402
from evaluation import report_generator as rg  # noqa: E402
from evaluation import retrieval_metrics as rm  # noqa: E402

st.set_page_config(page_title="RAG Evaluation", page_icon=":bar_chart:", layout="wide")

PAGES = ["1. Evaluation Overview", "2. Golden Dataset", "3. Retrieval Evaluation", "4. Generation Evaluation",
         "5. Hallucination Evaluation", "6. Safety Evaluation", "7. Performance Evaluation",
         "8. Question-Level Analysis"]
NOT_MEASURED = "Not measured"


# ------------------------------------------------------------------------------------------------ formatting
def pct(v) -> str:
    return NOT_MEASURED if v is None or (isinstance(v, float) and pd.isna(v)) else f"{float(v) * 100:.1f}%"


def ms(v) -> str:
    return NOT_MEASURED if v is None or (isinstance(v, float) and pd.isna(v)) else f"{float(v):,.0f} ms"


def usd(v) -> str:
    return NOT_MEASURED if v is None else f"${float(v):.5f}"


def target_badge(key: str, value) -> str:
    t = config.TARGETS.get(key)
    if not t or value is None:
        return ""
    ok = value >= t["target"] if t["direction"] == "min" else value <= t["target"]
    sign = ">=" if t["direction"] == "min" else "<="
    return f"{'Meets' if ok else 'Below'} target ({sign} {t['target'] * 100:.0f}%)" if t["direction"] == "min" \
        else f"{'Meets' if ok else 'Above'} target ({sign} {t['target'] * 100:.0f}%)"


def cost_unknown(perf: dict) -> bool:
    """Tokens were used but the chatbot has no price configured for its provider -> cost is unknown, not $0."""
    return bool(perf.get("cost_note")) or (not perf.get("total_cost_usd") and bool(perf.get("total_input_tokens")))


COST_NOTE = ("Token prices for the chatbot's current provider are not configured in .env (OPENAI_PRICE_INPUT_PER_1M /"
             " OPENAI_PRICE_OUTPUT_PER_1M), so it reports $0 - the real cost is unknown, not zero.")


def metric_card(col, label: str, value_text: str, note: str = "") -> None:
    col.metric(label, value_text)
    if note:
        col.caption(note)


@st.cache_data(show_spinner=False)
def cached_run(run_id: str, mtime: float) -> dict:
    data = rg.load_run(run_id)
    data["path"] = str(data["path"])
    return data


def selected_run() -> dict | None:
    runs = rg.list_runs()
    if not runs:
        return None
    ids = [r["run_id"] for r in runs]
    default = st.session_state.get("selected_run")
    idx = ids.index(default) if default in ids else 0
    labels = {r["run_id"]: f"{r['run_id']}" + (f"  ({r['label']})" if r["label"] else "") for r in runs}
    rid = st.sidebar.selectbox("Evaluation run", ids, index=idx, format_func=lambda x: labels[x])
    st.session_state["selected_run"] = rid
    path = config.RESULTS_DIR / rid / "summary.json"
    return cached_run(rid, path.stat().st_mtime)


def not_run_notice() -> None:
    st.info("**Evaluation Not Yet Run.** Upload or review the golden dataset, then start a run from "
            "*1. Evaluation Overview*. No scores are shown until a real evaluation has been executed.")


def _fmt_value(value, unit: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return NOT_MEASURED
    if unit == "fraction":
        return f"{float(value) * 100:.1f}%"
    if unit == "ms":
        return f"{float(value):,.0f} ms"
    if unit == "USD":
        return f"${float(value):.5f}"
    if unit in ("tokens", "count"):
        return f"{int(value):,}"
    return f"{float(value):.3f} {unit}"


def download_buttons(run: dict, key: str) -> None:
    d = Path(run["path"])
    cols = st.columns(7)
    for col, (name, label, mime) in zip(cols, [
            ("evaluation_metrics.csv", "Evaluation metrics CSV (3 layers)", "text/csv"),
            ("load_test.csv", "Load test CSV", "text/csv"),
            ("question_report.csv", "Question-level CSV", "text/csv"),
            ("failed_questions.csv", "Failed questions CSV", "text/csv"),
            ("summary.json", "Summary JSON", "application/json"),
            ("questions.json", "Full traces JSON", "application/json"),
            ("safety_report.csv", "Safety CSV", "text/csv")]):
        if (d / name).exists():
            col.download_button(label, (d / name).read_bytes(), file_name=f"{run['run_id']}_{name}", mime=mime,
                                key=f"{key}_{name}", width="stretch")


def golden_lookup() -> dict[str, dict]:
    return {r["question_id"]: r for r in dm.load().to_dict("records")}


# ------------------------------------------------------------------------------------------------ page 1
def page_overview(run: dict | None) -> None:
    st.title("Evaluation Overview")
    with st.expander("Run a new evaluation", expanded=run is None):
        run_panel()
    if run is None:
        not_run_notice()
        return
    m, meta = run["summary"].get("metrics", {}), run["meta"]
    bot = meta.get("chatbot", {})
    st.caption(f"Run **{run['run_id']}** - started {meta.get('started_at', '')[:19]} UTC - "
               f"{meta.get('question_count')} questions + {meta.get('safety_test_count', 0)} safety tests - "
               f"git {meta.get('git_commit') or 'n/a'} - dataset {meta.get('dataset_hash')}")
    st.caption(f"Production LLM: **{bot.get('llm_model')}** ({bot.get('llm_provider')}) - embeddings: "
               f"**{bot.get('embedding_model')}** - retrieval: {bot.get('vector_backend')} + BM25, re-ranker "
               f"{bot.get('reranker')} - evaluator: **{meta.get('evaluator', {}).get('ragas_model')}**")
    cp = m.get("ragas_context_precision") if m.get("ragas_context_precision") is not None else m.get("context_precision")
    cr = m.get("ragas_context_recall") if m.get("ragas_context_recall") is not None else m.get("context_recall")
    cards = [
        ("Total questions evaluated", str(m.get("questions_evaluated", 0)), f"pass rate {pct(m.get('pass_rate'))}"),
        ("Faithfulness", pct(m.get("faithfulness")), target_badge("faithfulness", m.get("faithfulness"))),
        ("Answer Correctness", pct(m.get("answer_correctness")), target_badge("answer_correctness", m.get("answer_correctness"))),
        ("Answer Relevancy", pct(m.get("answer_relevancy")), target_badge("answer_relevancy", m.get("answer_relevancy"))),
        (f"Retrieval Precision@{run['summary'].get('k', 5)}", pct(m.get("precision_at_k")), ""),
        (f"Retrieval Recall@{run['summary'].get('k', 5)}", pct(m.get("recall_at_k")), target_badge("recall_at_k", m.get("recall_at_k"))),
        ("Context Precision", pct(cp), ("RAGAS. " if m.get("ragas_context_precision") is not None else "rank-based. ")
         + target_badge("context_precision", cp)),
        ("Context Recall", pct(cr), ("RAGAS. " if m.get("ragas_context_recall") is not None else "lexical. ")
         + target_badge("context_recall", cr)),
        ("Hallucination Rate", pct(m.get("hallucination_rate")), target_badge("hallucination_rate", m.get("hallucination_rate"))),
        ("Average Response Latency", ms(m.get("avg_latency_ms")), ""),
        ("P95 Latency", ms(m.get("p95_latency_ms")), ""),
        ("Cost Per Query", "Price not configured" if cost_unknown(run["performance"]) else usd(m.get("cost_per_query_usd")),
         "LLM token usage x the chatbot's .env prices" if not cost_unknown(run["performance"])
         else f"{run['performance'].get('total_input_tokens', 0):,} in / {run['performance'].get('total_output_tokens', 0):,} out tokens measured"),
    ]
    for row in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, (label, value, note) in zip(cols, cards[row:row + 4]):
            metric_card(col, label, value, note)

    layers = run["summary"].get("layers") or []
    if layers:
        st.subheader("Three-layer evaluation")
        st.caption("1. Retrieval: does the right content reach the LLM?  2. Generation: is the answer faithful, "
                   "relevant and correct (RAGAS + LLM-as-judge)?  3. End-to-end production: behaviour, safety, "
                   "latency, load, cost and errors through the live pipeline.")
        ldf = pd.DataFrame(layers)
        for layer_name in ldf["layer"].unique():
            part = ldf[ldf["layer"] == layer_name].drop(columns=["layer"])
            part = part.assign(value=[_fmt_value(v, u) for v, u in zip(part["value"], part["unit"])])
            with st.expander(layer_name, expanded=True):
                st.dataframe(part, hide_index=True, width="stretch")
        st.download_button("Download evaluation metrics (CSV)", ldf.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{run['run_id']}_evaluation_metrics.csv", mime="text/csv", type="primary")

    st.subheader("Benchmark targets")
    st.caption("Configurable engineering targets (evaluation/config.py) - not guaranteed results.")
    trows = []
    for key, t in run["summary"].get("targets", {}).items():
        sign = ">=" if t["direction"] == "min" else "<="
        status = NOT_MEASURED if t["met"] is None else ("Met" if t["met"] else "Not met")
        trows.append({"Metric": t["label"], "Target": f"{sign} {t['target'] * 100:.0f}%", "Actual": pct(t["value"]),
                      "Status": status})
    st.dataframe(pd.DataFrame(trows), hide_index=True, width="stretch")
    warnings = meta.get("dataset_warnings") or []
    if bot.get("llm_provider") != "anthropic":
        warnings = [f"The production LLM in this run was {bot.get('llm_model')} ({bot.get('llm_provider')}), not "
                    "Claude: no ANTHROPIC_API_KEY is configured, so the chatbot used its fallback provider."] + warnings
    if warnings:
        with st.expander(f"Run notes ({len(warnings)})"):
            for w in warnings:
                st.warning(w)
    st.subheader("Reports")
    download_buttons(run, "ov")

    st.subheader("Compare evaluation runs")
    runs = [r["run_id"] for r in rg.list_runs()]
    chosen = st.multiselect("Runs to compare", runs, default=runs[:min(3, len(runs))])
    if chosen:
        cmp = rg.compare_runs(chosen)
        keys = ["questions", "llm", "embedding", "pass_rate", "faithfulness", "answer_correctness",
                "answer_relevancy", "ragas_context_precision", "ragas_context_recall", "recall_at_k",
                "precision_at_k", "mrr", "ndcg_at_k", "hallucination_rate", "abstention_accuracy",
                "restricted_rejection_rate", "safety_pass_rate", "avg_latency_ms", "p95_latency_ms",
                "cost_per_query_usd"]
        show = cmp.set_index("run_id")[[k for k in keys if k in cmp.columns]].T
        st.dataframe(show, width="stretch")
        st.download_button("Download comparison CSV", show.to_csv().encode("utf-8"), "run_comparison.csv", "text/csv")


def run_panel() -> None:
    from evaluation.run_evaluation import RunOptions, run as run_eval

    df = dm.load()
    if df.empty:
        st.warning("No golden dataset yet - add questions on *2. Golden Dataset*.")
        return
    errors, warnings = dm.validate(df)
    st.write(f"Golden dataset: **{len(df)} questions** ({config.GOLDEN_DATASET_PATH.name})")
    if errors:
        for e in errors:
            st.error(e)
        st.write("Fix these on *2. Golden Dataset* before running an evaluation.")
        return
    cats = st.multiselect("Question categories to evaluate", config.QUESTION_CATEGORIES,
                          default=[c for c in config.QUESTION_CATEGORIES if c in set(df["question_category"])])
    c1, c2, c3, c4 = st.columns(4)
    use_ragas = c1.checkbox("RAGAS generation metrics", True)
    use_hall = c2.checkbox("Claim-level hallucination", True)
    use_safety = c3.checkbox("Safety test suite", True)
    judge = c4.selectbox("Claim judge", ["llm", "nli"], help="llm = evaluator model; nli = local NLI cross-encoder")
    c5, c6 = st.columns(2)
    limit = c5.number_input("Limit questions (0 = all selected)", 0, len(df), 0)
    label = c6.text_input("Run label (optional)", "")
    ragas_n = st.number_input("RAGAS sample size (0 = every answered question)", 0, len(df), 0,
                              help="Stratified by category. Use a sample when the judge model is rate-limited.")
    load_url = st.text_input("Load test: running chatbot API URL (blank = skip)", "",
                             help="Concurrent requests through the real HTTP path, e.g. http://127.0.0.1:8001 started "
                                  "with a high RATE_LIMIT_PER_MINUTE")
    from evaluation.ragas_evaluator import ragas_status

    ok, msg = ragas_status()
    (st.caption if ok else st.warning)(msg)
    n = len(df[df["question_category"].isin(cats)]) if cats else 0
    n = min(n, limit) if limit else n
    st.caption(f"This run calls the live chatbot pipeline {n} times"
               + (f" + {len(pd.read_csv(config.SAFETY_TESTS_PATH))} safety prompts" if use_safety else "")
               + " (production LLM API usage is billed as normal), plus evaluator-model calls for RAGAS / claims.")
    if st.button("Run Evaluation", type="primary", disabled=not cats):
        bar, status = st.progress(0.0), st.empty()

        def progress(frac: float, text: str) -> None:
            bar.progress(frac)
            status.write(text)

        try:
            rid = run_eval(df, RunOptions(categories=cats, limit=int(limit) or None, run_ragas=use_ragas,
                                          run_hallucination=use_hall, run_safety=use_safety,
                                          hallucination_judge=judge, label=label,
                                          load_test_url=load_url.strip(), ragas_sample=int(ragas_n)), progress)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state["selected_run"] = rid
        st.success(f"Evaluation finished: {rid}")
        st.rerun()


# ------------------------------------------------------------------------------------------------ page 2
def page_dataset() -> None:
    st.title("Golden Dataset")
    st.caption(f"File: {config.GOLDEN_DATASET_PATH}. Every save keeps the previous version in "
               f"{config.DATASET_HISTORY_DIR.name}/.")
    up = st.file_uploader("Upload golden_dataset.csv", type=["csv"])
    if up is not None:
        try:
            new = dm.read_upload(up)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            new = None
        if new is not None:
            errors, warnings = dm.validate(new, _inventory())
            st.write(f"Uploaded **{len(new)}** questions.")
            for e in errors:
                st.error(e)
            for w in warnings:
                st.warning(w)
            st.dataframe(new, width="stretch", hide_index=True)
            if st.button("Replace the golden dataset with this upload", disabled=bool(errors)):
                backup = dm.save(new)
                st.success("Saved." + (f" Previous version backed up to {backup.name}." if backup else ""))
                st.rerun()

    df = dm.load()
    st.subheader(f"Questions ({len(df)})")
    f1, f2 = st.columns(2)
    cat_filter = f1.multiselect("Filter by category", config.QUESTION_CATEGORIES)
    text = f2.text_input("Search questions")
    view = df
    if cat_filter:
        view = view[view["question_category"].isin(cat_filter)]
    if text:
        view = view[view["question"].str.contains(text, case=False, regex=False)]
    edited = st.data_editor(
        view, num_rows="dynamic", width="stretch", hide_index=True, key="golden_editor",
        column_config={
            "question_category": st.column_config.SelectboxColumn(options=config.QUESTION_CATEGORIES, required=True),
            "expected_behavior": st.column_config.SelectboxColumn(options=config.EXPECTED_BEHAVIORS, required=True),
            "role": st.column_config.SelectboxColumn(options=list(config.ROLE_SETS), default="public"),
            "expected_answer": st.column_config.TextColumn(width="large"),
            "question": st.column_config.TextColumn(width="large"),
            "relevant_document": st.column_config.TextColumn(help="Document name, file, section or approved-answer "
                                                                  "id; separate several with |"),
            "relevant_chunk_ids": st.column_config.TextColumn(help="Child or parent chunk ids, separated by |"),
        })
    c1, c2, c3 = st.columns(3)
    if c1.button("Validate dataset"):
        merged = _merge_edit(df, view, edited)
        errors, warnings = dm.validate(merged, _inventory())
        if not errors:
            st.success(f"No blocking errors ({len(merged)} questions).")
        for e in errors:
            st.error(e)
        for w in warnings:
            st.warning(w)
    if c2.button("Save changes", type="primary"):
        merged = _merge_edit(df, view, edited)
        errors, _ = dm.validate(merged)
        if errors:
            for e in errors:
                st.error(e)
        else:
            backup = dm.save(merged)
            st.success("Saved." + (f" Previous version: {backup.name}" if backup else ""))
            st.rerun()
    c3.download_button("Download golden_dataset.csv", dm.normalize(df).to_csv(index=False).encode("utf-8"),
                       "golden_dataset.csv", "text/csv")

    with st.expander("Add a new question"):
        with st.form("add_question", clear_on_submit=True):
            a1, a2, a3 = st.columns(3)
            qid = a1.text_input("question_id", value=_next_id(df))
            cat = a2.selectbox("question_category", config.QUESTION_CATEGORIES)
            beh = a3.selectbox("expected_behavior", config.EXPECTED_BEHAVIORS)
            question = st.text_area("question")
            expected = st.text_area("expected_answer")
            ref = st.text_area("reference_context (key statements, separate with |)")
            b1, b2, b3 = st.columns(3)
            doc = b1.text_input("relevant_document (separate with |)")
            chunks = b2.text_input("relevant_chunk_ids (separate with |)")
            role = b3.selectbox("role", list(config.ROLE_SETS))
            if st.form_submit_button("Add question"):
                row = {"question_id": qid, "question": question, "expected_answer": expected,
                       "reference_context": ref, "relevant_document": doc, "relevant_chunk_ids": chunks,
                       "question_category": cat, "expected_behavior": beh, "role": role, "notes": "added in dashboard"}
                merged = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                errors, _ = dm.validate(merged)
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    dm.save(merged)
                    st.success(f"Added {qid}.")
                    st.rerun()
    with st.expander("Import the project's existing evaluation dataset"):
        st.write("Converts datasets/evaluation_dataset/evaluation_dataset.csv (read-only) into the golden format. "
                 "Company questions are skipped because the Company knowledge base is disabled.")
        if st.button("Import (replaces the current golden dataset; previous version is backed up)"):
            dm.save(dm.import_legacy_dataset())
            st.rerun()


def _merge_edit(full: pd.DataFrame, view: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
    """Apply edits made on a filtered view back onto the full dataset (rows removed in the view are deleted)."""
    rest = full.drop(index=view.index)
    return pd.concat([rest, edited], ignore_index=True).fillna("")


def _next_id(df: pd.DataFrame) -> str:
    n = len(df) + 1
    existing = set(df["question_id"])
    while f"Q{n:03d}" in existing:
        n += 1
    return f"Q{n:03d}"


@st.cache_resource(show_spinner="Reading the vector index (read-only)...")
def _inventory_cached():
    from evaluation.chatbot_adapter import index_inventory

    return index_inventory()


def _inventory():
    try:
        return _inventory_cached()
    except Exception as exc:
        st.warning(f"Could not read the vector index for chunk-id validation: {exc}")
        return None


# ------------------------------------------------------------------------------------------------ page 3
def page_retrieval(run: dict | None) -> None:
    st.title("Retrieval Evaluation")
    if run is None:
        return not_run_notice()
    k = run["summary"].get("k", config.RETRIEVAL_K)
    labels = dm.relevance_labels()
    rows = []
    for q in run["questions"]:
        if q["expected_behavior"] != "Answer":
            continue
        ctx = [c["text"] for c in q["run"]["contexts"]]
        r = rm.compute(q["run"]["retrieved"], q, k, ctx, labels.get(q["question_id"]))
        rows.append({"question_id": q["question_id"], "category": q["question_category"],
                     "label_source": r["label_source"], "needs_annotation": r["needs_annotation"],
                     **{m: r[m] for m in rm.METRIC_KEYS}})
    agg = rm.aggregate(rows)
    st.caption(f"Computed live on the run's captured retrieval (top {k} re-ranked child chunks) with the current "
               "relevance labels. Questions without any label are excluded and listed for annotation.")
    cols = st.columns(7)
    for col, (key, label) in zip(cols, [("precision_at_k", f"Precision@{k}"), ("recall_at_k", f"Recall@{k}"),
                                        ("mrr", "MRR"), ("ndcg_at_k", f"NDCG@{k}"), ("hit_rate_at_k", f"Hit Rate@{k}"),
                                        ("context_precision", "Context Precision"),
                                        ("context_recall", "Context Recall (lexical)")]):
        col.metric(label, pct(agg[key]))
        col.caption(f"n = {agg[f'{key}_n']}")
    if agg["needs_annotation"]:
        st.warning(f"{agg['needs_annotation']} question(s) need human relevance labels.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Inspect retrieval and annotate relevance")
    qids = [r["question_id"] for r in rows]
    pending = [r["question_id"] for r in rows if r["needs_annotation"]]
    only_pending = st.checkbox("Only questions that need annotation", value=bool(pending))
    choices = pending if only_pending and pending else qids
    if not choices:
        return
    qid = st.selectbox("Question", choices)
    q = next(x for x in run["questions"] if x["question_id"] == qid)
    st.markdown(f"**{q['question']}**")
    st.caption(f"relevant_document: {q.get('relevant_document') or '-'} | relevant_chunk_ids: "
               f"{q.get('relevant_chunk_ids') or '-'}")
    current = labels.get(qid, {})
    show_n = st.slider("Chunks to show", k, max(k, len(q["run"]["retrieved"])), min(2 * k, max(k, len(q["run"]["retrieved"]))))
    flags, _ = rm.label_chunks(q["run"]["retrieved"], q, current)
    new_labels = {}
    for ch, flag in zip(q["run"]["retrieved"][:show_n], flags[:show_n]):
        head = (f"#{ch['rank']}  score {ch['rerank_score']:.3f}  |  {ch['doc_name']} > {ch['section']}"
                + ("  |  sent to LLM" if ch["sent_to_llm"] else ""))
        with st.expander(head, expanded=ch["rank"] <= 3):
            st.caption(f"child: {ch['chunk_id']}  |  parent: {ch['parent_id'] or '(none - standalone chunk)'}  |  "
                       f"vector rank {ch['vector_rank']}, BM25 rank {ch['bm25_rank']}")
            st.text(ch["text"])
            parent = next((c for c in q["run"]["contexts"] if ch["parent_id"] and c["parent_id"] == ch["parent_id"]), None)
            if parent:
                st.caption("Parent context sent to the LLM:")
                st.text(parent["text"][:3000])
            new_labels[ch["text_key"]] = st.checkbox("Relevant to the question", value=bool(flag),
                                                     key=f"rel_{run['run_id']}_{qid}_{ch['text_key']}")
    if st.button("Save relevance labels", type="primary"):
        dm.save_relevance_labels(qid, new_labels)
        st.success(f"Saved {len(new_labels)} labels for {qid} (evaluation/annotations/relevance_labels.json).")
        st.rerun()


# ------------------------------------------------------------------------------------------------ page 4
def page_generation(run: dict | None) -> None:
    st.title("Generation Evaluation (RAGAS)")
    if run is None:
        return not_run_notice()
    ev = run["meta"].get("evaluator", {})
    st.caption(f"Evaluator model: **{ev.get('ragas_model')}** - production model: "
               f"**{run['meta'].get('chatbot', {}).get('llm_model')}**. Status: {ev.get('ragas')}")
    m = run["summary"]["metrics"]
    cols = st.columns(5)
    for col, (key, label) in zip(cols, [("faithfulness", "Faithfulness"), ("answer_relevancy", "Answer Relevancy"),
                                        ("answer_correctness", "Answer Correctness"),
                                        ("ragas_context_precision", "Context Precision"),
                                        ("ragas_context_recall", "Context Recall")]):
        col.metric(label, pct(m.get(key)))
    rows = []
    for q in run["questions"]:
        rag = q.get("ragas", {})
        if q["run"]["behavior"] != "Answer":
            status = f"not scored ({q['run']['behavior'].lower()})"
        elif rag.get("ragas_error"):
            status = rag["ragas_error"]
        elif rag:
            status = "scored"
        else:
            status = "RAGAS not run"
        rows.append({"question_id": q["question_id"], "category": q["question_category"],
                     "faithfulness": rag.get("faithfulness"), "answer_relevancy": rag.get("answer_relevancy"),
                     "answer_correctness": rag.get("answer_correctness"),
                     "context_precision": rag.get("context_precision"), "context_recall": rag.get("context_recall"),
                     "lexical_f1": q.get("lexical_f1"), "evaluation_status": status})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    qid = st.selectbox("Question detail", [r["question_id"] for r in rows])
    q = next(x for x in run["questions"] if x["question_id"] == qid)
    question_detail(q, show=("answer", "contexts", "ragas"))


# ------------------------------------------------------------------------------------------------ page 5
def page_hallucination(run: dict | None) -> None:
    st.title("Hallucination Evaluation")
    if run is None:
        return not_run_notice()
    st.caption("Hallucination Rate = unsupported factual claims / total factual claims x 100. Contradicted claims "
               "are a subset of unsupported claims. Judge: "
               f"**{run['meta'].get('evaluator', {}).get('hallucination_judge')}** - judge verdicts can be wrong; "
               "review flagged claims below. Human verdicts are stored separately and override the judge here.")
    reviews = dm.claim_reviews(run["run_id"])
    per_q = [{"question_id": q["question_id"], **q.get("hallucination", {})} for q in run["questions"]
             if q.get("hallucination", {}).get("claims")]
    agg = hm.aggregate(per_q, reviews)
    cols = st.columns(6)
    for col, (label, value) in zip(cols, [("Total factual claims", agg["total_claims"]),
                                          ("Supported", agg["supported_claims"]),
                                          ("Unsupported", agg["unsupported_claims"]),
                                          ("Contradictory", agg["contradicted_claims"]),
                                          ("Hallucination rate", pct(agg["hallucination_rate"])),
                                          ("Awaiting human review", agg["awaiting_human_review"])]):
        col.metric(label, value)
    st.write("**Failed questions:** " + (", ".join(agg["failed_questions"]) or "none"))
    flagged = []
    for q in per_q:
        for c in hm.apply_human_reviews(q["claims"], reviews.get(q["question_id"], {})):
            if c["needs_human_review"]:
                flagged.append({"question_id": q["question_id"], "claim_key": c["claim_key"], "claim": c["claim"],
                                "judge_verdict": c["verdict"], "judge_evidence": c["evidence"],
                                "human_verdict": c["human_verdict"]})
    st.subheader(f"Human verification of flagged claims ({len(flagged)})")
    if flagged:
        fdf = pd.DataFrame(flagged)
        edited = st.data_editor(fdf, hide_index=True, width="stretch", disabled=[c for c in fdf.columns
                                                                                            if c != "human_verdict"],
                                column_config={"human_verdict": st.column_config.SelectboxColumn(
                                    options=["", hm.SUPPORTED, hm.UNSUPPORTED, hm.CONTRADICTED])},
                                key=f"claims_{run['run_id']}")
        if st.button("Save human verdicts", type="primary"):
            for qid, grp in edited.groupby("question_id"):
                dm.save_claim_reviews(run["run_id"], qid, dict(zip(grp["claim_key"], grp["human_verdict"])))
            st.success("Saved to evaluation/annotations/claim_reviews.json")
            st.rerun()
    else:
        st.write("No claims were flagged.")
    st.subheader("All claims")
    all_claims = [{"question_id": q["question_id"], **c} for q in per_q
                  for c in hm.apply_human_reviews(q["claims"], reviews.get(q["question_id"], {}))]
    if all_claims:
        st.dataframe(pd.DataFrame(all_claims).drop(columns=["claim_key"]), hide_index=True, width="stretch")


# ------------------------------------------------------------------------------------------------ page 6
def page_safety(run: dict | None) -> None:
    st.title("Safety Evaluation")
    st.caption(f"Test suite: {config.SAFETY_TESTS_PATH.name} (editable CSV). Tests run against the unchanged "
               "chatbot pipeline; failures are reported, chatbot behaviour is never altered.")
    if run is None:
        return not_run_notice()
    safety = run.get("safety") or []
    if not safety:
        st.info("The safety suite was not part of this run. Enable 'Safety test suite' when starting a run.")
        return
    s = run["summary"].get("safety", {})
    cols = st.columns(4)
    cols[0].metric("Tests", s.get("total"))
    cols[1].metric("Passed", s.get("passed"))
    cols[2].metric("Failed", s.get("failed"))
    cols[3].metric("Strict rejection rate", pct(s.get("strict_rejection_rate")),
                   help="Share of 'Reject' tests the chatbot rejected outright (out-of-scope / blocked)")
    st.dataframe(pd.DataFrame(s.get("by_category", [])), hide_index=True, width="stretch")
    df = pd.DataFrame(safety)
    only_fail = st.checkbox("Show failures only")
    if only_fail:
        df = df[~df["status"].str.startswith("PASS")]
    st.dataframe(df[["test_id", "category", "prompt", "expected_behavior", "actual_behavior", "status",
                     "failure_reason", "response", "notes"]], hide_index=True, width="stretch")


# ------------------------------------------------------------------------------------------------ page 7
def page_performance(run: dict | None) -> None:
    st.title("Performance Evaluation")
    if run is None:
        return not_run_notice()
    p = run["performance"]
    rows = [
        ("TTFT (avg)", ms(p.get("avg_ttft_ms")) if p.get("ttft_measured") else "Unavailable", p.get("ttft_note") or "measured (streaming LLM)"),
        ("Average latency", ms(p.get("avg_latency_ms")), "measured end-to-end inside the chatbot pipeline"),
        ("P50 latency", ms(p.get("p50_latency_ms")), "measured"),
        ("P95 latency", ms(p.get("p95_latency_ms")), "measured"),
        ("P99 latency", ms(p.get("p99_latency_ms")), "measured" + (" (few samples - indicative only)" if (p.get("requests") or 0) < 100 else "")),
        ("Retrieval latency (avg)", ms(p.get("avg_retrieval_ms")), "measured (vector + BM25 + fusion)"),
        ("Re-ranking latency (avg)", ms(p.get("avg_rerank_ms")), "measured"),
        ("LLM generation latency (avg)", ms(p.get("avg_llm_total_ms")), "measured (LLM call, requests that used the LLM)"),
        ("Input tokens (total / avg)", f"{p.get('total_input_tokens', 0):,} / {p.get('avg_input_tokens') or 0:,.0f}", "reported by the LLM API"),
        ("Output tokens (total / avg)", f"{p.get('total_output_tokens', 0):,} / {p.get('avg_output_tokens') or 0:,.0f}", "reported by the LLM API"),
        ("API cost (total)", "Unavailable" if cost_unknown(p) else usd(p.get("total_cost_usd")),
         COST_NOTE if cost_unknown(p) else "tokens x the chatbot's configured prices (.env)"),
        ("Cost per query", "Unavailable" if cost_unknown(p) else usd(p.get("cost_per_query_usd")),
         "price not configured" if cost_unknown(p) else "includes questions answered without the LLM"),
        ("Error rate", pct(p.get("error_rate")), f"{p.get('errors', 0)} error(s) in {p.get('requests')} requests"),
        ("Throughput", NOT_MEASURED if not p.get("throughput_qps") else f"{p['throughput_qps']:.2f} requests/s",
         "sequential, one request at a time (not a load test)"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value", "Source"]), hide_index=True, width="stretch")
    if p.get("llm_answers"):
        st.caption(f"{p.get('answers_by_fallback_model', 0)} of {p['llm_answers']} LLM answers came from the fallback "
                   "model (primary model rate-limited).")
    if p.get("load_test"):
        st.subheader("Concurrent load test")
        st.caption(f"Real HTTP requests through the streaming chat endpoint: {run['meta'].get('load_test', '')}")
        ldf = pd.DataFrame(p["load_test"])
        st.dataframe(ldf, hide_index=True, width="stretch")
        st.line_chart(ldf.set_index("concurrency")[["latency_p50_ms", "latency_p95_ms"]])
    lat = pd.DataFrame([{"question_id": q["question_id"], "latency_ms": q["run"]["latency_ms"],
                         "retrieval_ms": (q["run"].get("timings") or {}).get("retrieval_ms"),
                         "llm_total_ms": (q["run"].get("timings") or {}).get("llm_total_ms"),
                         "answer_type": q["run"]["answer_type"]} for q in run["questions"]])
    if not lat.empty:
        st.subheader("Latency per question")
        st.bar_chart(lat.set_index("question_id")[["latency_ms"]])
        st.dataframe(lat, hide_index=True, width="stretch")


# ------------------------------------------------------------------------------------------------ page 8
def page_questions(run: dict | None) -> None:
    st.title("Question-Level Analysis")
    if run is None:
        return not_run_notice()
    flat = pd.DataFrame([rg.flatten(q) for q in run["questions"]])
    c1, c2, c3, c4 = st.columns(4)
    status = c1.selectbox("Status", ["All", "FAIL", "PASS"])
    cats = c2.multiselect("Categories", sorted(flat["question_category"].unique()))
    metric = c3.selectbox("Metric filter", ["(none)", "faithfulness", "answer_correctness", "answer_relevancy",
                                            "recall_at_k", "precision_at_k", "hallucination_rate", "latency_ms"])
    view = flat
    if status != "All":
        view = view[view["status"] == status]
    if cats:
        view = view[view["question_category"].isin(cats)]
    if metric != "(none)":
        series = pd.to_numeric(view[metric], errors="coerce")
        lo, hi = float(series.min() if series.notna().any() else 0), float(series.max() if series.notna().any() else 1)
        if hi > lo:
            rng = c4.slider(f"{metric} range", lo, hi, (lo, hi))
            view = view[series.between(*rng)]
        else:
            view = view[series.notna()]
    st.caption(f"{len(view)} of {len(flat)} questions")
    st.dataframe(view, hide_index=True, width="stretch")
    st.download_button("Download filtered CSV", view.to_csv(index=False).encode("utf-8"), "questions_filtered.csv")
    if view.empty:
        return
    qid = st.selectbox("Question detail", view["question_id"].tolist())
    q = next(x for x in run["questions"] if x["question_id"] == qid)
    question_detail(q, show=("answer", "contexts", "retrieval", "ragas", "claims", "perf"))


def question_detail(q: dict, show=()) -> None:
    run = q["run"]
    status = q.get("status", "")
    st.markdown(f"### {q['question_id']} - {status}")
    if q.get("failure_reason"):
        st.error(q["failure_reason"])
    st.markdown(f"**Question:** {q['question']}")
    st.caption(f"Category: {q['question_category']} | expected: {q['expected_behavior']} | actual: {run['behavior']} "
               f"({run['answer_type']}) | relevant document: {q.get('relevant_document') or '-'}")
    a, b = st.columns(2)
    a.markdown("**Expected answer**")
    a.write(q.get("expected_answer") or "-")
    b.markdown(f"**Actual answer** ({run.get('model')})")
    b.write(run.get("answer") or "-")
    if "contexts" in show:
        with st.expander(f"Context sent to the LLM ({len(run['contexts'])} sources)"):
            for c in run["contexts"]:
                st.caption(f"[{c['sid']}] {c['doc_name']} > {c['section']} | child {c['chunk_id']} | parent "
                           f"{c['parent_id'] or '-'}")
                st.text(c["text"][:4000])
    if "retrieval" in show:
        with st.expander(f"Retrieved child chunks ({len(run['retrieved'])}) and retrieval metrics"):
            st.json({k: v for k, v in q.get("retrieval", {}).items() if k != "relevance_flags"})
            st.dataframe(pd.DataFrame([{k: c[k] for k in ("rank", "rerank_score", "doc_name", "section", "chunk_id",
                                                          "parent_id", "sent_to_llm")} for c in run["retrieved"]]),
                         hide_index=True, width="stretch")
    if "ragas" in show:
        st.markdown("**RAGAS scores**")
        st.json(q.get("ragas") or {"status": "not scored"})
    if "claims" in show and q.get("hallucination", {}).get("claims"):
        with st.expander("Claim analysis"):
            st.dataframe(pd.DataFrame(q["hallucination"]["claims"]), hide_index=True, width="stretch")
    if "perf" in show:
        t = run.get("timings") or {}
        st.caption(f"Latency {ms(run.get('latency_ms'))} | retrieval {ms(t.get('retrieval_ms'))} | LLM "
                   f"{ms(t.get('llm_total_ms'))} | TTFT {ms(run.get('ttft_ms')) if run.get('ttft_measured') else 'unavailable'}"
                   f" | tokens {run.get('prompt_tokens')} in / {run.get('completion_tokens')} out | "
                   f"cost {usd(run.get('cost_usd'))}")


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    config.ensure_dirs()
    st.sidebar.title("RAG Evaluation")
    st.sidebar.caption("Independent of the chatbot - reads its pipeline, never modifies it.")
    page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")
    run = selected_run()
    if run is None:
        st.sidebar.info("No evaluation runs yet.")
    {
        PAGES[0]: lambda: page_overview(run), PAGES[1]: page_dataset, PAGES[2]: lambda: page_retrieval(run),
        PAGES[3]: lambda: page_generation(run), PAGES[4]: lambda: page_hallucination(run),
        PAGES[5]: lambda: page_safety(run), PAGES[6]: lambda: page_performance(run),
        PAGES[7]: lambda: page_questions(run),
    }[page]()


main()
