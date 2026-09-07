"""LangGraph: route → execute. LUT looks + multi-turn param patch."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graphs.route import AgentState, after_route, route_node
from app.jobs.store import asset_path, dump_output, new_id, save_job
from app.jobs.threads import load_thread, save_thread
from app.renderers.photo_look import LOOKS, PhotoLookRenderer, encode_jpeg
from app.schemas.job import InputAsset, JobOut, JobTrace


def execute_node(state: AgentState) -> AgentState:
    style_id = state["style_id"]
    asset_id = state["asset_id"]
    assert style_id and asset_id
    source = asset_path(asset_id)
    renderer = PhotoLookRenderer()
    bgr, compare, params = renderer.run(source, style_id, state.get("params"))
    out_id = dump_output(encode_jpeg(bgr), style_id, kind=f"look:{style_id}")
    cmp_id = dump_output(encode_jpeg(compare), style_id, kind="compare")
    actions = ["perceive_intent", "plan_greedy_look", "execute_cube_lut"]
    if state.get("is_patch"):
        actions.insert(1, "patch_params")
    job = JobOut(
        job_id=new_id("j"),
        status="succeeded",
        modality="image.photo_look",
        estimated_cost_cny=0.0,
        actual_cost_cny=0.0,
        outputs=[
            InputAsset(asset_id=out_id, role="result"),
            InputAsset(asset_id=cmp_id, role="compare"),
        ],
        trace=JobTrace(
            style_id=style_id,
            renderer="photo_look.cube_lut",
            params=params,
            actions=actions,
            source_asset_id=asset_id,
            comparison_asset_id=cmp_id,
            lut=LOOKS[style_id]["lut"],
        ),
    )
    save_job(job)
    reply = state.get("reply") or ""
    return {
        **state,
        "job_id": job.job_id,
        "params": params,
        "reply": reply + f" 任务 {job.job_id} 已完成，费用 0 元。对比图已写入 trace。",
    }


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route_node)
    g.add_node("execute", execute_node)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, {"execute": "execute", "end": END})
    g.add_edge("execute", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_agent(
    message: str,
    asset_id: str | None = None,
    thread_id: str | None = None,
) -> AgentState:
    thread_id = thread_id or new_id("t")
    prev = load_thread(thread_id) or {}
    graph = get_graph()
    state = graph.invoke(
        {
            "message": message,
            "asset_id": asset_id or prev.get("asset_id"),
            "thread_id": thread_id,
            "style_id": prev.get("style_id"),
            "params": prev.get("params") or {},
        }
    )
    save_thread(
        thread_id,
        style_id=state.get("style_id") or prev.get("style_id"),
        asset_id=state.get("asset_id") or prev.get("asset_id"),
        params=state.get("params") or prev.get("params") or {},
        last_job_id=state.get("job_id") or prev.get("last_job_id"),
    )
    state["thread_id"] = thread_id
    return state
