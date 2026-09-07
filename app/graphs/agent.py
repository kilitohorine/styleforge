"""LangGraph: route → execute (PhotoAgent-style closed loop, day-1)."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graphs.route import AgentState, after_route, route_node
from app.jobs.store import dump_output, new_id, save_job
from app.renderers.photo_look import PhotoLookRenderer, encode_jpeg
from app.jobs.store import asset_path
from app.schemas.job import InputAsset, JobOut, JobTrace


def execute_node(state: AgentState) -> AgentState:
    style_id = state["style_id"]
    asset_id = state["asset_id"]
    assert style_id and asset_id
    source = asset_path(asset_id)
    renderer = PhotoLookRenderer()
    bgr, params = renderer.run(source, style_id)
    out_id = dump_output(encode_jpeg(bgr), style_id)
    job = JobOut(
        job_id=new_id("j"),
        status="succeeded",
        modality="image.photo_look",
        estimated_cost_cny=0.0,
        actual_cost_cny=0.0,
        outputs=[InputAsset(asset_id=out_id, role="source")],
        trace=JobTrace(
            style_id=style_id,
            renderer="photo_look.opencv",
            params=params,
            actions=["perceive_intent", "plan_greedy_look", "execute_opencv", "skip_mcts"],
        ),
    )
    save_job(job)
    reply = state.get("reply") or ""
    return {
        **state,
        "job_id": job.job_id,
        "reply": reply + f" 任务 {job.job_id} 已完成，费用 0 元。",
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


def run_agent(message: str, asset_id: str | None = None) -> AgentState:
    graph = get_graph()
    return graph.invoke({"message": message, "asset_id": asset_id})
