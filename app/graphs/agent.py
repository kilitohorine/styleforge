"""LangGraph: route → perceive → execute → critique (Look). image.2d skips critique."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graphs.critique import critique_node
from app.graphs.perceive import perceive_node
from app.graphs.route import AgentState, after_route, route_node
from app.jobs.run import run_image_2d, run_photo_look
from app.jobs.store import new_id
from app.jobs.threads import load_thread, save_thread
from app.renderers.image_2d import ART_STYLES
from app.settings import settings


def execute_node(state: AgentState) -> AgentState:
    style_id = state["style_id"]
    asset_id = state.get("asset_id")
    assert style_id
    step = int(state.get("step") or 1)
    if state.get("intent") == "image2d":
        job = run_image_2d(style_id, state.get("message") or "", asset_id)
        name = ART_STYLES.get(style_id, {}).get("name") or style_id
        if job.status == "succeeded":
            extra = f" 任务 {job.job_id} 已完成，约 {job.actual_cost_cny} 元（{job.trace.renderer}）。"
        else:
            err = job.error.message if job.error else "failed"
            extra = f" 2D 未出图：{err}"
        reply = (state.get("reply") or "") + extra
        return {**state, "job_id": job.job_id, "params": job.trace.params, "reply": reply, "step": step}

    assert asset_id
    actions = ["perceive", "plan_greedy_look", "execute_cube_lut"]
    if state.get("is_patch"):
        actions.insert(1, "patch_params")
    if step > 1:
        actions.append("critique_retry")
    job = run_photo_look(
        style_id,
        asset_id,
        state.get("params"),
        extra_trace={
            "actions": actions,
            "scene": state.get("scene"),
            "retry_count": max(0, step - 1),
        },
    )
    reply = state.get("reply") or ""
    return {
        **state,
        "job_id": job.job_id,
        "params": job.trace.params,
        "step": step,
        "reply": reply + f" 任务 {job.job_id} 已完成，费用 0 元。对比图已写入 trace。",
    }


def after_execute(state: AgentState) -> str:
    if state.get("intent") == "look" and settings.enable_critique:
        return "critique"
    return "end"


def after_critique(state: AgentState) -> str:
    step = int(state.get("step") or 1)
    if state.get("need_retry") and step < int(settings.max_critique_step):
        return "execute"
    return "end"


def bump_step(state: AgentState) -> AgentState:
    if state.get("need_retry"):
        return {**state, "step": int(state.get("step") or 1) + 1, "need_retry": False}
    return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route_node)
    g.add_node("perceive_scene", perceive_node)
    g.add_node("execute", execute_node)
    g.add_node("critique_look", critique_node)
    g.add_node("bump", bump_step)
    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        after_route,
        {"perceive": "perceive_scene", "execute": "execute", "end": END},
    )
    g.add_edge("perceive_scene", "execute")
    g.add_conditional_edges("execute", after_execute, {"critique": "critique_look", "end": END})
    g.add_conditional_edges("critique_look", after_critique, {"execute": "bump", "end": END})
    g.add_edge("bump", "execute")
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
            "step": 1,
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
