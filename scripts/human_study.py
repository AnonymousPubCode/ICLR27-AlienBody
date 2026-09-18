#!/usr/bin/env python3
"""Streamlit-based human study interface for AlienBody.

Usage:
    streamlit run scripts/human_study.py

Features:
    - Grid image display (512x512) with sprite-based rendering
    - Action buttons (0-5) + "Done Exploring" button
    - Phase indicator and progress bar
    - Timer tracking per environment (subtle, non-pressuring)
    - JSONL data collection per participant
    - Tutorial with practice environment
    - Progress persistence (can resume)
    - Supports all 6 families
    - Chinese/English bilingual UI
"""
from __future__ import annotations

import json
import uuid
import time
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import streamlit as st
except ImportError:
    print("Install streamlit: pip install streamlit")
    sys.exit(1)

import numpy as np
from alienbody.env.core import AlienBodyEnv
from alienbody.env.grid import EnvConfig
from alienbody.trajectory import TrajectoryWriter

# ── i18n ─────────────────────────────────────────────────────────────

T = {
    "en": {
        # App
        "app.title": "AlienBody Human Study",
        "app.icon": "🧑‍🚀",
        "lang.label": "Language / 语言",
        "lang.en": "English",
        "lang.zh": "中文",

        # Sidebar
        "sidebar.participant": "Participant",
        "sidebar.progress": "Progress",
        "sidebar.success_rate": "Success rate",
        "sidebar.rounds_done": "rounds",
        "sidebar.language": "Language",

        # Consent
        "consent.title": "Welcome!",
        "consent.intro": """
### What you'll do

You will control a character in a grid world. In each round:

1. **Phase 1 (Explore):** You have unlabeled buttons (Action 0–3).
   Press them to discover what each one does.
2. **Phase 2 (Navigate):** A cyan target will appear.
   Navigate to it using what you learned.

**Each round takes ~1–2 minutes. There are 30 rounds total (~30–45 minutes).**

Your data will be anonymized and used for academic research.
""",
        "consent.age": "Age",
        "consent.gender": "Gender",
        "consent.gender.male": "Male",
        "consent.gender.female": "Female",
        "consent.gender.nb": "Non-binary",
        "consent.gender.na": "Prefer not to say",
        "consent.gaming": "Gaming experience",
        "consent.gaming.none": "None",
        "consent.gaming.casual": "Casual",
        "consent.gaming.regular": "Regular",
        "consent.gaming.expert": "Expert",
        "consent.puzzle": "Puzzle/logic game experience",
        "consent.puzzle.none": "None",
        "consent.puzzle.some": "Some",
        "consent.puzzle.frequent": "Frequent",
        "consent.checkbox": "I consent to participate in this study. I understand my data will be anonymized.",
        "consent.submit": "Start Tutorial →",

        # Tutorial
        "tutorial.title": "Tutorial — Learn the Interface",
        "tutorial.instructions": """
### How it works

You see a **grid** with colored tiles. Your character is the **white square with a red arrow**
showing which direction you're facing.

#### The two phases

| Phase | What you see | What you do |
|-------|-------------|-------------|
| **Phase 1 — Explore** 🔍 | Grid + your character (no target yet) | Press action buttons freely to learn what each does. Press **"Done Exploring"** when ready. |
| **Phase 2 — Navigate** 🎯 | A **cyan-bordered target cell** appears | Use what you learned to reach the target as efficiently as possible. |

#### Understanding feedback

After each action, you'll see feedback text below the grid:
- **Position moved** → the action moved you somewhere
- **Position unchanged (wall or no-op)** → you hit a wall, or the action doesn't move
- **Now facing: up/right/down/left** → your direction changed
- **Color changed** → your character's color changed (matters in some rounds!)

#### Controls (right panel)
- **Action 0–3 buttons** — press to take that action
- **"Done Exploring" button** — end Phase 1 and reveal the target

#### Tips
- In Phase 1, try each action at least once to learn what it does.
- Pay attention to **conditions** — some actions behave differently depending on your color or facing direction.
- You can end Phase 1 early once you feel confident about all actions.
""",
        "tutorial.practice_label": "Practice environment below — try clicking the action buttons:",
        "tutorial.start_experiment": "Start Experiment →",

        # Experiment
        "experiment.progress": "Round {idx}/{total} — {family}",
        "experiment.phase1": "🔍 Phase 1: Explore — Discover what each action does",
        "experiment.phase2": "🎯 Phase 2: Navigate — Reach the cyan target!",
        "experiment.actions": "Actions",
        "experiment.done_exploring": "✅ Done Exploring",
        "experiment.next_round": "Next Round →",
        "experiment.success_msg": "✅ Target reached in {steps} steps!",
        "experiment.fail_msg": "❌ Ran out of steps ({steps} steps used).",
        "experiment.p1_steps": "Phase 1 Steps",
        "experiment.p2_steps": "Phase 2 Steps",
        "experiment.duration": "Duration",

        # Done
        "done.title": "Study Complete!",
        "done.thank_you": "### Thank you for participating!",
        "done.rounds_completed": "Rounds completed",
        "done.success_rate_label": "Success rate",
        "done.total_time": "Total time",
        "done.data_saved": "Your data has been saved.",
        "done.complete_button": "Complete on Prolific",

        # Game UI
        "game.step_label": "Step {step}",
        "game.timer": "{elapsed:.0f}s",
    },

    "zh": {
        # App
        "app.title": "AlienBody 人类实验",
        "app.icon": "🧑‍🚀",
        "lang.label": "Language / 语言",
        "lang.en": "English",
        "lang.zh": "中文",

        # Sidebar
        "sidebar.participant": "参与者",
        "sidebar.progress": "进度",
        "sidebar.success_rate": "成功率",
        "sidebar.rounds_done": "轮",
        "sidebar.language": "语言",

        # Consent
        "consent.title": "欢迎！",
        "consent.intro": """
### 实验内容

你将在一个方格世界中控制一个角色。每一轮分为两个阶段：

1. **阶段一（探索）：** 你会看到几个未标记的按钮（Action 0–3）。
   按下它们，摸索每个按钮的作用。
2. **阶段二（导航）：** 一个青色目标会出现。
   用你刚刚摸索出的规律，导航到目标位置。

**每轮约 1–2 分钟。共 30 轮（约 30–45 分钟）。**

你的数据将被匿名化，仅用于学术研究。
""",
        "consent.age": "年龄",
        "consent.gender": "性别",
        "consent.gender.male": "男",
        "consent.gender.female": "女",
        "consent.gender.nb": "非二元",
        "consent.gender.na": "不愿透露",
        "consent.gaming": "游戏经验",
        "consent.gaming.none": "无",
        "consent.gaming.casual": "休闲玩家",
        "consent.gaming.regular": "经常玩",
        "consent.gaming.expert": "资深玩家",
        "consent.puzzle": "解谜/逻辑游戏经验",
        "consent.puzzle.none": "无",
        "consent.puzzle.some": "有一些",
        "consent.puzzle.frequent": "经常玩",
        "consent.checkbox": "我同意参与本实验，并了解我的数据将被匿名化处理。",
        "consent.submit": "开始教程 →",

        # Tutorial
        "tutorial.title": "教程 — 熟悉界面",
        "tutorial.instructions": """
### 操作说明

你会看到一个带彩色方块的**网格**。你的角色是**白色方块+红色箭头**，箭头指示你的朝向。

#### 两个阶段

| 阶段 | 你会看到 | 你要做什么 |
|------|---------|-----------|
| **阶段一 — 探索** 🔍 | 网格 + 你的角色（还没有目标） | 自由按下 Action 按钮，摸索每个按钮的效果。确认了解所有按钮后，按 **"完成探索"**。 |
| **阶段二 — 导航** 🎯 | 出现一个**青色边框的目标格** | 用你学到的规律，用尽可能少的步数到达目标。 |

#### 如何理解反馈

每次按下按钮后，网格下方会显示反馈文字：
- **Position moved** → 你的位置移动了
- **Position unchanged (wall or no-op)** → 撞墙了，或者这个动作不产生移动
- **Now facing: up/right/down/left** → 你的朝向改变了
- **Color changed** → 你的角色颜色变了（在某些关卡里很重要！）

#### 操作区（右侧面板）
- **Action 0–3 按钮** — 按下来执行对应动作
- **"完成探索"按钮** — 结束阶段一，显示目标位置

#### 小提示
- 在阶段一中，**每个按钮至少试一次**，了解它们各自的作用。
- 注意**条件效果**——有些动作在不同颜色或朝向下会有不同效果。
- 确认了解所有按钮后，可以提前结束阶段一（不用用完所有步数）。
""",
        "tutorial.practice_label": "下方是练习环境——试试按 Action 按钮：",
        "tutorial.start_experiment": "开始正式实验 →",

        # Experiment
        "experiment.progress": "第 {idx}/{total} 轮 — {family}",
        "experiment.phase1": "🔍 阶段一：探索 — 摸索每个按钮的作用",
        "experiment.phase2": "🎯 阶段二：导航 — 到达青色目标！",
        "experiment.actions": "操作",
        "experiment.done_exploring": "✅ 完成探索",
        "experiment.next_round": "下一轮 →",
        "experiment.success_msg": "✅ 成功到达目标！用了 {steps} 步",
        "experiment.fail_msg": "❌ 步数用完了（共 {steps} 步）",
        "experiment.p1_steps": "阶段一步数",
        "experiment.p2_steps": "阶段二步数",
        "experiment.duration": "用时",

        # Done
        "done.title": "实验完成！",
        "done.thank_you": "### 感谢你的参与！",
        "done.rounds_completed": "完成轮数",
        "done.success_rate_label": "成功率",
        "done.total_time": "总用时",
        "done.data_saved": "你的数据已保存。",
        "done.complete_button": "在 Prolific 上完成",

        # Game UI
        "game.step_label": "第 {step} 步",
        "game.timer": "{elapsed:.0f}秒",
    },
}

# Chinese family names
FAMILY_NAMES_ZH = {
    1: "重映射",
    2: "方向型",
    3: "状态依赖",
    4: "关系型",
    5: "复合型",
    6: "时序型",
}


def t(key: str, **kwargs) -> str:
    """Look up translated string for current language."""
    lang = st.session_state.get("lang", "en")
    text = T.get(lang, T["en"]).get(key, T["en"].get(key, key))
    if kwargs:
        return text.format(**kwargs)
    return text


# ── Configuration ──────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
ENVS_DIR = DATA_DIR / "envs"
HUMAN_DATA_DIR = DATA_DIR / "human_study"
ENVS_PER_FAMILY = 5   # 5 envs per family = 30 total per participant
PROLIFIC_URL = "https://app.prolific.com/submissions/complete?cc=XXXXXXXX"  # TODO: replace with real code

# All 6 families for full benchmark
ALL_FAMILIES = [1, 2, 3, 4, 5, 6]

FAMILY_NAMES = {
    1: "Remapped",
    2: "Directional",
    3: "State-Dependent",
    4: "Relational",
    5: "Compositional",
    6: "Temporal",
}

def _get_family_display(family: int) -> str:
    """Get family name in current language."""
    lang = st.session_state.get("lang", "en")
    if lang == "zh":
        return FAMILY_NAMES_ZH.get(family, FAMILY_NAMES.get(family, f"Family {family}"))
    return FAMILY_NAMES.get(family, f"Family {family}")


# ── Session State ──────────────────────────────────────────────────

def init_session():
    """Initialize session state for a new participant."""
    defaults = {
        "initialized": True,
        "participant_id": str(uuid.uuid4())[:8],
        "phase": "consent",          # consent → tutorial → experiment → done
        "env_queue": [],
        "current_env_idx": 0,
        "env": None,
        "obs": None,
        "info": None,
        "env_start_time": None,
        "trajectories": [],           # completed trajectories
        "demographics": None,
        "lang": "en",                 # default language
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def load_env_queue():
    """Load randomized environment queue for participant (all 6 families)."""
    import random
    rng = random.Random(hash(st.session_state.participant_id))
    queue = []
    for family in ALL_FAMILIES:
        env_dir = ENVS_DIR / f"family{family}" / "test"
        if not env_dir.exists():
            continue
        files = sorted(env_dir.glob("env_*.json"))
        selected = rng.sample(files, min(ENVS_PER_FAMILY, len(files)))
        for f in selected:
            queue.append((family, f))
    rng.shuffle(queue)
    return queue


# ── Data Persistence ───────────────────────────────────────────────

def save_progress():
    """Save current progress to JSON (for resume)."""
    progress = {
        "participant_id": st.session_state.participant_id,
        "demographics": st.session_state.demographics,
        "current_env_idx": st.session_state.current_env_idx,
        "trajectories": st.session_state.trajectories,
        "lang": st.session_state.lang,
        "timestamp": datetime.now().isoformat(),
    }
    HUMAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = HUMAN_DATA_DIR / f"progress_{st.session_state.participant_id}.json"
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def save_all_trajectories():
    """Save all completed trajectories to JSONL."""
    if not st.session_state.trajectories:
        return
    HUMAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pid = st.session_state.participant_id
    path = HUMAN_DATA_DIR / f"participant_{pid}.jsonl"

    with TrajectoryWriter(path) as writer:
        for traj in st.session_state.trajectories:
            writer.write(
                traj,
                participant_id=pid,
                demographics=st.session_state.demographics or {},
            )


# ── Pages ──────────────────────────────────────────────────────────

def page_consent():
    """Consent and demographics form."""
    st.title(t("app.title"))
    st.markdown(t("consent.intro"))

    with st.form("consent"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input(t("consent.age"), 18, 80, 25)
            gender = st.selectbox(
                t("consent.gender"),
                [
                    t("consent.gender.male"),
                    t("consent.gender.female"),
                    t("consent.gender.nb"),
                    t("consent.gender.na"),
                ],
            )
        with col2:
            gaming = st.selectbox(
                t("consent.gaming"),
                [
                    t("consent.gaming.none"),
                    t("consent.gaming.casual"),
                    t("consent.gaming.regular"),
                    t("consent.gaming.expert"),
                ],
            )
            puzzle = st.selectbox(
                t("consent.puzzle"),
                [
                    t("consent.puzzle.none"),
                    t("consent.puzzle.some"),
                    t("consent.puzzle.frequent"),
                ],
            )

        consent = st.checkbox(t("consent.checkbox"))
        submitted = st.form_submit_button(t("consent.submit"), type="primary")

        if submitted and consent:
            st.session_state.demographics = {
                "age": age,
                "gender": gender,
                "gaming": gaming,
                "puzzle": puzzle,
                "participant_id": st.session_state.participant_id,
                "timestamp": datetime.now().isoformat(),
            }
            st.session_state.phase = "tutorial"
            save_progress()
            st.rerun()


def page_tutorial():
    """Practice environment with enhanced interface guidance."""
    st.title(t("tutorial.title"))
    st.markdown(t("tutorial.instructions"))

    # Load practice env
    if st.session_state.env is None:
        practice_files = list((ENVS_DIR / "family1" / "dev").glob("env_000.json"))
        if practice_files:
            config = EnvConfig.from_file(str(practice_files[0]))
            st.session_state.env = AlienBodyEnv(config, render_mode="image")
            st.session_state.obs, st.session_state.info = (
                st.session_state.env.reset()
            )
            st.session_state.env_start_time = time.time()

    st.markdown(f"**{t('tutorial.practice_label')}**")
    _render_game_ui(is_tutorial=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(t("tutorial.start_experiment"), type="primary", use_container_width=True):
            st.session_state.env = None
            st.session_state.obs = None
            st.session_state.env_queue = load_env_queue()
            st.session_state.current_env_idx = 0
            st.session_state.trajectories = []
            st.session_state.phase = "experiment"
            save_progress()
            st.rerun()


def page_experiment():
    """Main experiment loop."""
    queue = st.session_state.env_queue
    idx = st.session_state.current_env_idx

    if idx >= len(queue):
        # All done
        save_all_trajectories()
        st.session_state.phase = "done"
        st.rerun()
        return

    family, env_file = queue[idx]
    family_display = _get_family_display(family)

    # Progress bar
    progress_pct = idx / len(queue)
    st.progress(progress_pct, text=t("experiment.progress", idx=idx + 1, total=len(queue), family=family_display))

    # Load next environment if needed
    if st.session_state.env is None:
        config = EnvConfig.from_file(str(env_file))
        st.session_state.env = AlienBodyEnv(config, render_mode="image")
        st.session_state.obs, st.session_state.info = (
            st.session_state.env.reset()
        )
        st.session_state.env_start_time = time.time()

    _render_game_ui(is_tutorial=False)

    # Check if episode is done
    if st.session_state.env.done:
        _on_episode_done()


def page_done():
    """Completion page."""
    st.title(t("done.title"))
    st.balloons()

    trajectories = st.session_state.trajectories
    n_success = sum(1 for t in trajectories if t.get("success", False))
    total_time = sum(
        t.get("_duration_seconds", 0) for t in trajectories
    )

    st.markdown(t("done.thank_you"))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(t("done.rounds_completed"), f"{len(trajectories)}/30")
    with col2:
        pct = n_success / len(trajectories) * 100 if trajectories else 0
        st.metric(t("done.success_rate_label"), f"{pct:.0f}%")
    with col3:
        st.metric(t("done.total_time"), f"{total_time/60:.1f} min")

    st.markdown(t("done.data_saved"))

    # Save final data
    save_all_trajectories()

    st.link_button(t("done.complete_button"), PROLIFIC_URL)


# ── Game UI ────────────────────────────────────────────────────────

def _render_game_ui(is_tutorial: bool = False):
    """Render the grid and action buttons."""
    env = st.session_state.env
    obs = st.session_state.obs
    if env is None or obs is None:
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        # Display grid image
        if "image" in obs:
            img = obs["image"]
            if isinstance(img, np.ndarray):
                st.image(img, width=512,
                         caption=t("game.step_label", step=obs.get("step", 0)))
            else:
                st.image(img, width=512)

        # Phase indicator
        phase = obs.get("phase", 1)
        phase_label = (
            t("experiment.phase1") if phase == 1
            else t("experiment.phase2")
        )
        if phase == 1:
            st.info(phase_label)
        else:
            st.warning(phase_label)

        # Feedback from last action (more prominent)
        feedback = obs.get("feedback", "")
        if feedback:
            st.code(feedback)

    with col2:
        st.subheader(t("experiment.actions"))
        step_key = f"{obs.get('step', 0)}_{phase}"

        if not env.done:
            n_actions = env.n_actions if hasattr(env, 'n_actions') else 4
            for action in range(n_actions):
                # Action label: keep "Action N" in both languages — matches AI prompt convention
                label = f"Action {action}"
                if st.button(
                    label,
                    key=f"act_{action}_{step_key}",
                    use_container_width=True,
                ):
                    _take_action(action)

            st.divider()

            if phase == 1:
                if st.button(
                    t("experiment.done_exploring"),
                    key=f"done_{step_key}",
                    type="secondary",
                    use_container_width=True,
                ):
                    obs_new = env.start_phase2()
                    st.session_state.obs = obs_new
                    st.session_state.info = env._get_info()
                    st.rerun()

        # Timer — subtle, below actions, non-pressuring
        if st.session_state.env_start_time:
            elapsed = time.time() - st.session_state.env_start_time
            st.caption(f"⏱ {t('game.timer', elapsed=elapsed)}")


def _take_action(action: int):
    """Execute an action and update state."""
    env = st.session_state.env
    obs, reward, terminated, truncated, info = env.step(action)
    st.session_state.obs = obs
    st.session_state.info = info
    st.rerun()


def _on_episode_done():
    """Handle episode completion."""
    env = st.session_state.env
    traj = env.get_trajectory()

    # Add timing
    if st.session_state.env_start_time:
        traj["_duration_seconds"] = time.time() - st.session_state.env_start_time

    # Add family info
    traj["family_name"] = FAMILY_NAMES.get(traj.get("family", 1), "Unknown")

    success = traj.get("success", False)
    if success:
        st.success(t("experiment.success_msg", steps=traj["total_steps"]))
    else:
        st.error(t("experiment.fail_msg", steps=traj["total_steps"]))

    # Show stats
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(t("experiment.p1_steps"), traj.get("phase1_steps", 0))
    with col_b:
        st.metric(t("experiment.p2_steps"), traj.get("phase2_steps", 0))
    with col_c:
        st.metric(t("experiment.duration"), f"{traj.get('_duration_seconds', 0):.0f}s")

    # Save trajectory
    st.session_state.trajectories.append(traj)
    save_progress()

    # Next button
    if st.button(t("experiment.next_round"), type="primary", use_container_width=True):
        st.session_state.current_env_idx += 1
        st.session_state.env = None
        st.session_state.obs = None
        st.session_state.info = None
        st.session_state.env_start_time = None
        st.rerun()


# ── Main ───────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="AlienBody Human Study",
        page_icon="🧑‍🚀",
        layout="wide",
    )
    init_session()

    phase = st.session_state.phase

    # Sidebar: participant info + language toggle
    with st.sidebar:
        # Language switch at top
        st.markdown(f"**{t('lang.label')}**")
        current_lang = st.session_state.lang
        lang_options = [t("lang.en"), t("lang.zh")]
        default_idx = 0 if current_lang == "en" else 1

        lang_choice = st.radio(
            "lang_selector",
            options=lang_options,
            index=default_idx,
            label_visibility="collapsed",
            key="lang_radio",
        )
        # Map display text back to lang code
        new_lang = "en" if lang_choice == t("lang.en") else "zh"
        if new_lang != st.session_state.lang:
            st.session_state.lang = new_lang
            st.rerun()

        st.divider()

        st.markdown(f"**{t('sidebar.participant')}:** `{st.session_state.participant_id}`")
        if st.session_state.phase == "experiment":
            st.markdown(f"**{t('sidebar.progress')}:** {st.session_state.current_env_idx}/30 {t('sidebar.rounds_done')}")
            n_success = sum(
                1 for t in st.session_state.trajectories if t.get("success", False)
            )
            n_done = len(st.session_state.trajectories)
            if n_done > 0:
                st.markdown(f"**{t('sidebar.success_rate')}:** {n_success}/{n_done}")

    if phase == "consent":
        page_consent()
    elif phase == "tutorial":
        page_tutorial()
    elif phase == "experiment":
        page_experiment()
    elif phase == "done":
        page_done()


if __name__ == "__main__":
    main()
