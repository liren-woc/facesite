from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pipeline, save_feedback
from .reference_library import summarize_reference_library


APP_CSS = """
:root {
  --page-bg: linear-gradient(135deg, #f7efe5 0%, #fcfaf6 50%, #e6efe8 100%);
  --panel-bg: rgba(255, 255, 255, 0.92);
  --panel-border: rgba(48, 38, 28, 0.08);
  --title: #201813;
  --text: #4d443d;
  --muted: #7b6f64;
  --accent: #9b5a33;
  --accent-soft: #f4e4d8;
}

.gradio-container {
  background: var(--page-bg);
  color: var(--text);
  font-family: "Microsoft YaHei UI", "PingFang SC", "Segoe UI", sans-serif;
}

.page-shell {
  max-width: 1260px;
  margin: 0 auto;
  padding: 18px 10px 40px;
}

.hero,
.panel {
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
  border-radius: 28px;
  box-shadow: 0 18px 42px rgba(32, 24, 19, 0.08);
  backdrop-filter: blur(16px);
}

.hero {
  padding: 30px;
  margin-bottom: 18px;
  background:
    radial-gradient(circle at top right, rgba(155, 90, 51, 0.14), transparent 28%),
    radial-gradient(circle at bottom left, rgba(68, 110, 88, 0.12), transparent 30%),
    var(--panel-bg);
}

.hero-badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(155, 90, 51, 0.1);
  color: var(--accent);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.08em;
}

.hero h1 {
  margin: 14px 0 10px;
  color: var(--title);
  font-size: 34px;
  line-height: 1.08;
  font-weight: 800;
}

.hero p {
  margin: 0;
  max-width: 920px;
  color: var(--text);
  line-height: 1.82;
  font-size: 15px;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 18px;
}

.metric {
  padding: 14px 16px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(48, 38, 28, 0.06);
}

.metric strong {
  display: block;
  margin-bottom: 6px;
  color: var(--title);
  font-size: 14px;
}

.metric span {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.65;
}

.panel {
  padding: 18px;
}

.panel h2 {
  margin: 0 0 6px;
  color: var(--title);
  font-size: 22px;
  font-weight: 800;
}

.panel p {
  margin: 0 0 16px;
  color: var(--muted);
  font-size: 14px;
  line-height: 1.75;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

.chip {
  padding: 7px 12px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--title);
  font-size: 12px;
  font-weight: 700;
}

#run-btn {
  min-height: 56px;
  border-radius: 18px;
  font-size: 16px;
  font-weight: 800;
  background: linear-gradient(135deg, var(--accent) 0%, #cb7b44 100%);
  border: none;
  box-shadow: 0 14px 24px rgba(155, 90, 51, 0.2);
}

#run-btn:hover {
  filter: brightness(1.03);
}

@media (max-width: 900px) {
  .metric-grid {
    grid-template-columns: 1fr;
  }

  .hero h1 {
    font-size: 28px;
  }
}
"""


FACE_SHAPE_LABELS = {
    "oval": "椭圆脸",
    "round": "圆脸",
    "square": "方脸",
    "heart": "心形脸",
    "oblong": "长脸",
    "diamond": "菱形脸",
    "unknown": "待确认",
}

HAIRLINE_LABELS = {
    "low": "发际线偏低",
    "balanced": "发际线均衡",
    "high": "发际线偏高",
    "unknown": "待确认",
}

VIEW_LABELS = {
    "front": "正脸",
    "left": "左侧脸",
    "right": "右侧脸",
    "hairline": "发际线特写",
    "crown": "头顶/后脑辅助图",
}

SAME_PERSON_LABELS = {
    "pass": "通过",
    "review_needed": "需要复核",
    "insufficient_data": "信息不足",
    "unknown": "未知",
}

PRESENTATION_LABELS = {
    "masculine": "男性风格",
    "feminine": "女性风格",
    "any": "不限",
}

MAINTENANCE_LABELS = {
    "low": "省事好打理",
    "medium": "适中",
    "high": "愿意花时间打理",
    "any": "不限",
}

AGE_GROUP_LABELS = {
    "teen": "青少年",
    "young_adult": "青年",
    "adult": "成年",
    "middle_aged": "中年",
    "senior": "老年",
    "any": "不限",
}

FOREHEAD_GOAL_LABELS = {
    "auto": "系统自动平衡",
    "cover": "优先遮额修饰发际线",
    "balance": "额头比例自然平衡",
    "open": "偏利落露额",
}

STYLE_TAG_LABELS = {
    "any": "不限",
    "clean": "干净利落",
    "natural": "自然松弛",
    "workplace": "通勤职场",
    "stable": "稳重成熟",
    "rejuvenating": "减龄",
    "soft": "柔和修饰",
    "student": "学生感",
    "fresh": "清爽",
    "mature": "成熟气质",
    "cute": "甜感",
    "mainstream": "中国主流潮流",
    "korean": "韩系补充",
}


def _uploaded_path(uploaded) -> str | None:
    if uploaded is None:
        return None
    if isinstance(uploaded, str):
        return uploaded
    return getattr(uploaded, "name", None)


def _label(mapping: dict[str, str], value: str | None) -> str:
    if value is None:
        return "未设置"
    return mapping.get(value, value)


def _format_summary(payload: dict) -> str:
    analysis = payload.get("analysis", {})
    generation = payload.get("generation", {})
    same_person = analysis.get("same_person_verification", {})
    missing = analysis.get("missing_required_views", [])
    session_metrics = analysis.get("session_metrics", {})
    score = same_person.get("score")
    score_text = "无" if score is None else f"{score:.3f}"

    lines = [
        f"会话目录：{payload.get('session', {}).get('session_dir', '未生成')}",
        f"主判定脸型：{_label(FACE_SHAPE_LABELS, str(analysis.get('dominant_face_shape_hint', 'unknown')))}",
        f"主判定发际线：{_label(HAIRLINE_LABELS, str(analysis.get('dominant_hairline_height_hint', 'unknown')))}",
        f"风格池：{_label(PRESENTATION_LABELS, str(payload.get('presentation_preference', 'any')))}",
        f"打理偏好：{_label(MAINTENANCE_LABELS, str(payload.get('maintenance_preference', 'any')))}",
        f"年龄阶段：{_label(AGE_GROUP_LABELS, str(payload.get('age_group', 'any')))}",
        f"额头策略：{_label(FOREHEAD_GOAL_LABELS, str(payload.get('forehead_goal', 'auto')))}",
        f"风格倾向：{_label(STYLE_TAG_LABELS, str(payload.get('preferred_style_tag', 'any')))}",
        f"缺失视角：{'、'.join(_label(VIEW_LABELS, item) for item in missing) if missing else '无'}",
        f"同一人一致性校验：{_label(SAME_PERSON_LABELS, str(same_person.get('status', 'unknown')))}（评分：{score_text}）",
        f"当前生成后端：{generation.get('backend', 'unknown')}",
        f"自动选中发型：{generation.get('selected_style_name') or '尚未选中'}",
        f"生成状态：{generation.get('status', 'unknown')}",
    ]
    if generation.get("fallback_from"):
        lines.append(
            f"回退说明：主后端 {generation.get('fallback_from')} 失败，已自动切到 {generation.get('backend')}"
        )
    if session_metrics:
        lines.append(
            "平均几何指标："
            f"脸长宽比 {session_metrics.get('mean_face_ratio_h_w', '无')}，"
            f"额头占脸比例 {session_metrics.get('mean_forehead_to_face_ratio', '无')}"
        )
    if generation.get("message"):
        lines.append(f"说明：{generation['message']}")
    return "\n".join(lines)


def _feedback_choices(payload: dict) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for index, item in enumerate(payload.get("recommendations", []), start=1):
        style_id = str(item.get("style_id", ""))
        name = str(item.get("name", style_id))
        if not style_id:
            continue
        choices.append((f"Top {index} · {name}", style_id))
    return choices


def _format_recommendations(payload: dict) -> str:
    items = payload.get("recommendations", [])
    generation = payload.get("generation", {})
    selected_style_id = generation.get("selected_style_id")
    if not items:
        return "暂无推荐结果。请先上传至少一张可识别的人脸照片。"

    blocks: list[str] = []
    for index, item in enumerate(items, start=1):
        reasons = item.get("reasons") or []
        reasons_text = "；".join(str(reason) for reason in reasons) if reasons else "当前规则库暂无额外说明。"
        selected_tag = " `当前用于生成`" if item.get("style_id") == selected_style_id else ""
        presentation = _label(PRESENTATION_LABELS, str(item.get("presentation", "any")))
        maintenance = _label(MAINTENANCE_LABELS, str(item.get("maintenance_level", "medium")))
        style_tags = item.get("style_tags") or []
        style_tag_text = "、".join(_label(STYLE_TAG_LABELS, str(tag)) for tag in style_tags) if style_tags else "未标注"
        blocks.append(
            f"### Top {index} · {item.get('name', item.get('style_id', '发型方案'))}{selected_tag}\n"
            f"- 适配分：`{item.get('score', 0):.2f}`\n"
            f"- 风格归属：`{presentation}`\n"
            f"- 打理强度：`{maintenance}`\n"
            f"- 风格标签：{style_tag_text}\n"
            f"- 推荐依据：{reasons_text}"
        )
    return "\n\n".join(blocks)


def build_demo(
    *,
    generator_backend: str,
    generator_repo: str,
    output_dir: str,
    catalog_path: str,
    generator_python: str | None = None,
):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("请先安装 gradio：pip install -r requirements-core.txt") from exc

    output_dir_path = Path(output_dir)
    library_status = summarize_reference_library(catalog_path)

    def run_session(
        front_file,
        left_file,
        right_file,
        hairline_file,
        crown_file,
        user_label,
        user_notes,
        presentation_preference,
        maintenance_preference,
        age_group,
        forehead_goal,
        preferred_style_tag,
    ):
        front_path = _uploaded_path(front_file)
        if front_path is None:
            return (
                None,
                "请至少上传一张正脸照片。",
                "暂无推荐结果。请先上传完整的人脸照片后再分析。",
                "{}",
                "",
                gr.update(choices=[], value=None),
                "等待本次分析完成后再保存反馈。",
            )

        try:
            payload = run_pipeline(
                front_path=front_path,
                left_path=_uploaded_path(left_file),
                right_path=_uploaded_path(right_file),
                hairline_path=_uploaded_path(hairline_file),
                crown_path=_uploaded_path(crown_file),
                catalog_path=catalog_path,
                generator_backend=generator_backend,
                generator_repo=generator_repo,
                generator_python=generator_python,
                output_dir=output_dir_path,
                skip_generation=False,
                generate_side_views=False,
                top_k=5,
                presentation_preference=presentation_preference,
                maintenance_preference=maintenance_preference,
                age_group=age_group,
                forehead_goal=forehead_goal,
                preferred_style_tag=preferred_style_tag,
                session_label=user_label or "manual_run",
                session_notes=user_notes,
            )
        except Exception as exc:
            error_payload = {
                "status": "error",
                "message": str(exc),
            }
            return (
                None,
                f"本次分析失败：{exc}",
                "暂无推荐结果。请先检查图片、模型环境或参考库状态。",
                json.dumps(error_payload, ensure_ascii=False, indent=2),
                "",
                gr.update(choices=[], value=None),
                f"运行失败：{exc}",
            )

        generated = payload.get("generation", {}).get("generated_views", {})
        feedback_choices = _feedback_choices(payload)
        return (
            generated.get("front"),
            _format_summary(payload),
            _format_recommendations(payload),
            json.dumps(payload, ensure_ascii=False, indent=2),
            payload.get("session", {}).get("session_dir", ""),
            gr.update(
                choices=feedback_choices,
                value=feedback_choices[0][1] if feedback_choices else None,
            ),
            "本次会话已保存。你可以直接对推荐结果做正负反馈。",
        )

    def save_feedback_from_ui(raw_json_text, selected_style_id, feedback_label, feedback_note):
        if not raw_json_text.strip():
            return "当前没有可保存的会话结果。"
        if not selected_style_id:
            return "请先选择要反馈的发型。"
        try:
            payload = json.loads(raw_json_text)
            label = 1 if feedback_label == "positive" else 0
            result = save_feedback(
                payload=payload,
                style_id=selected_style_id,
                label=label,
                note=feedback_note,
            )
        except Exception as exc:
            return f"保存反馈失败：{exc}"

        label_text = "适合" if label == 1 else "不适合"
        return f"已保存反馈：{selected_style_id} -> {label_text}。CSV：{result['feedback_csv']}"

    with gr.Blocks(title="中国审美发型设计系统") as demo:
        with gr.Column(elem_classes=["page-shell"]):
            gr.HTML(
                f"""
                <section class="hero">
                  <span class="hero-badge">CHINA-FIRST HAIR DESIGN</span>
                  <h1>中国审美优先的人脸发型设计系统</h1>
                  <p>
                    你上传同一个人的正脸、左侧脸、右侧脸和发际线照片后，系统会先做脸型、发际线、
                    额头比例和多视角一致性分析，再按中国主流潮流发型优先、韩系作为补充的参考库做推荐。
                    当前版本先把“推荐方向准、正面融合自然”做好，默认只生成一张正面效果图。
                    当前版本统一使用 Stable-Hair 作为主生成后端，重点先攻克更真实的头骨贴合和发际线融合。
                  </p>
                  <div class="metric-grid">
                    <div class="metric">
                      <strong>已审核参考图</strong>
                      <span>当前已接入 {library_status.vetted_styles} 款可直接用于生成的中国审美参考图。</span>
                    </div>
                    <div class="metric">
                      <strong>男性风格可用</strong>
                      <span>{library_status.vetted_masculine} 款，优先覆盖三七分、短碎盖等常用方向。</span>
                    </div>
                    <div class="metric">
                      <strong>女性风格可用</strong>
                      <span>{library_status.vetted_feminine} 款，优先覆盖短波波和柔和修饰层次。</span>
                    </div>
                    <div class="metric">
                      <strong>待补参考</strong>
                      <span>还有 {library_status.pending_styles} 款待补，重点是八字刘海和轻纹理碎盖。</span>
                    </div>
                  </div>
                </section>
                """
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=7):
                    with gr.Group(elem_classes=["panel"]):
                        gr.HTML(
                            """
                            <h2>上传照片</h2>
                            <p>
                              建议上传同一个人在相近光线下拍摄的清晰彩色照片。发际线特写越清楚，
                              系统对额头高低、遮额需求和不适合露额发型的判断越稳定。
                            </p>
                            """
                        )
                        with gr.Row():
                            with gr.Column():
                                front = gr.File(label="正脸照片", file_types=[".png", ".jpg", ".jpeg", ".webp"])
                                left = gr.File(label="左侧脸照片", file_types=[".png", ".jpg", ".jpeg", ".webp"])
                                right = gr.File(label="右侧脸照片", file_types=[".png", ".jpg", ".jpeg", ".webp"])
                            with gr.Column():
                                hairline = gr.File(label="发际线特写", file_types=[".png", ".jpg", ".jpeg", ".webp"])
                                crown = gr.File(label="头顶 / 后脑辅助图（可选）", file_types=[".png", ".jpg", ".jpeg", ".webp"])
                                user_label = gr.Textbox(
                                    label="用户标识 / 会话名",
                                    placeholder="例如：zhangsan 或 2026-05-01-首轮试发",
                                )
                                user_notes = gr.Textbox(
                                    label="本次备注（可选）",
                                    placeholder="例如：希望更减龄、不想太贴头皮、想修饰发际线",
                                    lines=3,
                                )

                        gr.HTML(
                            """
                            <h2 style="margin-top: 18px;">个性化目标</h2>
                            <p>
                              这里是给推荐器和后续训练模型的显式偏好。你后面给我更多个人照片和真实反馈时，
                              这套字段会逐步学成更贴近你自己的中国风格发型偏好。
                            </p>
                            """
                        )
                        with gr.Row():
                            presentation = gr.Radio(
                                choices=[("男性风格", "masculine"), ("女性风格", "feminine"), ("不限", "any")],
                                value="any",
                                label="风格池",
                                info="先限制推荐池，避免明显不符的风格混入。",
                            )
                            maintenance = gr.Radio(
                                choices=[("省事好打理", "low"), ("适中", "medium"), ("愿意花时间打理", "high"), ("不限", "any")],
                                value="any",
                                label="打理偏好",
                                info="决定更偏自然稳妥，还是更精致更需要打理的发型。",
                            )
                        with gr.Row():
                            age_group = gr.Dropdown(
                                choices=[
                                    ("不限", "any"),
                                    ("青少年", "teen"),
                                    ("青年", "young_adult"),
                                    ("成年", "adult"),
                                    ("中年", "middle_aged"),
                                    ("老年", "senior"),
                                ],
                                value="any",
                                label="年龄阶段",
                                info="这是个性化推荐的重要条件，用来避免把明显过年轻或过老气的款式硬套到本人身上。",
                            )
                        with gr.Row():
                            forehead = gr.Radio(
                                choices=[
                                    ("系统自动平衡", "auto"),
                                    ("优先遮额修饰发际线", "cover"),
                                    ("额头比例自然平衡", "balance"),
                                    ("偏利落露额", "open"),
                                ],
                                value="auto",
                                label="额头 / 发际线策略",
                                info="这会直接影响刘海、遮挡和额头暴露程度。",
                            )
                            style_tag = gr.Dropdown(
                                choices=[
                                    ("不限", "any"),
                                    ("中国主流潮流", "mainstream"),
                                    ("干净利落", "clean"),
                                    ("自然松弛", "natural"),
                                    ("通勤职场", "workplace"),
                                    ("稳重成熟", "stable"),
                                    ("减龄", "rejuvenating"),
                                    ("柔和修饰", "soft"),
                                    ("学生感", "student"),
                                    ("清爽", "fresh"),
                                    ("成熟气质", "mature"),
                                    ("甜感", "cute"),
                                    ("韩系补充", "korean"),
                                ],
                                value="any",
                                label="风格倾向",
                                info="用于在同等适配的发型里再做偏好排序。",
                            )

                        gr.HTML(
                            """
                            <div class="chip-row">
                              <span class="chip">建议正脸无遮挡</span>
                              <span class="chip">建议侧脸露出耳部和下颌线</span>
                              <span class="chip">发际线图尽量拍到额头和两侧额角</span>
                              <span class="chip">当前默认只生成一张正面效果图</span>
                            </div>
                            """
                        )
                        run_button = gr.Button("开始分析并生成正面效果图", variant="primary", elem_id="run-btn")

                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["panel"]):
                        gr.HTML(
                            """
                            <h2>结果概览</h2>
                            <p>
                              这里会显示主判定脸型、主判定发际线、同一人几何一致性校验、
                              当前使用的生成后端，以及本次自动选中的中国风格发型方案。
                            </p>
                            """
                        )
                        summary = gr.Textbox(label="分析摘要", lines=12)
                        recommendations = gr.Markdown("推荐结果会显示在这里。")
                        session_dir_box = gr.Textbox(label="本次输出目录", interactive=False)

            with gr.Group(elem_classes=["panel"]):
                gr.HTML(
                    """
                    <h2>正面效果图</h2>
                    <p>
                      当前版本默认只输出正面图。左侧、右侧和可旋转查看会放到下一阶段。
                      如果当前推荐库没有匹配到已审核的中国审美参考图，系统会保留分析和推荐结果，
                      但不会继续生成错误图片。
                    </p>
                    """
                )
                front_result = gr.Image(label="正面效果", type="filepath", height=560)

            with gr.Group(elem_classes=["panel"]):
                gr.HTML(
                    """
                    <h2>训练反馈</h2>
                    <p>
                      这里是把“这次适不适合你”沉淀成训练数据的入口。你每次反馈一次，
                      后面的推荐模型就更接近你自己的中国风格偏好。
                    </p>
                    """
                )
                with gr.Row():
                    feedback_style = gr.Dropdown(label="反馈哪一个发型", choices=[], value=None)
                    feedback_label = gr.Radio(
                        choices=[("适合，方向对", "positive"), ("不适合，方向不对", "negative")],
                        value="positive",
                        label="反馈结论",
                    )
                feedback_note = gr.Textbox(
                    label="反馈备注",
                    placeholder="例如：发际线还是露太多、两侧太厚、想再利落一点",
                    lines=3,
                )
                feedback_button = gr.Button("保存这次反馈", variant="secondary")
                feedback_status = gr.Textbox(label="反馈状态", interactive=False)

            with gr.Accordion("查看原始会话 JSON", open=False):
                raw_json = gr.Textbox(label="调试输出", lines=24)

            run_button.click(
                run_session,
                inputs=[
                    front,
                    left,
                    right,
                    hairline,
                    crown,
                    user_label,
                    user_notes,
                    presentation,
                    maintenance,
                    age_group,
                    forehead,
                    style_tag,
                ],
                outputs=[front_result, summary, recommendations, raw_json, session_dir_box, feedback_style, feedback_status],
            )
            feedback_button.click(
                save_feedback_from_ui,
                inputs=[raw_json, feedback_style, feedback_label, feedback_note],
                outputs=[feedback_status],
            )

    return demo


def main() -> None:
    import gradio as gr

    parser = argparse.ArgumentParser(description="运行发型设计 Gradio 中文前端。")
    parser.add_argument("--generator-backend", default="stable_hair", choices=["stable_hair", "disabled"])
    parser.add_argument("--generator-repo", default="third_party/Stable-Hair")
    parser.add_argument("--generator-python", default=None)
    parser.add_argument("--output-dir", default="outputs/tryon")
    parser.add_argument("--catalog", default="data/hairstyles/catalog.example.json")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo(
        generator_backend=args.generator_backend,
        generator_repo=args.generator_repo,
        output_dir=args.output_dir,
        catalog_path=args.catalog,
        generator_python=args.generator_python,
    )
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
