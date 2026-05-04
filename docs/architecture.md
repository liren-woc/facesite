# 项目架构

## 当前目标

输入同一用户的多角度照片，输出：

- 个性化发型分析
- 中国审美优先的发型推荐
- 基于 `Stable-Hair` 的正面效果图

## 当前技术路线

```text
多图输入
  -> 质量检查
  -> 人脸关键点 / 几何比例分析
  -> 发际线 / 额角开放程度启发式分析
  -> 个性化推荐器
  -> Stable-Hair 正面发型迁移
  -> 结果图 + JSON 分析报告
```

## 目录

```text
face_shape_project/
  configs/
    pipeline.example.yaml
    reference_pool_activation.json
  data/
    hairstyles/
      catalog.example.json
      reference_library.cn.json
      reference_candidates/
  docs/
    architecture.md
    generator_upgrade_plan.md
    reference_sources_2025_2026.md
    research.md
  scripts/
    create_stable_hair_env.sh
    setup_stable_hair_wsl.ps1
    run_stable_hair_wsl.sh
    run_generation_backend.py
  src/
    hairstyle_tryon/
      analysis.py
      pipeline.py
      recommend.py
      app.py
      reference_library.py
      feedback_store.py
      backends/
        factory.py
        stable_hair.py
```

## 当前主生成后端

- 唯一主后端：`Stable-Hair`
- `HairFastGAN` 已从项目中移除
- 当前机器的 6GB 显存不足以稳定跑完 `Stable-Hair` 第二阶段
- 建议迁移到更大显存机器继续运行

## 下一阶段

1. 补齐中国审美参考图主库
2. 接入更强的发际线 / hair parsing
3. 打通 `Stable-Hair` 高显存环境
4. 再扩展到多视角一致生成和 3D 查看
