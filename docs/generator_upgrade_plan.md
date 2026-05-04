# 生成器升级计划

## 当前决定

项目已经收口为 `Stable-Hair` 单路线：

- 主生成后端：`Stable-Hair`
- 多视角未来路线：`StableHairV2`
- `HairFastGAN` 已删除，不再保留兼容分支

## 当前阻塞

当前本机环境的主要问题不是代码，而是显存：

- 本机 GPU：`RTX 3060 Laptop 6GB`
- `Stable-Hair` 第二阶段会在本机 OOM
- 因此当前代码适合迁移到更大显存机器运行

## 推荐目标机

- 最低建议：`12GB`
- 更稳妥：`16GB`
- 如果后面要做多视角和 3D：`24GB`

## 当前项目状态

- 推荐器已经切到个性化打分
- 中国审美参考池已经接入
- 前端已经改成中文
- 命令行和 Web 入口都统一使用 `Stable-Hair`

## 新机器落地顺序

1. 克隆本仓库
2. 准备 `WSL Ubuntu`
3. 运行 `scripts/setup_stable_hair_wsl.ps1`
4. 准备 `runwayml/stable-diffusion-v1-5`
5. 配置 `configs/stable_hair_sd15_path.txt`
6. 跑 `scripts/check_stable_hair_backend.py`
7. 启动 `scripts/run_app.ps1`
