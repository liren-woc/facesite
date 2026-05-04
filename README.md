# face_shape_project

面向中文审美的人脸发型分析、推荐与试发项目。

当前主生成后端已经收口为 `Stable-Hair`，`HairFastGAN` 已从项目中移除。

## 当前结构

- `src/hairstyle_tryon`
  - 多角度人脸分析
  - 个性化发型推荐
  - `Stable-Hair` 生成后端封装
- `data/hairstyles`
  - 中国审美发型库
  - 参考发型图
- `scripts`
  - 启动前端
  - 运行 pipeline
  - `Stable-Hair` 环境检查与 WSL 启动脚本

## 运行环境

- 主 Python: `D:\anaconda\envs\pytorch\python.exe`
- 生成后端: `Stable-Hair`
- 推荐的生成运行方式: `WSL Ubuntu + stablehair conda env`

需要的本地配置文件：

- `configs/stable_hair_python.txt`
  - 例如：`wsl://Ubuntu`
- `configs/stable_hair_sd15_path.txt`
  - 例如：`/home/sa/stable-hair-cache/sd15`

## 启动前端

```powershell
cd d:\face_shape_project
.\scripts\run_app.ps1
```

默认地址：

```text
http://127.0.0.1:7860
```

## 运行命令行 pipeline

```powershell
cd d:\face_shape_project
.\scripts\run_pipeline.ps1 `
  -Front "D:\images\front.jpg" `
  -Left "D:\images\left.jpg" `
  -Right "D:\images\right.jpg" `
  -Hairline "D:\images\hairline.jpg" `
  -PresentationPreference masculine `
  -AgeGroup adult
```

## 检查 Stable-Hair

```powershell
D:\anaconda\envs\pytorch\python.exe scripts\check_stable_hair_backend.py
```

## 当前已知限制

- 当前项目已经删除 `HairFastGAN` 代码和旧脚本，只保留 `Stable-Hair` 路线。
- `Stable-Hair` 在 `6GB` 显存机器上容易 OOM，建议换到更大显存机器运行。
- 当前 3D / 多视角一致生成还未完成，主目标仍是先把正面真实试发做好。
