# A100 上线前全流程测试

本文记录在 `ict-A100` 上验证 KernelPerf、PlayGround 提交接口和 databank
静态排行榜的方法。测试边界到 GitHub 提交之前：不配置 GitHub token，不向远端仓库写入数据。

## 1. 当前部署

- 部署根目录：`/home/wangyinshan/yjk/KernelPerf-online-leaderboard-20260814`
- KernelPerf 应用：`app/`
- SQLite 运行数据：`data/kernelperf.sqlite`、`data/contest.sqlite`
- 自动结果 CSV：`data/result_exports/spmv/<backend-id>/`
- 静态排行榜：`leaderboard-site/databank-alpha/`
- KernelPerf 监听：`127.0.0.1:18081`
- 静态站点监听：`127.0.0.1:18082`
- 测试数据集：`suitesparse_sample_100`，100 个矩阵

服务只监听服务器回环地址。不要把密码、GitHub token 或管理员 token 写入本文、
仓库文件、shell 历史或提交产物。

## 2. 启动与访问

在 A100 上启动 KernelPerf：

```bash
cd /home/wangyinshan/yjk/KernelPerf-online-leaderboard-20260814/app
python3 -m kernelperf.cli serve --config config/service.json
```

另一个终端启动静态站点：

```bash
cd /home/wangyinshan/yjk/KernelPerf-online-leaderboard-20260814
python3 -m http.server 18082 \
  --bind 127.0.0.1 \
  --directory leaderboard-site
```

从本地建立 SSH 隧道：

```bash
ssh -p 2133 -N \
  -L 18081:127.0.0.1:18081 \
  -L 18082:127.0.0.1:18082 \
  wangyinshan@10.208.16.50
```

访问入口：

- 结果页：`http://127.0.0.1:18081/`
- 在线提交：`http://127.0.0.1:18081/submit`
- 静态排行榜：`http://127.0.0.1:18082/databank-alpha/#/spmv`

## 3. 提交前检查

```bash
curl http://127.0.0.1:18081/api/v1/workers
curl http://127.0.0.1:18081/api/v1/queue
curl http://127.0.0.1:18081/api/v1/leaderboard/config
```

应满足：

- `worker-ict-a100-online` 在线且空闲；
- 队列没有遗留的 running 或 queued job；
- 未配置 GitHub token 时 `leaderboard.enabled` 为 `false`；
- 结果 CSV 下载可用，但 `Submit to leaderboard` 按钮禁用。

## 4. CUDA 源码测试

打开 `/submit`，填写方法名并从固定的 `base_format` 选项中选择：

`CSR`、`COO`、`ELL`、`SELL`、`HYB`、`BSR`、`DIA`、`Auto-tuned` 或 `Unmarked`。

具体实现参数（例如 SELL 的 C32 分块）写在方法名中，例如 `SELL-C32`；
基础格式标签只表示存储家族，不再使用自由文本。

- Backend：ICT A100；
- Dataset：`suitesparse_sample_100`；
- Operator：先 `spmv.csr.fp32`，再单独提交 `spmv.csr.fp64`；
- Artifact type：`CUDA source`；
- Source：实现 `preprocess`、`solve`、`destroy` 生命周期接口。

每个 dtype 都应运行 100 个矩阵。结束后检查 job 为 `succeeded`，100 个 case
全部为 `pass`。自定义 SELL 格式的转换、候选数据分配和搬运必须发生在
`preprocess` 中；`solve` 只执行 SpMV。

## 5. 预编译对象测试

使用目标架构生成 relocatable object：

```bash
nvcc -std=c++17 -arch=sm_80 -dc candidate.cu -o candidate.o
```

在 `/submit` 中选择 `nvcc relocatable .o`，上传 `.o` 并填写 `sm_80`。对象必须
导出与源码提交相同的生命周期符号。分别提交 FP32 和 FP64，并再次确认
每个 job 都是 100/100 pass。

## 6. 结果与 CSV 验收

结果页按 job、backend、suite、operator 和 method 分组。检查：

- 纵轴显示 `GFLOP/s`，数据点使用 case 的 `gflops`；
- FP32 和 FP64 使用不同颜色；
- `LOG` 可下载本组日志；
- `CSV` 通过 `/api/v1/results.csv` 直接下载服务器自动导出的对应规范文件；
- 失败或未完成的 job 不能提交排行榜。

job 进入终态且存在结果行后还应自动在服务器生成规范 CSV，无需点击网页按钮：

```text
/home/wangyinshan/yjk/KernelPerf-online-leaderboard-20260814/
  data/result_exports/spmv/ict-a100/<submission-id>.csv
```

job 日志中应出现 `exported result CSV:`。没有结果行的失败或取消 job 不生成空文件；
单个 case 的 `fail` 会写入对应终态 job 的 CSV，便于保留完整回归结果。

CSV 至少检查以下字段：

```text
schema_version,submission_id,method_id,method_name,base_format,operator_id,
dtype,backend_id,hardware,peak_gflops,job_id,dataset_id,matrix_id,
matrix_name,rows,cols,nnz,status,operations,preprocess_ms,solve_ms,
solve_gflops,solve_only_efficiency_percent,timestamp,source_kind
```

## 7. 静态排行榜验收

把待验收的规范 CSV 作为独立文件放入平台目录，并增量更新 `data/index.json`。
不要用本地构建中的 `dist/data` 覆盖服务器现有 `data/`，否则会丢失已暂存提交。

页面应满足：

- 每个方法是一条独立横栏；
- 每条横栏左侧为排名和方法信息，右上角为 `Efficiency`；
- 每条横栏内 FP32、FP64 散点图并排，纵轴均为 `GFLOP/s`；
- 排名分数仍为 `performance / dtype peak * 100%` 的几何平均；
- 总排名合并 FP32 与 FP64，但缺少任一 dtype 完整覆盖的方法不参与排名；
- `solve-only`、`pre+solve`、`pre/100+solve` 会同时重算 GFLOP/s 和 Efficiency；
- Operator、Hardware、Dataset 筛选不会把不同平台或算子的数据合并；
- `Download CSV` 导出当前筛选和计时模式下的数据。

## 8. 本轮回归记录

2026-08-17 在 ICT A100 SXM4 80GB 上完成：

| 方法 | dtype | job id | 结果 |
| --- | --- | --- | --- |
| SELL-C32 | FP32 | `8aa91710-56fb-420b-ab92-34b90073538b` | 100/100 pass |
| SELL-C32 | FP64 | `f39ebf3d-8256-4c66-bdce-2ebd0de4cab3` | 100/100 pass |
| cuSPARSE-12.2-CSR-ALG1 `.o` | FP32 | `9c8c95ab-1337-4407-91f5-a85087945576` | 100/100 pass |
| cuSPARSE-12.2-CSR-ALG1 `.o` | FP64 | `d963c6b0-cada-4bf4-a1f8-4cfac2c3b3a5` | 100/100 pass |

浏览器回归结果：400 个最新 case、2 个完整方法横栏、每条横栏 2 张散点图；
桌面和 390 px 移动端均无横向溢出，计时模式切换会重绘全部散点图。

## 9. 启用 GitHub 提交后的补充测试

只有准备实际写入 databank 分支时才设置 `KERNELPERF_GITHUB_TOKEN` 并重启服务。
确认 `/api/v1/leaderboard/config` 返回 `enabled: true` 后，点击
`Submit to leaderboard`，再验证：

1. GitHub 分支新增一个独立 CSV；
2. `public/data/index.json` 只增量追加或替换同一提交；
3. 静态站点重新构建后能加载该方法；
4. 页面下载的数据与服务器生成的规范 CSV 一致。
