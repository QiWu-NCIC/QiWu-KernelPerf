# CSR5 PR #13 复测报告

## 代码版本

CSR5 使用上游 [pull request #13](https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5/pull/13)
的修正版，固定提交为 `caff9d80665052bd5379fb37fd135f0d72d7b525`。该版本在原有 CUDA 修复的基础上，为
`anonymouslibHandle::spmv` 增加可选 `cudaStream_t`，并让 CSR5 的计算、校准和尾部 kernel 使用调用者的
stream。QiWu adapter 只负责复制 CSR 输入（CSR5 会原地转换列索引和值）、生命周期管理和接口适配，不修改
CSR5 算法本身。

## 测试口径

- 数据集：`suitesparse_sample_100`，100 个实数 SuiteSparse 矩阵。
- 算子：CSR SpMV，FP32 和 FP64 分开测试。
- 每个矩阵：5 次预热、20 次 CUDA-event 计时迭代；预处理时间单独记录。
- 正确性：使用 CPU 高精度 CSR 参考值，按实际输出存储精度采用动态行误差界限。
- 结果：每个平台 100 个 FP32 + 100 个 FP64，共 200 个矩阵案例。

## 平台结果

| 平台 | FP32 | FP64 | 结果文件 |
| --- | ---: | ---: | --- |
| A100-SXM4-80GB | 100/100 pass | 100/100 pass | `databank/public/data/results/spmv/A100-SXM4-80GB/suitesparse_sample_100/` |
| H100-SXM5-80GB | 100/100 pass | 100/100 pass | `databank/public/data/results/spmv/H100-SXM5-80GB/suitesparse_sample_100/` |
| RTX5090-SL3061 | 100/100 pass | 100/100 pass | `databank/public/data/results/spmv/RTX5090-SL3061/suitesparse_sample_100/` |

六个 CSV 均为 100 行、100 个唯一矩阵、无失败记录。结果索引和 candidate-pool 已同步更新，全部引用当前
CSR5 插件内容哈希及 PR #13 版本信息。

## H100 环境说明

一次复测在 H100 节点上出现单矩阵 1200 秒超时。诊断显示该节点同时运行模型服务/MPS，GPU 显存和计算资源
被外部进程占用；同一二进制在无占用节点上完成小矩阵诊断，并最终完成全量 200/200。因此最终发布数据只
采用无外部 GPU 进程的完整运行，不保留超时批次。

## 验证命令

```bash
cd KernelPerf
python -m pytest -q
python scripts/validate_submissions.py submissions

cd ../databank
npm run audit:spmv
npm run build
```

本次审计结果为 `312 public submissions`、三个后端各 `104` 条、`failures: []`。
