# KernelPerf

KernelPerf是一个面向GPU Kernel的多后端评测平台，用于统一和性能对比。

一次评测任务会按目标后端拆分为多个独立子任务，由对应Worker并发执行；
具体的编译、运行和结果验证逻辑由各Benchmark插件负责。

## 文件架构

当前文件架构：

```text
KernelPerf/
├── kernelperf/          # API、调度器、Worker、配置与数据存储
├── benchmarks/          # Benchmark插件，每个评测拥有独立目录和Driver
│   ├── sol_execbench/   # SOL-ExecBench
│   └── spmv/            # SpMV bench
├── config/              # Worker、Benchmark、dataset配置
├── web/                 # Web UI（纯AI）
├── scripts/             # 常用脚本
├── tests/               # 测试
├── data/                # SQLite数据库
├── DEPLOY.md            # 部署说明
└── pyproject.toml       # Python项目与依赖配置
```

## SpMV 在线评测

服务启动后提供三个入口：

- 结果与 Worker 状态页：/
- 源码或预编译对象提交页：/submit
- Contest 列表：/contest

SpMV 使用 preprocess / solve / destroy 插件接口。框架读取 Matrix Market
文件并在计时前准备 Host CSR、Device CSR 和 x/y；候选 preprocess 可以直接
使用 Device CSR，也可以从 Host/Device CSR 构建 SELL 等自定义格式。
平台输入的 Host-to-Device 搬运不计入候选 preprocess，候选格式转换、候选
数据搬运、descriptor、workspace 和库预处理均计入 preprocess。

源码 artifact 支持单文件 `source`，也支持 `source_files`、`entry_source` 和
`compile_units` 组成的多文件源码树。`entry_source` 是实现生命周期接口的 `.cu`
适配层；其余上游源码按原目录保存，只有 `compile_units` 中列出的编译单元会和
managed benchmark 一起编译。源码提交保存在
`data/source/<job-id>/<operator-id>/`，其中包含 manifest 和原相对目录结构。

在线提交支持 CUDA 源码和 Linux ELF relocatable .o。对象必须实现同一套
生命周期符号，并声明编译时 CUDA 架构；服务端会在对应 Worker 上与
统一 benchmark harness 链接。

每组成功结果可以从 /api/v1/results.csv 下载规范 CSV，也可以通过
/api/v1/leaderboard/submissions 增量发布到 GitHub databank。发布配置位于
config/service.json，服务端需要设置 KERNELPERF_GITHUB_TOKEN。SQLite 只
保存 job 队列、运行状态、日志和服务端结果；静态排行榜的公开数据源是
GitHub 中的独立 CSV 和 manifest。
源码提交发布到排行榜时，databank 同时保存
`public/source/<submission-id>/` 源码快照。
