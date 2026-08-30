# Qiwu SpMV 独立源码插件改造说明

## 1. 改造目标

排行榜下载的源码应当是一个可独立构建和调用的CUDA源码插件，而不是只能被
KernelPerf源码模板嵌入的快照。插件与评测框架共享同一份公开接口，但不依赖
KernelPerf的Python代码、矩阵读取、计时、数据库或网页。

改造前状态已分别提交：

- KernelPerf：`82d12aabff32c8e16746209f84119bbb948f43b5`
- PlayGround：`e52aaa511be4f05839eac966b887b99d1549e18a`
- databank-alpha：`5ef3fa2573e2d8f6ea385575c1f1e38cc3b72c43`

三个仓库的改造分支均为 `feature/standalone-spmv-plugins`。

## 2. 架构

```text
用户提交源码目录
      |
      v
SpMV插件包生成器
      |-- 原始adapter与上游源码
      |-- include/qiwu/spmv_plugin.cuh
      |-- plugin.json
      |-- CMakeLists.txt
      |-- examples/standalone.cu
      `-- source_sha256
      |
      +--> KernelPerf runner + 独立插件编译单元 --> 评测与CSV
      |
      `--> databank source包 --> 排行榜下载 --> 其他项目直接编译调用
```

KernelPerf不执行用户包中的CMake文件。服务器仍使用固定编译参数和白名单
`build_profile`，仅将runner、插件入口和声明的`compile_units`分别编译后链接。
下载者使用包内CMake，不需要评测框架。

## 3. 公共接口

公共头文件为 `include/qiwu/spmv_plugin.cuh`。旧的
`kernelperf_spmv_*` 接口已删除，不提供兼容层。插件实现三个入口：

```cpp
qiwu_spmv_preprocess(...)
qiwu_spmv_solve(...)
qiwu_spmv_destroy(...)
```

输入为Host/Device CSR，插件可以在`preprocess`中创建任意自定义存储；`solve`
只执行 `y = A * x`；`destroy`释放插件资源。FP32和FP64使用同一份源码，分别
编译，FP64目标定义`QIWU_SPMV_FP64=1`。

这是源码级插件协议，因此不需要`abi_version`。公共头文件随源码包一起分发，
插件和调用方使用同一份头文件编译。预编译`.o`仍然受CUDA架构、工具链和精度
约束。

## 4. 下载包

解压排行榜下载的ZIP后，`plugin/`目录包含：

```text
plugin/
├── plugin.json
├── CMakeLists.txt
├── README-QIWU-PLUGIN.md
├── include/qiwu/spmv_plugin.cuh
├── examples/standalone.cu
├── adapter.cu或variants/*.cu
└── 必要的上游源码
```

直接构建和运行：

```bash
cmake -S plugin -B plugin/build
cmake --build plugin/build -j
./plugin/build/qiwu_spmv_example_fp32
./plugin/build/qiwu_spmv_example_fp64
```

一个包包含多个配置时，通过下列参数选择入口：

```bash
cmake -S plugin -B plugin/build \
  -DQIWU_PLUGIN_ENTRY=variants/csr_alg1.cu
```

GHOST包需要固定安装的GHOST库：

```bash
cmake -S plugin -B plugin/build -DGHOST_ROOT=/path/to/ghost/install
```

`plugin.json`是唯一的包清单，记录入口、完整配置映射、编译单元、基础格式、支持精度、依赖和文件列表。排行榜下载某个具体配置时会把该配置设为默认入口；
`source_sha256`对完整插件文件集合计算，用于内容寻址和去重。新发布源码保存到
`public/source/<source_sha256>/`。

## 5. 仓库改动

### KernelPerf

- 公共接口从评测模板中抽出到 `include/qiwu/spmv_plugin.cuh`。
- runner不再拼接adapter文本，改为独立编译和链接。
- `Benchmark.source_package()`成为通用扩展点，其他算子可以提供自己的包生成器。
- `benchmarks/spmv/package.py`生成完整插件包和内容哈希。
- `scripts/package_spmv_plugin.py`用于将已有源码目录打包为同一结构。
- 本地源码归档和GitHub发布均保存完整插件包。

### PlayGround

- 所有adapter迁移到 `qiwu_spmv_preprocess/solve/destroy`。
- adapter显式包含公共头文件，不再依赖评测模板提供类型。
- 根目录CMake可以构建任意入口的FP32、FP64静态库和独立示例。
- `.o`构建脚本使用同一公共头文件。

### databank-alpha

- 五类baseline的source均重新打包为独立插件，结果CSV未修改。
- 下载ZIP去掉多余的`files/`嵌套，解压后可以直接执行CMake。
- 手工增量导入只接受规范插件包，并按`source_sha256`保存。
- 数据审计新增插件清单、必需文件、内容哈希和旧接口残留检查。

## 6. 回归结果

- KernelPerf：`61 passed`。
- databank：156条公开结果，A100、RTX 5090、H100各52条，审计无失败。
- `public/data`与改造前提交`5ef3fa25`无差异。
- databank生产前端构建成功，插件清单、CMake、公共头文件和示例均可通过HTTP下载。
- 无头Chrome实际点击`cuSPARSE SELL C32 ALG1`的`Download source`后，ZIP根清单自动选择
  `variants/sell_c32_alg1.cu`；该下载产物在Windows上完成FP32、FP64编译和调用。
- Windows CUDA 12.5 / RTX 3050：CSR-Adaptive、AlphaSparseLib和cuSPARSE的
  FP32、FP64独立编译调用通过。
- A100 / Linux CUDA 12.2：CSR-Adaptive、AlphaSparseLib、cuSPARSE、CSR5和
  GHOST的FP32、FP64 CMake独立构建调用全部通过。
- A100上的新KernelPerf runner实际完成FP32、FP64评测和正确性校验，验证
  runner与adapter分离编译后的完整评测路径。

A100当前系统CUDA为12.2，因此本次现场回归使用12.2；插件源码不绑定该版本。
部署统一CUDA 12.9时仍应使用相同包再次执行独立构建测试。
