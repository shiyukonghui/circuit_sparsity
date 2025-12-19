# 电路稀疏性可视化工具和模型

用于检查[Gao 等人 2025年](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/)提出的稀疏电路模型的工具。提供运行推理的代码以及一个Streamlit仪表板，允许您与通过剪枝找到的任务特定电路进行交互。注意：此README由AI生成并经过轻微编辑。

## 安装

```bash
pip install -e .
```

## 启动可视化工具

从项目根目录启动Streamlit应用程序：

```bash
streamlit run circuit_sparsity/viz.py
```

应用程序从openaipublic网页加载数据并在本地缓存。当可视化工具加载时，您可以使用左侧列中的控件选择模型、数据集、剪枝扫描和节点预算`k`。图表使用Plotly渲染；大多数元素都是可交互的，支持悬停/点击探索。

带节点消融增量和激活预览的Streamlit电路可视化工具示例视图（wte/wpe选项卡）：

![Streamlit电路可视化工具](annotated-circuit-sparsity-viz.png)

## 运行模型前向传递

Transformer定义位于`circuit_sparsity.inference.gpt`中。该模块导出：

- `GPTConfig` / `GPT`：适用于CPU/GPU推理的轻量级GPT实现。
- `load_model(model_dir, cuda=False)`：便捷加载器，期望在`models/...`中找到`beeg_config.json`和`final_model.pt`对。

使用示例（改编自`tests/test_gpt.py`）：

```python
from circuit_sparsity.inference.gpt import GPT, GPTConfig, load_model
from circuit_sparsity.inference.hook_utils import hook_recorder
from circuit_sparsity.registries import MODEL_BASE_DIR

config = GPTConfig(block_size=8, vocab_size=16, n_layer=1, n_head=1, d_model=8)
model = GPT(config)
logits, loss, _ = model(idx, targets=targets)

# 获取激活值
with hook_recorder() as rec:
    model(idx)

# rec是一个类似{"0.attn.act_in": tensor(...), ...}的字典

pretrained = load_model(f"{MODEL_BASE_DIR}/models/<model_name>", cuda=False)
```

使用以下命令运行测试：

```bash
pytest tests/test_gpt.py
```

## 数据布局

项目资产位于`https://openaipublic.blob.core.windows.net/circuit-sparsity`下，具有以下结构：

- `models/<model_name>/`
  - `beeg_config.json`：用于重建模型的序列化`GPTConfig`。
  - `final_model.pt`：`circuit_sparsity.inference.gpt.load_model`使用的检查点。
- `viz/<experiment>/<model_name>/<task_name>/<sweep>/<k>/`
  - `viz_data.pkl`：`viz.py`加载的主要有效载荷（包含电路掩码、激活值、样本、重要性等）。
  - 预处理脚本产生时，额外的每次运行输出（掩码、直方图、样本桶）存储在同一树下。
- `train_curves/<model_name>/progress.json`：仪表板摘要表使用的训练指标。
- 其他实验特定目录（例如`csp_yolo1/`、`csp_yolo2/`）保存准备剪枝运行时产生的原始工件。

`viz.py`和`registries.py`中显示的文件路径假定此布局。如果重新定位数据，请更新`registries.py`。

## 模型

我们发布了用于获得论文结果的所有模型。有关所有模型的列表，请参见`registries.py`。确切的训练超参数可以在[todo]中找到

- `csp_yolo1`：这是用于`single_double_quote`定性结果的模型。这是一个总共118M参数的模型。这是一个较旧的模型，其训练方法与论文中的方法不完全相同；特别是同时使用多个L0值进行训练的方法。
- `csp_yolo2`：这是用于`bracket_counting`和`set_or_string_fixedvarname`定性结果的模型。这是一个总共475M参数的模型。
- `csp_sweep1_*`：这些模型用于获得图3的结果。名称表示模型大小（相对于任意基线大小的"扩展因子"）、权重L0和激活稀疏度级别（afrac）。
- `csp_bridge1`：用于获得论文结果的桥接模型。
- `csp_bridge2`：另一个桥接模型。
- `dense1_1x`：在我们的数据集上训练的密集模型。
- `dense1_2x`：在我们的数据集上训练的密集模型。宽度是2倍。
- `dense1_4x`：在我们的数据集上训练的密集模型。宽度是4倍。

## 扫描ID
- `prune_v2`：256次CARBS迭代，bs=16，非常旧的（未发布的）剪枝算法。目标是固定的`k`而不是固定的目标损失
- `prune_v3`：256次CARBS迭代，bs=64，epochs=32，旧的（未发布的）算法。目标是固定的目标损失
- `prune_v4`：768次CARBS迭代，bs=64，epochs=48，已发布的算法。目标是固定的目标损失
- `prune_v5_logitscaling`：256次CARBS迭代，bs=32，epochs=32，带有logit缩放的已发布算法。目标是固定的目标损失

## 其他实用程序

- `per_token_viz_demo.py`：令牌级可视化的最小示例。
- `clear_cache.py`：删除blobstore文件的本地缓存副本（Streamlit/viz缓存和tiktoken缓存）；如果您需要重新获取新的工件，请运行此命令。

该项目依赖于Streamlit、Plotly、matplotlib、seaborn和torch（完整依赖项列表请参见`pyproject.toml`）。