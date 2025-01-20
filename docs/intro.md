---
sidebar_position: 998
---

# 常见问题解答

**DSPy 优化器调整什么？** 或者，_编译实际上做了什么？_ 每个优化器都不同，但它们都试图通过更新提示或 LM 权重来最大化程序上的指标。当前的 DSPy `optimizers` 可以检查您的数据，模拟通过您的程序的轨迹来生成每个步骤的好/坏示例，根据过去的结果为每个步骤提出或完善指令，对自我生成的示例微调您的 LM 的权重，或结合其中几个步骤来提高质量或降低成本。我们很乐意合并新的优化器来探索更丰富的空间：您目前为提示工程、“合成数据”生成或自我改进所经历的大多数手动步骤都可以概括为作用于任意 LM 程序的 DSPy 优化器。

其他常见问题解答。我们欢迎 PR 为此处的每个问题添加正式答案。您将在现有问题、教程或所有或大部分这些问题的论文中找到答案。

- **如何获得多个输出？**

您可以指定多个输出字段。对于短格式签名，您可以将多个输出列为逗号分隔的值，后跟 "->" 指示符（例如 "inputs -> output1, output2"）。对于长格式签名，您可以包含多个 `dspy.OutputField`。

- **如何定义我自己的指标？指标可以返回浮点数吗？**

您可以将指标定义为简单的 Python 函数，它们处理模型生成并根据用户定义的需求对其进行评估。指标可以将现有数据（例如，黄金标签）与模型预测进行比较，或者可以使用来自 LM 的验证反馈（例如，LLM 作为评委）来评估输出的各种组件。指标可以返回 `bool`、`int` 和 `float` 类型的分数。查看官方[指标文档](/building-blocks/5-metrics)以了解有关定义自定义指标以及使用 AI 反馈和/或 DSPy 程序进行高级评估的更多信息。

- **编译有多贵或多慢？**

为了反映编译指标，我们突出一个实验以供参考，使用 `gpt-3.5-turbo-1106` 模型在 7 个候选程序和 10 个线程上，使用 [`dspy.BootstrapFewShotWithRandomSearch`](/deep-dive/teleprompter/bootstrap-fewshot) 优化器编译 [`SimplifiedBaleen`](/tutorials/simplified-baleen)。我们报告编译此程序大约需要 6 分钟，进行 3200 次 API 调用，使用 270 万个输入token和 15.6 万个输出token，总成本为 3 美元（按 OpenAI 模型的当前定价）。

编译 DSPy `optimizers` 自然会产生额外的 LM 调用，但我们通过最小化的执行来证实这种开销，目标是最大化性能。这为通过使用更大的模型编译 DSPy 程序，在编译时学习增强的行为，并在推理时将这种行为传播到测试的较小模型，从而提高较小模型的性能提供了途径。

## 部署或可重复性问题

- **如何保存我编译的程序的检查点？**

以下是一个保存/加载编译模块的示例：

```python
cot_compiled = teleprompter.compile(CoT(), trainset=trainset, valset=devset)

#保存
cot_compiled.save('compiled_cot_gsm8k.json')

#加载：
cot = CoT()
cot.load('compiled_cot_gsm8k.json')
```

- **如何导出以进行部署？**

导出 DSPy 程序就像上面突出显示的保存它们一样简单！

- **如何搜索我自己的数据？**

诸如 [RAGautouille](https://github.com/bclavie/ragatouille) 之类的开源库使您可以通过高级检索模型（如 ColBERT）来搜索自己的数据，并提供嵌入和索引文档的工具。在开发 DSPy 程序时，请随意集成此类库以创建可搜索的数据集！

- **如何关闭缓存？如何导出缓存？**

从 v2.5 开始，您可以通过在 `dspy.LM` 中将 `cache` 参数设置为 `False` 来关闭缓存：

```python
dspy.LM('openai/gpt-4o-mini',  cache=False)
```

您的本地缓存将保存到全局 env 目录 `os.environ["DSP_CACHEDIR"]` 或笔记本的 `os.environ["DSP_NOTEBOOK_CACHEDIR"]`。您通常可以将 cachedir 设置为 `os.path.join(repo_path, 'cache')` 并从此处导出此缓存：
```python
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(os.getcwd(), 'cache')
```

!!! warning "重要"
    `DSP_CACHEDIR` 负责旧客户端（包括 dspy.OpenAI、dspy.ColBERTv2 等），`DSPY_CACHEDIR` 负责新的 dspy.LM 客户端。

    在 AWS lambda 部署中，您应该禁用 DSP_* 和 DSPY_*。

## 高级用法

- **如何并行化？**
您可以通过在各自的 DSPy `optimizers` 中或在 `dspy.Evaluate` 实用函数中指定多个线程设置，在编译和评估期间并行化 DSPy 程序。

- **如何冻结模块？**

可以通过将其 `._compiled` 属性设置为 True 来冻结模块，表明该模块已通过优化器编译，不应调整其参数。这在 `dspy.BootstrapFewShot` 等优化器中内部处理，其中确保在教师在引导过程中传播收集的少样本演示之前冻结学生程序。

- **如何使用 DSPy 断言？**

    a) **如何向您的程序添加断言**：
    - **定义约束**：使用 `dspy.Assert` 和/或 `dspy.Suggest` 在您的 DSPy 程序中定义约束。这些约束基于对您想要强制执行的结果的布尔值验证检查，这些检查可以简单地是 Python 函数来验证模型输出。
    - **集成断言**：保持您的断言语句在模型生成之后（提示：在模块层之后）

    b) **如何激活断言**：
    1. **使用 `assert_transform_module`**：
        - 使用 `assert_transform_module` 函数以及 `backtrack_handler` 包装带有断言的 DSPy 模块。此函数转换您的程序以包括内部断言回溯和重试逻辑，这些逻辑也可以自定义：
        `program_with_assertions = assert_transform_module(ProgramWithAssertions(), backtrack_handler)`
    2. **激活断言**：
        - 直接对带有断言的 DSPy 程序调用 `activate_assertions`： `program_with_assertions = ProgramWithAssertions().activate_assertions()`

    **注意**：要正确使用断言，您必须 **激活** 一个包含来自上述任一方法的 `dspy.Assert` 或 `dspy.Suggest` 语句的 DSPy 程序。

## 错误

- **如何处理 “上下文过长” 错误？**

如果您在 DSPy 中遇到 “上下文过长” 错误，则您可能正在使用 DSPy 优化器在提示中包含演示，而这超出了您当前的上下文窗口。尝试减少这些参数（例如 `max_bootstrapped_demos` 和 `max_labeled_demos`）。此外，您还可以减少检索到的段落/文档/嵌入的数量，以确保您的提示适合您的模型上下文长度。

一个更通用的修复方法是简单地增加为 LM 请求指定的 `max_tokens` 的数量（例如 `lm = dspy.OpenAI(model = ..., max_tokens = ...)`）。

