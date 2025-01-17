---
sidebar_position: 998
---

# 常见问题解答

## DSPy 适合我吗？DSPy 与其他框架的对比

**DSPy** 的理念和抽象与其他库和框架有很大不同，因此通常很容易判断 **DSPy** 是否适合您的用例。如果您是 NLP/AI 研究人员（或探索新流程或新任务的从业者），答案通常是肯定的 **yes**。如果您是从事其他工作的从业者，请继续阅读。

**DSPy 与提示的简单封装（OpenAI API、MiniChain、基本模板）的对比** 换句话说：_为什么我不能直接将提示写成字符串模板？_ 嗯，对于非常简单的设置，这 _可能_ 效果很好。（如果您熟悉神经网络，这就像用 Python for 循环来表示一个小的两层神经网络。它有点用。）但是，当您需要更高的质量（或可控的成本）时，您需要迭代地探索多阶段分解、改进的提示、数据引导、仔细的微调、检索增强和/或使用更小（或更便宜，或本地）的模型。使用基础模型构建的真正表达能力在于这些部分之间的交互。但是，每次更改一个部分时，您可能会破坏（或削弱）多个其他组件。**DSPy** 清晰地抽象出（_并_ 强大地优化）这些交互中与您的实际系统设计无关的部分。它让您专注于设计模块级别的交互：用 10 或 20 行 **DSPy** 代码表达的_相同程序_可以轻松地编译为用于 `GPT-4` 的多阶段指令、用于 `Llama2-13b` 的详细提示或用于 `T5-base` 的微调。哦，您不再需要在项目的核心维护长而脆弱的、特定于模型的字符串了。

**DSPy 与像 LangChain、LlamaIndex 这样的应用程序开发库的对比** LangChain 和 LlamaIndex 面向高级应用程序开发；它们提供了 _开箱即用_、预构建的应用程序模块，可以与您的数据或配置相结合。如果您乐于使用通用的、现成的提示来对 PDF 或标准文本到 SQL 进行问答，您会在这些库中找到丰富的生态系统。**DSPy** 内部不包含针对特定应用程序的手工提示。相反，**DSPy** 引入了一小组功能更强大、用途更广泛的模块，_可以在您的数据上学习提示（或微调）您的 LM_。当您更改数据、调整程序的控制流或更改目标 LM 时，**DSPy 编译器**可以将您的程序映射到一组新的提示（或微调），这些提示是专门为这个流程优化的。因此，您可能会发现 **DSPy** 以最少的努力获得最高的任务质量，前提是您愿意实现（或扩展）自己的简短程序。简而言之，**DSPy** 适用于您需要轻量级但自动优化的编程模型的情况，而不是预定义提示和集成的库。如果您熟悉神经网络：这就像 PyTorch（即表示 **DSPy**）和 HuggingFace Transformers（即表示更高级的库）之间的区别。

**DSPy 与像 Guidance、LMQL、RELM、Outlines 这样的生成控制库的对比** 这些都是用于控制 LM 单个补全的新兴库，例如，如果您想强制执行 JSON 输出模式或将采样限制为特定的正则表达式。这在许多设置中非常有用，但它通常侧重于对单个 LM 调用的低级别、结构化控制。它不能帮助确保您获得的 JSON（或结构化输出）对您的任务是正确或有用的。相比之下，**DSPy** 自动优化程序中的提示，使其与各种任务需求保持一致，其中可能还包括生成有效的结构化输出。也就是说，我们正在考虑允许 **DSPy** 中的 **Signatures** 来表达由这些库实现的类正则表达式的约束。

## 基本用法

**我应该如何为我的任务使用 DSPy？** 我们编写了一个[八步指南](/building-blocks/solving_your_task)。简而言之，使用 DSPy 是一个迭代过程。您首先定义您的任务和您想要最大化的指标，并准备一些示例输入——通常没有标签（或者只有最终输出的标签，如果您的指标需要它们）。然后，您通过选择要使用的内置层（`modules`）、为每个层提供一个 `signature`（输入/输出规范），然后在您的 Python 代码中自由调用您的模块来构建您的流程。最后，您使用 DSPy `optimizer` 将您的代码编译为高质量的指令、自动的少样本示例或为您的 LM 更新的 LM 权重。

**如何将我复杂的提示转换为 DSPy 流程？** 请参阅上面的相同答案。

**DSPy 优化器调整什么？** 或者，_编译实际上做了什么？_ 每个优化器都不同，但它们都试图通过更新提示或 LM 权重来最大化程序上的指标。当前的 DSPy `optimizers` 可以检查您的数据，模拟通过您的程序的轨迹来生成每个步骤的好/坏示例，根据过去的结果为每个步骤提出或完善指令，对自我生成的示例微调您的 LM 的权重，或结合其中几个步骤来提高质量或降低成本。我们很乐意合并新的优化器来探索更丰富的空间：您目前为提示工程、“合成数据”生成或自我改进所经历的大多数手动步骤都可以概括为作用于任意 LM 程序的 DSPy 优化器。

其他常见问题解答。我们欢迎 PR 为此处的每个问题添加正式答案。您将在现有问题、教程或所有或大部分这些问题的论文中找到答案。

- **如何获得多个输出？**

您可以指定多个输出字段。对于短格式签名，您可以将多个输出列为逗号分隔的值，后跟 "->" 指示符（例如 "inputs -> output1, output2"）。对于长格式签名，您可以包含多个 `dspy.OutputField`。

- **如何定义我自己的指标？指标可以返回浮点数吗？**

您可以将指标定义为简单的 Python 函数，它们处理模型生成并根据用户定义的需求对其进行评估。指标可以将现有数据（例如，黄金标签）与模型预测进行比较，或者可以使用来自 LM 的验证反馈（例如，LLM 作为评委）来评估输出的各种组件。指标可以返回 `bool`、`int` 和 `float` 类型的分数。查看官方[指标文档](/building-blocks/5-metrics)以了解有关定义自定义指标以及使用 AI 反馈和/或 DSPy 程序进行高级评估的更多信息。

- **编译有多贵或多慢？**

为了反映编译指标，我们突出一个实验以供参考，使用 `gpt-3.5-turbo-1106` 模型在 7 个候选程序和 10 个线程上，使用 [`dspy.BootstrapFewShotWithRandomSearch`](/deep-dive/teleprompter/bootstrap-fewshot) 优化器编译 [`SimplifiedBaleen`](/tutorials/simplified-baleen)。我们报告编译此程序大约需要 6 分钟，进行 3200 次 API 调用，使用 270 万个输入令牌和 15.6 万个输出令牌，报告总成本为 3 美元（按 OpenAI 模型的当前定价）。

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

