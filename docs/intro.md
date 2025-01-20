---
sidebar_position: 998
---

# 常见问题解答

**DSPy 优化器调整什么？** 或者，_编译实际上做了什么？_ 每个优化器都不同，但它们都试图通过更新提示或 LM 权重来最大化程序上的指标。当前的 DSPy `optimizers` 可以检查您的数据，模拟通过您的程序的轨迹来生成每个步骤的好/坏示例，根据过去的结果为每个步骤提出或完善指令，对自我生成的示例微调您的 LM 的权重，或结合其中几个步骤来提高质量或降低成本。我们很乐意合并新的优化器来探索更丰富的空间：您目前为提示工程、“合成数据”生成或自我改进所经历的大多数手动步骤都可以概括为作用于任意 LM 程序的 DSPy 优化器。

其他常见问题解答。我们欢迎 PR 为此处的每个问题添加正式答案。您将在现有问题、教程或所有或大部分这些问题的论文中找到答案。

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

