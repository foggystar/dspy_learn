# dspy笔记

dspy——自动化prompt优化框架

这篇文档仅仅是一个笔记、简介，详细内容、系统性学习请访问 [DSPy官网](https://dspy.ai/)

[Cheat Sheet](https://dspy.ai/cheatsheet)

[toc]

## 为何使用DSPy？与其他框架的对比

**DSPy** 的理念和抽象与其他库和框架有很大不同，因此通常很容易判断 **DSPy** 是否适合您的用例。如果您是 NLP/AI 研究人员，答案通常是肯定的。如果您是从事其他工作的从业者，请继续阅读。

**DSPy 与提示的简单封装（OpenAI API、MiniChain、基本模板）的对比** 换句话说：为什么我不直接使用字符串提示词？ 对于非常简单的设置，这可能效果很好。（如果您熟悉神经网络，这就像用 Python for 循环来表示一个小的两层神经网络。）但是，当您需要更高的质量（或可控的成本）时，您需要迭代地探索多阶段分解、改进的提示、数据引导、仔细的微调、检索增强和/或使用更小（或更便宜，或本地）的模型。使用基础模块进行构建的真正强大之处在于这些部分的互动。但是，每次更改一个部分时，您可能会破坏（或削弱）多个其他组件。**DSPy** 清晰地抽象出（并强大地优化）这些交互中与实际系统设计无关的部分。它让您专注于设计模块级别的交互：用 10 或 20 行 **DSPy** 代码编写的代码可以轻松编译为 `GPT-4` 的多阶段指令、 `Llama2-13b` 的详细提示或 `T5-base` 的微调。您不再需要维护项目核心中那些长而脆弱的、特定于模型的字符串了。

**DSPy 与像 LangChain、LlamaIndex 这样的应用程序开发库的对比** LangChain 和 LlamaIndex 面向高级应用程序开发；它们提供了 开箱即用、预构建的应用程序模块，可以与您的数据或配置相结合。如果您乐于使用通用的、现成的prompt来对 PDF 进行问答或标准文本到 SQL，您会在这些库中得到充足的资源。**DSPy** 内部不包含针对特定应用程序的手工提示。相反，**DSPy** 引入了一小组功能更强大、用途更广泛的模块，它们可以基于您的数据学习提示（prompt）或微调您的 LM。当您更改数据、调整程序的控制流或更改目标 LM 时，**DSPy 编译器**可以将您的程序映射到一组新的提示或微调，这些提示是专门为当前流程优化的。因此，您可能会发现 **DSPy** 以最少的努力获得最高的任务质量。简而言之，**DSPy** 适用于您需要轻量级但自动优化提示词的情况，而非预定义提示和集成的库。如果您熟悉神经网络：这就像 PyTorch（即表示 **DSPy**）和 HuggingFace Transformers（即表示更高级的库）之间的区别。

**DSPy 与像 Guidance、LMQL、RELM、Outlines 这样的生成控制库的对比** 这些都是用于控制 LM 单个补全（individual completions）的新兴库，例如，如果您想强制以 JSON 格式输出，或以正则表达式限制采样。这在许多情况中很有用，但它通常侧重于对单个 LM 调用的底层、结构化控制。它不能确保您获得的 JSON（或结构化输出）对是正确或有用的。相比之下，**DSPy** 生成的prompt可以满足各种任务需求，其中还可包括结构化输出。也就是说， **DSPy** 中的 **Signatures** 可实现类正则表达式的约束。

**如何使用 DSPy？** 使用dspy本质上是一个迭代过程。首先，需要明确、定义好你要完成的任务，以及需要优化的指标（metrics）（初学时，若不在练习项目中使用提示词优化，可暂时忽略这部分），并准备一些样例数据，可以仅包含指标中要求的标签。之后，使用模块（modules）构建程序，给予每一个模块一个签名（signature），用于规定模块的输入、输出，之后便可调用LM运行程序。最终，使用优化器（optimizer）将代码编译为高质量的指令、自动的少样本示例或为您的 LM 更新的 LM 权重。


## 主要功能：
### 零、初始化
- 
    ```python
    import dspy
    import os
    API_Key = os.getenv('DEEPSEEK_API_KEY') #写在系统变量中
    lm = dspy.LM('deepseek-chat', api_base='https://api.deepseek.com/v1', api_key=API_Key)
    dspy.configure(lm=lm)
    ```
### 一、使用模块指导LM完成任务 
#### 签名 Signature [源网站](https://dspy.ai/learn/programming/signatures/)
dspy包中有多个LM控制接口，最重要的传入参数为字符串，**Signature**，内涵关键字：**提供给LM的知识，要回答的问题，这个步骤要达成的目标等**。**Signature**内容会交由LM解析翻译为完整prompt，因此其仅需包含几个关键词，并使用"->"等符号指示逻辑关系，关键词意思正确即可，无需是特定词语

样例：
1. 问答: `"question -> answer"`, 等同于 `"question: str -> answer: str"`，因为默认类型是**str**
2. 语气分类: `"sentence -> sentiment: bool"`, 若语气积极则输出**True**
3. 总结: `"document -> summary"`

DSPy 提供两种定义Signature的方法：行内签名和类定义签名

##### 行内（简单）
直接将Signature作为Modules的参数

```python
math = dspy.ChainOfThought("question -> result")
math(question="Two dice are tossed. What is the probability that the sum equals two?")
```

##### 类定义（复杂）

此时使用自己定义的类作为Modules参数，需要包含以下内容：
- 输入域 **dspy.InputField**
- 输出域 **dspy.OutputField**
- 用于明确内容、要求的语句 **docstring**，以`desc="Requires"`格式作为输入、输出域的参数
```python
class ExtractInfo(dspy.Signature):
    """Extract structured information from text."""

    text: str = dspy.InputField()
    title: str = dspy.OutputField()
    headings: list[str] = dspy.OutputField()
    entities: list[dict[str, str]] = dspy.OutputField(desc="a list of entities and their metadata")

module = dspy.Predict(ExtractInfo)

text = "Apple Inc. announced its latest iPhone 14 today." \
    "The CEO, Tim Cook, highlighted its new features in a press release."
response = module(text=text)

print(response.title)
print(response.headings)
print(response.entities)
```

通过这样的模块化、标准化编程，在LM得出的最终结果出现错误时，易于排查错误位置，节省API资源

#### 模块 [源网站](https://dspy.ai/learn/programming/modules)
DSPy提供了一些模块，其会根据Signature生成提示词并调用大模型，给出结果

这些模块都是基于**dspy.Predict**的，其类似于问答模型的根本功能——文章续写，或者说接龙
```python
sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.
# 1) Declare with a signature.
classify = dspy.Predict('sentence -> sentiment: bool')
# 2) Call with input argument(s). 
response = classify(sentence=sentence)
# 3) Access the output.
print(response.sentiment)
```
使用其他合适的模块可以提高回答质量
```python
question = "What's something great about the ColBERT retrieval model?"
# 1) Declare with a signature, and pass some config.
classify = dspy.ChainOfThought('question -> answer', n=5)
# 2) Call with input argument.
response = classify(question=question)
# 3) Access the outputs.
response.completions.answer
```
一部分其他Modules的介绍：
它们区别不大，Signature -> Prompt 的过程实现会有一定区别
1. **`dspy.Predict`**: 基本预测器。不修改Signature。处理学习的主要形式（即存储指令、演示和更新 LM）。

2. **`dspy.ChainOfThought`**: 让 LM 在承诺签名回应之前逐步思考。

3. **`dspy.ProgramOfThought`**: 让 LM 输出代码，代码的执行结果将决定响应。

4. **`dspy.ReAct`**: 可以使用工具执行给定函数。

5. **`dspy.MultiChainComparison`**: 可以比较来自**CoT**的多个输出，以得出最终预测结果。



------------------------------



### 二、评估当前模型
#### 1. 构建训练数据

##### 构造**Examples**对象，其类似于**dict**
- 
    ```python
    # Code
    qa_pair = dspy.Example(question="This is a question?", answer="This is an answer.")

    print(qa_pair)
    print(qa_pair.question)
    print(qa_pair.answer)
    ```
    ```bash
    # Output
    Example({'question': 'This is a question?', 'answer': 'This is an answer.'}) (input_keys=None)
    This is a question?
    This is an answer.
    ```

##### 标记输入与标签
-
    ```python
    article_summary = dspy.Example(article= "This is an article.", summary= "This is a summary.").with_inputs("article")

    input_key_only = article_summary.inputs() # 输入
    non_input_key_only = article_summary.labels() # 标签

    print("Example object with Input fields only:", input_key_only)
    print("Example object with Non-Input fields only:", non_input_key_only)
    ```
##### 构造训练集
-
    ```python
    trainset = [dspy.Example(report="LONG REPORT 1", summary="short summary 1"), ...]
    ```
#### 2. 构造评估标准
##### 简易度量函数，定义此形式的函数从而使用**dspy**内置训练函数
-
    ```python
    def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()
    ```
##### 可以使用内置度量函数
- 
    ```python
    dspy.evaluate.metrics.answer_exact_match
    dspy.evaluate.metrics.answer_passage_match
    ```
##### 度量函数可以更加复杂，例如检查多个属性。若 trace 为 None（即用于评估或优化），度量函数返回 float，否则将返回 bool（即用于引导演示）
- 
    ```python
    def validate_context_and_answer(example, pred, trace=None):
    # 将真实答案和预测答案转换为小写后进行比较，确保大小写不影响匹配结果。
    answer_match = example.answer.lower() == pred.answer.lower()

    # 检查预测答案是否出现在提供的上下文列表中的任意一个上下文中。
    context_match = any((pred.answer.lower() in c) for c in pred.context)

    if trace is None: # 评估或优化模式，一个分数，表示预测的质量
        return (answer_match + context_match) / 2.0
    else: # 自举模式，用于筛选高质量的示例用于训练
        return answer_match and context_match
    ```
##### 使用数据集
- 
    ```python
    # 构造训练集
    def make_trainset(dataPath):
        df = pd.read_csv(dataPath)
        trainset = []
        for index,row in df.iterrows():
            trainset.append(dspy.Example(query=row['query'],analysis=row['res'] ,label=row['label']).with_inputs("query"))
        return trainset
    
    # 评估函数
    def evaluate(trainset):
    scores = []
    for x in trainset:
        pred = run_model(**x.inputs())
        score = metric(x, pred)
        scores.append(score)
    return scores
    ```
##### 使用**AI**进行度量
- 
    ```python
    # Define the signature for automatic assessments.
    class Assess(dspy.Signature):
        """Assess the quality of a tweet along the specified dimension."""

        assessed_text = dspy.InputField()
        assessment_question = dspy.InputField()
        assessment_answer: bool = dspy.OutputField()
    ```
    ```python
    def metric(gold, pred, trace=None):
    # 1. 提取问题、答案和生成的推文
    question, answer, tweet = gold.question, gold.answer, pred.output

    # 2. 定义两个评估标准的问题
    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    correct = f"The text should answer `{question}` with `{answer}`. Does the assessed text contain this answer?"

    # 3. 使用 dspy.Predict(Assess) 评估推文
    correct =  dspy.Predict(Assess)(assessed_text=tweet, assessment_question=correct)
    engaging = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=engaging)

    # 4. 提取评估结果
    correct, engaging = [m.assessment_answer for m in [correct, engaging]]
    
    # 5. 计算最终分数
    # 如果推文正确且长度在280字符以内，分数为correct和engaging的和
    # 否则分数为0
    score = (correct + engaging) if correct and (len(tweet) <= 280) else 0

    # 6. 根据trace参数返回不同格式的结果
    if trace is not None: return score >= 2  # 返回布尔值
    return score / 2.0  # 返回0-1之间的分数
    ```