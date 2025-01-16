# dspy笔记

dspy——自动化prompt优化框架

[DSPy官网](https://dspy.ai/)

[toc]

## 主要功能：
### 一、使用模块指导LM完成任务
dspy包中有多个LM控制接口，传入参数为字符串，记为**str**，内涵关键字：**提供给LM的知识，要回答的问题，这个步骤要达成的目标等**。**str**内容会交由LM解析翻译为完整prompt，因此其仅需包含几个关键词，并使用"->"等符号指示逻辑关系，关键词意思正确即可，无需是特定词语
##### 初始化
- 
    ```python
    import dspy
    import os
    API_Key = os.getenv('DEEPSEEK_API_KEY') #写在系统变量中
    lm = dspy.LM('deepseek-chat', api_base='https://api.deepseek.com/v1', api_key=API_Key)
    dspy.configure(lm=lm)
    ```
##### 简答案例——数学问题
- 
    ```python
    math = dspy.ChainOfThought("question -> result")
    math(question="Two dice are tossed. What is the probability that the sum equals two?")
    ```
##### 较复杂案例——按格式提取信息
-
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