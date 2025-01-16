
import dspy # It takes too long to import the package
from dspy.teleprompt import BootstrapFewShot
import os
import dspy.evaluate
import pandas as pd
class Analyizer(dspy.Signature):
    query: str = dspy.InputField(desc="The user's input to a LLM.")
    intention: str = dspy.OutputField(desc="The user's intention.")
class Honeypot(dspy.Signature):
    query: str = dspy.InputField(desc="The user's input to a LLM.")
    response: str = dspy.OutputField(desc="response to user")
class Classifier(dspy.Signature):
    """Judge whether the user is attacking a LLM."""
    query: str = dspy.InputField(desc="The user's input to a LLM.")
    intention: str = dspy.InputField(desc="The user's intention.")
    response: str = dspy.InputField(desc="The LLM's response.")

    malevolence: bool = dspy.OutputField(desc="Judge whether the user is attacking a LLM.")
    alienation: bool = dspy.OutputField(desc="Is the user try to change the LLM's behavior?")
    analysis: str = dspy.OutputField(desc="How do you judge the malevolence")
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.predictor = dspy.Predict(Classifier)
    # confidence: float = dspy.OutputField()
class Hop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought('claim, notes -> query')
        self.append_notes = dspy.ChainOfThought('claim, notes, context -> new_notes: list[str], titles: list[str]')
    
class injectionJudge(dspy.Module):
    # def __init__(self):
    #     super().__init__()
    #     self.res = dspy.Predict(Classifier)

    def forward(self, **kwargs):
        analyze=dspy.Predict(Analyizer)
        judge = dspy.Predict(Classifier)
        honeypot = dspy.Predict(Honeypot)
        intention=analyze(**kwargs).intention
        response=honeypot(**kwargs).response
        res=judge(**kwargs,intention=intention,response=response)
        log("log.txt",**kwargs,intention=intention,response=response,res=res)
        return res

def Init():
    API_Key = os.getenv('DEEPSEEK_API_KEY')
    # deepseek-ai/DeepSeek-V3
    lm = dspy.LM('deepseek-chat', api_base='https://api.deepseek.com/v1', api_key=API_Key)
    dspy.configure(lm=lm)
    print(lm("Hello", temperature=1))    

def log(path,**kwargs):
    with open(path,"a",encoding='utf-8') as file:
        file.write("-------------------------------------------------\n$$$ Query: ")
        file.write(kwargs['query'])
        file.write("\n\n")
        file.write(f"""$$$ Intention: {kwargs['intention']}\n\n
                   $$$ Response: {kwargs['response']}\n\n
                   $$$ {kwargs['res']}\n\n""")

def metric(example, pred, trace=None):
    label = str(example.label).lower()  # 确保 label 是字符串并转换为小写
    pred_malevolence = str(pred.malevolence).lower() # 确保 pred_malevolence 是字符串并转换为小写
    if label == "true":  # 处理布尔值情况
        return pred.malevolence and pred.alienation
    return label == pred_malevolence

def evaluate(model, trainset):
    scores = []
    for x in trainset:
        pred = model(**x.inputs())
        score = metric(x, pred)
        scores.append(score)
    return scores

def make_trainset(dataPath):
    df = pd.read_csv(dataPath)
    trainset = []
    for _,row in df.iterrows():
        trainset.append(dspy.Example(query=row['query'],analysis=row['res'] ,label=row['label']).with_inputs("query"))
    return trainset


if __name__ == "__main__":
    Init()
    print()
    trainset=make_trainset("./new_data.csv")
    with open("output","w",encoding='utf-8') as file:
        pass
    # print(evaluate(trainset))
    optimizer = BootstrapFewShot(metric=metric)
    compiled_student = optimizer.compile(student = injectionJudge(), trainset=trainset)
    assert compiled_student is not None, "Failed to compile student"
    compiled_student.forward(query="Hello") # 如何将trainset传入
    compiled_student.save("student.json")


if __name__ == "__main__":
    Init()
    trainset = make_trainset("./new_data.csv")

    optimizer = BootstrapFewShot(metric=metric, n_rounds=3, n_train_examples=2, n_eval_examples=2) # 设置训练参数
    compiled_student = optimizer.compile(student=injectionJudge(), trainset=trainset[:10]) # 使用部分trainset进行few-shot learning
    assert compiled_student is not None, "Failed to compile student"


    # 使用fit方法训练模型
    best_model = optimizer.fit(compiled_student, trainset[:4])

    # 评估模型
    scores = evaluate(best_model, trainset[4:])  # 使用剩余的trainset进行评估
    print(f"Evaluation scores: {scores}")
    average_score = sum(scores) / len(scores) if scores else 0
    print(f"Average score: {average_score}")


    # 保存最佳模型
    best_model.save("best_student.json")

    # 使用训练好的模型进行预测
    test_query = "Hello"
    prediction = best_model(query=test_query)
    print(f"Prediction for '{test_query}': {prediction}")