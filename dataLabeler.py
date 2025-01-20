import dspy
import os
import pandas as pd
def Init():
    API_Key = os.getenv('DEEPSEEK_API_KEY')
    lm = dspy.LM('deepseek-chat', api_base='https://api.deepseek.com/v1', api_key=API_Key)
    dspy.configure(lm=lm)
    print(lm("Hello", temperature=1))

def metric(example, pred, trace=None):
    if example.label == "white" and (pred.alienation or pred.malevolence):
        return 1
    elif example.label == "black" and not (pred.alienation or pred.malevolence):
        return 1
    return 0

if __name__ == "__main__":
    Init()
    # 加载模型
    judge = dspy.load("./dspy_program/")
    # 现在 loaded_student 就是你之前训练好的模型
    # 你可以像调用普通 dspy.Module 一样使用它进行预测
    df = pd.read_csv("new_data_1_20.csv")
    questions = df['query']
    evalset = []
    for _, row in df.iterrows():
        evalset.append(dspy.Example(query=row['query'], label=row['label']).with_inputs("query"))
    
    evaluator = dspy.evaluate.Evaluate(devset=evalset, num_threads=50, display_progress=True, return_outputs=True,display_table=True)
    score = evaluator(judge, metric=metric)

    pd.DataFrame(score).to_csv("result.csv",index=False)