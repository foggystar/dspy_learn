import dspy
import pandas as pd
import inject_detect_dspy as detect

READ_PATH = "./dspy_program/"

def all_true(example, pred, trace=None):
    detect.log(f"ans: {pred}\n")
    return True

if __name__ == "__main__":
    detect.Init(model='ollama_chat/phi4', api_base='http://localhost:11434',hello_test=True,path="output/labeler")
    # 加载模型
    judge = dspy.load(READ_PATH)
    # 现在 loaded_student 就是你之前训练好的模型
    # 你可以像调用普通 dspy.Module 一样使用它进行预测
    df = pd.read_csv("input/new_data_1_20.csv")
    questions = df['query']
    labels = df['label'].map({'white': False, 'black': True})
    evalset = []
    for _, row in df.iterrows():
        evalset.append(dspy.Example(query=row['query'], label=row['label']).with_inputs("query"))
    
    evaluator = dspy.evaluate.Evaluate(devset=evalset, num_threads=50, display_progress=True, return_outputs=True)
    score = evaluator(judge, metric=all_true)
    analysis = [pred[1].analysis for pred in score[1]]
    judge = [(pred[1].malevolence or pred[1].alienation or pred[1].sinful or pred[1].larcenous) for pred in score[1]]

    correct = labels == judge

    res=pd.DataFrame({'question':questions,'analysis':analysis,'label':labels,'judge':judge,'correct':correct})
    res.to_csv("output/result.csv",index=False)
    res.to_excel("output/result.xlsx",index=False)