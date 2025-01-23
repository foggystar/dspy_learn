import dspy
from dspy.teleprompt import MIPROv2,LabeledFewShot ,BootstrapFewShot,BootstrapFewShotWithRandomSearch
import os
import dspy.evaluate
import pandas as pd
import random
import inject_detect_dspy as detect

SAVE_PATH = "dspy_program/"

if __name__ == "__main__":
    detect.Init(model='ollama_chat/phi4', api_base='http://localhost:11434',hello_test=True,path="output/log")
    allset = detect.make_allset("./input/new_data.csv")
    # trainset = random.sample(allset, 50)
    evalset = random.sample(allset, 200)

    # 初始化你的模型
    initial = detect.injectionJudge()

    # 使用 BootstrapFewShot 进行 few-shot learning
    print("Compiling model")
    # optimizer = LabeledFewShot(k=40)
    # optimizer = MIPROv2(metric=metric, max_labeled_demos=60,max_bootstrapped_demos=40,max_errors=10)
    # optimizer = BootstrapFewShot(metric=metric, max_rounds=5,max_labeled_demos=60,max_bootstrapped_demos=40)
    optimizer = BootstrapFewShot(metric=detect.metric, max_rounds=3,max_labeled_demos=20,max_bootstrapped_demos=5)
    trained = optimizer.compile(student=initial, trainset=allset)
    assert trained is not None, "Failed to compile student"
    
    # 保存最佳模型
    trained.save(SAVE_PATH, save_program=True)
    # # 评估器
    # evaluator = dspy.evaluate.Evaluate(devset=evalset, num_threads=50, display_progress=True, return_all_scores=True)
    # # 评估模型
    # print("Evaluating model")
    
    # initial_score = evaluator(initial, metric=metric)
    # trained_score = evaluator(trained, metric=metric)
    # print(f"Initial score: {initial_score}, Evaluation scores: {trained_score}")
