import dspy # It takes too long to import the package
import os
API_Key = os.getenv('DEEPSEEK_API_KEY')
# deepseek-ai/DeepSeek-V3
lm = dspy.LM('deepseek-chat', api_base='https://api.deepseek.com/v1', api_key=API_Key)
dspy.configure(lm=lm)
print(lm("Say this is a test!", temperature=0.7))  # => ['This is a test!']

math = dspy.ChainOfThought("question -> answer: float")
math(question="Two dice are tossed. What is the probability that the sum equals two?")