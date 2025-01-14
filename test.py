import dspy
import os
API_Key = os.getenv('DEEPSEEK_API_KEY')
lm = dspy.LM('deepseek-chat', api_base='https://api.deepseek.com', api_key=API_Key)
dspy.configure(lm=lm)