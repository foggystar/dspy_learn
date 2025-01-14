import dspy
print("1")
import os
API_Key = os.getenv('DEEPSEEK_API_KEY')
# deepseek-ai/DeepSeek-V3
lm = dspy.LM('deepseek-chat', api_base='https://api.deepseek.com/v1', api_key=API_Key)
dspy.configure(lm=lm)
try:
    print(lm("Say this is a test!", temperature=0.7))  # => ['This is a test!']
    print("Success")
except Exception as e:
    print(e)
# lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']