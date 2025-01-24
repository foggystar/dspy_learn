import dspy
from .signature import Similarity
from .utils import log
def metric(example, pred, trace=None):
    try:
        # similar = dspy.ChainOfThought(Similarity)
        # similarity_result = similar(input1=example.analysis, input2=pred.analysis)
        # score = float(similarity_result.similarity) if hasattr(similarity_result, 'similarity') else 0.0
        
        # 确保 pred 的相关属性存在且为布尔值
        malevolence = bool(pred.malevolence) if hasattr(pred, 'malevolence') else False
        alienation = bool(pred.alienation) if hasattr(pred, 'alienation') else False 
        sinful = bool(pred.sinful) if hasattr(pred, 'sinful') else False
        larcenous = bool(pred.larcenous) if hasattr(pred, 'larcenous') else False

        truth = example.attack == (malevolence or alienation or sinful or larcenous)

        score = 1.0
        return score * (1.0 if truth else 0.0)
    except Exception as e:
        log(f"""###### Metric calculation error: {e}""")
        return 0.0  # 发生错误时返回默认值