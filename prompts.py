def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
from fastapi import params
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
# from ibm_watson_machine_learning.foundation_models.extensions.langchain import␣
# ↪WatsonxLLM
from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain 

def llm_model(prompt_txt, params=None):
    model_id = 'ibm/granite-3-2-8b-instruct'
    default_params = {
    "max_new_tokens": 256,
    "min_new_tokens": 0,
    "temperature": 0.5,
    "top_p": 0.2,
    "top_k": 1
    } 

prompt = """Classify the following statement as true or false:
'The Eiffel Tower is located in Berlin.'
Answer:
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")