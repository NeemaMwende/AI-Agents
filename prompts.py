def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
from fastapi import params
from ibm_watsonx_ai.foundation_models import Model
# from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
# from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
# from ibm_watson_machine_learning.foundation_models.extensions.langchain import␣
# ↪WatsonxLLM
# from langchain_core.prompts import PromptTemplate
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

# zero shot prompting
# prompt = """Classify the following statement as true or false:
# 'The Eiffel Tower is located in Berlin.'
# Answer:
# """
# response = llm_model(prompt, params)
# print(f"prompt: {prompt}\n")
# print(f"response : {response}\n")


# one shot prompt
params = {
    "max_new_tokens": 20,
    "temperature": 0.1,
}
prompt = """Here is an example of translating a sentence from English to French:
            English: “How is the weather today?”
            French: “Comment est le temps aujourd'hui?”
            Now, translate the following sentence from English to French:
            English: “Where is the nearest supermarket?”
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")  



# few shot prompt
params = {
"max_new_tokens": 10,
}
prompt = """Here are few examples of classifying emotions in statements:
Statement: 'I just won my first marathon!'
Emotion: Joy
Statement: 'I can't believe I lost my keys again.'
Emotion: Frustration
Statement: 'My best friend is moving to another country.'
Emotion: Sadness
Now, classify the emotion in the following statement:
Statement: 'That movie was so scary I had to cover my eyes.’
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")




# chain of thought prompt 
params = {
"max_new_tokens": 512,
"temperature": 0.5,
}
prompt = """Consider the problem: 'A store had 22 apples. They sold 15 apples␣
↪today and got a new delivery of 8 apples.
How many apples are there now?’
Break down each step of your calculation
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")  





# Self-consistency 
params = {
"max_new_tokens": 512,
}
prompt = """When I was 6, my sister was half of my age. Now I am 70, what age␣
↪is my sister?
Provide three independent calculations and explanations, then␣
determine the most consistent result.
↪
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")


