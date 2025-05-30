import os
import openai
from dotenv import load_dotenv
from time import sleep

load_dotenv()

OPENAI_API_URL = "https://api.proxyapi.ru/openai/v1"

client = openai.OpenAI(base_url=OPENAI_API_URL, api_key=os.getenv('OPENAI_API_KEY'))


def get_prompt(language, target):
    if isinstance(target, int):
        cwe = f'CWE-{target}'
    elif isinstance(target, list):
        cwe = ' or '.join(f'CWE-{t}' for t in target)
    else:
        raise TypeError(type(target))

    return f"""You are an advanced AI model specialized in code security analysis. 
Your task is to perform binary classification of the given {language} code snippet 
to determine whether it contains a vulnerability related to {cwe}.

Instructions:  
- Analyze the provided code snippet for security issues specifically related to {cwe}.  
- If a vulnerability matching {cwe} is found, return "1".  
- If no such vulnerability is found, return "0".  
- Your response must be strictly a single number ("1" or "0"), with no explanations, comments, or additional text.  
- If the code is incomplete, unclear, or not related to programming, 
evaluate only the given snippet without making assumptions.  
Your response format must be strictly: """


def predict(classifier, language, target, code):
    sleep(1)
    system_prompt = get_prompt(language, target)

    if not classifier.startswith('o'):
        response = client.chat.completions.create(
            model=classifier,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": code}
            ],
            temperature=0.1
        )
    else:
        response = client.chat.completions.create(
            model=classifier,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": code}
            ],
        )
    return int('1' in response.choices[0].message.content)


def get_prediction(classifier, language, target, code):
    num_labels = classifier.get('num_labels')
    if num_labels == 1:
        p = predict(classifier['classifier'], language, target, code)
    elif num_labels == 2 or isinstance(target, int):
        p = predict(classifier['classifier'], language, target, code)
        p = [1 - p, p]
    elif isinstance(target, list):
        p = [predict(classifier['classifier'], language, t, code) for t in target]
        p = [int(1 not in p)] + p
    else:
        raise TypeError(type(target))
    return p
