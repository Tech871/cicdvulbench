import json

from ollama import chat


def get_prompt(language, target):
    if isinstance(target, int):
        cwe = f'CWE-{target}'
    elif isinstance(target, list):
        cwe = ' or '.join(f'CWE-{t}' for t in target)
    else:
        raise TypeError(type(target))

    return f"""You are an AI model specialized in code security analysis.
Your task is to perform binary classification of the given {language} code snippet 
to determine whether it contains a vulnerability related to {cwe}.

**Instructions:**
- Analyze the provided code snippet for security issues specifically related to {cwe}.
- If a vulnerability matching {cwe} is found, return exactly: **"1"**.
- If no such vulnerability is found, return exactly: **"0"**.
- **Do NOT include any explanations, thoughts, or additional text.**
- Your response must be a valid JSON object in the following format: `{{"vulnerable": "0" or "1"}}`"""


def predict(classifier, language, target, code):
    response = chat(
        model=classifier,
        messages=[
            {'role': 'system', 'content': get_prompt(language, target)},
            {'role': 'user', 'content': code}
        ],
        options={'temperature': 0.1, 'response_format': 'json'}
    )

    answer = response.message.content.strip()

    if '0' in answer:
        return 0
    if answer == '1':
        return 1

    try:
        answer = json.loads(answer)
        if not isinstance(answer, dict):
            return 0
        if 'vulnerable' not in answer:
            return 0
        answer = str(answer['vulnerable'])
        if answer == '1':
            return 1
    except json.JSONDecodeError as e:
        print(e)

    return 0


def get_prediction(classifier, language, target, code):
    num_labels = classifier.get('num_labels')
    if num_labels == 1:
        return predict(classifier['classifier'], language, target, code)
    if num_labels == 2 or isinstance(target, int):
        p = predict(classifier['classifier'], language, target, code)
        return [1 - p, p]
    if isinstance(target, list):
        p = [predict(classifier['classifier'], language, t, code) for t in target]
        return [int(1 not in p)] + p
    raise TypeError(type(target))
