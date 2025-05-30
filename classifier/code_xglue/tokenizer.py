from transformers import BertTokenizer, GPT2Tokenizer, OpenAIGPTTokenizer, RobertaTokenizer, DistilBertTokenizer

tokenizers = {
    'gpt2': GPT2Tokenizer,
    'openai-gpt': OpenAIGPTTokenizer,
    'bert': BertTokenizer,
    'roberta': RobertaTokenizer,
    'distilbert': DistilBertTokenizer
}


def load_pretrained_tokenizer(classifier, cache_dir=None, do_lower_case=True, num_tokens=0):
    Tokenizer = tokenizers.get(classifier['classifier'])
    tokenizer = Tokenizer.from_pretrained(
        classifier['pretrained_model_path'],
        do_lower_case=do_lower_case,
        cache_dir=cache_dir
    )
    if num_tokens > 0:
        tokenizer.num_tokens = min(num_tokens, tokenizer.max_len_single_sentence)
    else:
        tokenizer.num_tokens = tokenizer.max_len_single_sentence
    return tokenizer
