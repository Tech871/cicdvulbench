import torch

from storage import get_dir
from classifier import GNNReGVD, DefaultClassifier

from transformers import BertConfig, BertForSequenceClassification
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import OpenAIGPTConfig, OpenAIGPTLMHeadModel
from transformers import RobertaConfig, RobertaForSequenceClassification
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from tokenizer import load_pretrained_tokenizer

models = {
    'gpt2': GPT2LMHeadModel,
    'openai-gpt': OpenAIGPTLMHeadModel,
    'bert': BertForSequenceClassification,
    'roberta': RobertaForSequenceClassification,
    'distilbert': DistilBertForSequenceClassification
}

transformers = {
    'gpt2': 'transformer',
    'openai-gpt': 'transformer',
    'bert': 'bert',
    'roberta': 'roberta',
    'distilbert': 'distilbert'
}

configs = {
    'gpt2': GPT2Config,
    'openai-gpt': OpenAIGPTConfig,
    'bert': BertConfig,
    'roberta': RobertaConfig,
    'distilbert': DistilBertConfig
}


def get_output_dir(language, target, source, classifier):
    return get_dir(language, target, where='log', source=source, classifier=classifier)


def load_pretrained_config(classifier, target, tokenizer, cache_dir=None):
    Config = configs[classifier['classifier']]
    config = Config.from_pretrained(
        classifier['pretrained_model_path'],
        cache_dir=cache_dir
    )
    config.num_labels = classifier.get('num_labels') or len(target) + 1
    config.pad_token_id = tokenizer.eos_token_id
    return config


def load_pretrained_model(classifier, target, args):
    tokenizer = load_pretrained_tokenizer(classifier, args.output_dir)
    config = load_pretrained_config(classifier, target, tokenizer)

    Model = models[classifier['classifier']]
    if classifier['pretrained_model_path']:
        from_tf = '.ckpt' in classifier['pretrained_model_path']
        model = Model.from_pretrained(
            classifier['pretrained_model_path'],
            from_tf=from_tf,
            config=config,
            cache_dir=args.output_dir
        )
    else:
        model = Model(config)

    dropout_probability = classifier.get('dropout_probability', 0)
    layout = classifier.get('layout')
    if layout:
        transformer = getattr(model, transformers[classifier['classifier']])
        model = GNNReGVD(model, tokenizer, transformer, args.device, layout, dropout_probability)
    else:
        model = DefaultClassifier(model, tokenizer, dropout_probability)

    return model, tokenizer


def save_model(model, output_dir):
    model = model.module if hasattr(model, 'module') else model
    torch.save(model.state_dict(), f'{output_dir}/model.bin')


def load_model(output_dir, model):
    model.load_state_dict(torch.load(f'{output_dir}/model.bin'))


def load_saved_model(classifier, target, args):
    model, tokenizer = load_pretrained_model(classifier, target, args)
    load_model(args.output_dir, model)

    return model, tokenizer
