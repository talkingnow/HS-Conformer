
from transformers import AutoConfig, Wav2Vec2Model, HubertModel, WavLMModel

# model type
W2V2 = 'W2V2'
HUBERT = 'HuBERT'
WAVLM = 'WavLM'

SUPPORT_TYPE = [W2V2, HUBERT, WAVLM]

MODEL_CLASS = {
    W2V2: Wav2Vec2Model,
    HUBERT: HubertModel,
    WAVLM: WavLMModel
}

HUGGINGFACE_URL = {
    W2V2: ['facebook/wav2vec2-base', 'facebook/wav2vec2-large-960h', 'facebook/wav2vec2-large-xlsr-53'],
    HUBERT: ['facebook/hubert-base-ls960', 'facebook/hubert-large-ll60k', 'facebook/hubert-xlarge-ll60k'],
    WAVLM: ['microsoft/wavlm-base', 'microsoft/wavlm-base-plus', 'microsoft/wavlm-large']
}

def load_ssl_model(model_type, size):
    assert model_type in SUPPORT_TYPE, f'Does not support {model_type}\n(Available: {SUPPORT_TYPE})'
    assert type(size) is int, f'type(size) must be integer. (Current input: {type(size)})'
    assert HUGGINGFACE_URL[model_type][size] is not None, 'Does not support given size'
    
    name = HUGGINGFACE_URL[model_type][size]
    config = AutoConfig.from_pretrained(name)
    
    model = MODEL_CLASS[model_type].from_pretrained(
        name,
        from_tf=bool(".ckpt" in name),
        config=config,
        revision="main",
        ignore_mismatched_sizes=False,
    )
    
    print('###############################\n')
    print('Model name:', model_type)
    print('  num_hidden_layers:', config.num_hidden_layers)
    print('  hidden_size:', config.hidden_size)
    print('\n###############################')
    
    return model