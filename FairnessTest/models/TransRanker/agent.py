import importlib

my_module = importlib.import_module(".transformer")
model_class = getattr(my_module, "TransformerRankerAgent")

opt = {'init_opt': None, 'show_advanced_args': False, 'task': 'twitter',
       'download_path': '/mnt/ufs18/home-118/liuhaoc1/ParlAI/downloads', 'datatype': 'train', 'image_mode': 'raw', 'numthreads': 1,
       'hide_labels': False, 'multitask_weights': [1], 'batchsize': 10, 'model': 'transformer/ranker',
       'model_file': '/mnt/home/liuhaoc1/ParlAI/models/trans_twitter/trans_twitter', 'init_model': None,
       'dict_class': 'parlai.core.dict:DictionaryAgent', 'evaltask': None, 'eval_batchsize': None, 'display_examples': False,
       'num_epochs': -1, 'max_train_time': -1, 'validation_every_n_secs': 1800.0, 'save_every_n_secs': -1,
       'save_after_valid': False, 'validation_every_n_epochs': -1, 'validation_max_exs': -1, 'short_final_eval': False,
       'validation_patience': 10, 'validation_metric': 'accuracy', 'validation_metric_mode': None, 'validation_cutoff': 1.0,
       'load_from_checkpoint': False, 'validation_share_agent': False, 'aggregate_micro': False, 'metrics': 'default',
       'tensorboard_log': False, 'pytorch_teacher_task': None, 'pytorch_teacher_dataset': None, 'pytorch_datapath': None,
       'numworkers': 4, 'pytorch_preprocess': False, 'pytorch_teacher_batch_sort': False, 'batch_sort_cache_type': 'pop',
       'batch_length_range': 5, 'shuffle': False, 'batch_sort_field': 'text', 'pytorch_context_length': -1,
       'pytorch_include_labels': True, 'dict_maxexs': -1, 'dict_include_valid': False, 'dict_include_test': False,
       'log_every_n_secs': 2, 'image_size': 256, 'image_cropsize': 224, 'embedding_type': 'random', 'embedding_projection': 'random',
       'fp16': False, 'optimizer': 'adamax', 'learningrate': 0.0001, 'gradient_clip': 0.1, 'adam_eps': 1e-08, 'momentum': 0,
       'nesterov': True, 'nus': [0.7], 'betas': [0.9, 0.999], 'weight_decay': None, 'lr_scheduler': 'reduceonplateau',
       'lr_scheduler_patience': 3, 'lr_scheduler_decay': 0.5, 'warmup_updates': -1, 'warmup_rate': 0.0001, 'update_freq': 1,
       'rank_candidates': True, 'truncate': 1024, 'text_truncate': None, 'label_truncate': None, 'history_size': -1,
       'person_tokens': False, 'split_lines': False, 'use_reply': 'label', 'add_p1_after_newln': False, 'delimiter': '\n',
       'gpu': -1, 'no_cuda': False, 'candidates': 'batch', 'eval_candidates': 'fixed', 'repeat_blocking_heuristic': True,
       'fixed_candidates_path': '/mnt/home/liuhaoc1/ParlAI/twitter_500k.cands', 'fixed_candidate_vecs': 'reuse',
       'encode_candidate_vecs': True, 'train_predict': False, 'cap_num_predictions': 100, 'ignore_bad_candidates': True,
       'rank_top_k': -1, 'embedding_size': 300, 'n_layers': 2, 'ffn_size': 300, 'dropout': 0.0, 'attention_dropout': 0.0,
       'relu_dropout': 0.0, 'n_heads': 2, 'learn_positional_embeddings': False, 'embeddings_scale': True, 'n_positions': None,
       'n_segments': 0, 'variant': 'aiayn', 'activation': 'relu', 'output_scaling': 1.0, 'use_memories': False,
       'wrap_memory_encoder': False, 'memory_attention': 'sqrt', 'normalize_sent_emb': False, 'share_encoders': True,
       'share_word_embeddings': True, 'learn_embeddings': True, 'data_parallel': True, 'reduction_type': 'mean',
       'dict_file': '/mnt/home/liuhaoc1/ParlAI/models/trans_twitter/trans_twitter.dict', 'dict_initpath': None,
       'dict_language': 'english', 'dict_max_ngram_size': -1, 'dict_minfreq': 0, 'dict_maxtokens': 30000,
       'dict_nulltoken': '__null__', 'dict_starttoken': '__start__', 'dict_endtoken': '__end__', 'dict_unktoken': '__unk__',
       'dict_tokenizer': 're', 'dict_lower': False, 'bpe_debug': False, 'dict_textfields': 'text,labels',
       'parlai_home': '/mnt/ufs18/home-118/liuhaoc1/ParlAI',
       'override': {'task': 'twitter', 'model_file': 'models/trans_twitter/trans_twitter', 'model': 'transformer/ranker',
                    'batchsize': 10, 'validation_every_n_secs': 1800.0, 'candidates': 'batch', 'eval_candidates': 'batch',
                    'data_parallel': True, 'validation_patience': 10, 'datapath': '/mnt/home/liuhaoc1/ParlAI/data'},
       'starttime': 'Dec27_21-43', 'datapath': '/mnt/home/liuhaoc1/ParlAI/data', 'num_examples': 10,
       'display_ignore_fields': '', 'interactive_mode': False}

agent = model_class(opt)

def get_response(context_batch):
    batch_observation = []
    for context in context_batch:
        obs = {'text': context, 'episode_done': True, 'id': 'input', 'eval_labels': ['null']}
        observation = agent.observe(obs)
        batch_observation.append(observation)
    batch_actions = agent.batch_act(batch_observation)

    response_batch = [result['text'] for result in batch_actions]

    return response_batch