opt = {'task': 'twitter',
       'download_path': '/mnt/home/liuhaoc1/ParlAI/downloads',
       'datatype': 'train',
       'image_mode': 'raw',
       'numthreads': 1,
       'hide_labels': False,
       'batchsize': 32,
       'batch_sort': True,
       'context_length': -1,
       'include_labels': True,
       'datapath': '/mnt/home/liuhaoc1/ParlAI/augmented_data',
       'model': 'legacy:seq2seq:0',
       'model_file': '/mnt/home/liuhaoc1/ParlAI/models/seq_twitter_aug/seq_twitter_aug',
       'dict_class': '',
       'evaltask': None,
       'display_examples': False,
       'num_epochs': -1,
       'max_train_time': 205200.0,
       # 'validation_every_n_secs': 600.0,
       'save_every_n_secs': -1,
       'save_after_valid': True,
       # 'validation_max_exs': -1,
       # 'validation_patience': 18,
       # 'validation_metric': 'ppl',
       # 'validation_metric_mode': 'min',
       # 'validation_cutoff': 1.0,
       'dict_build_first': True,
       'load_from_checkpoint': True,
       'tensorboard_log': False,
       'tensorboard_tag': None,
       'tensorboard_metrics': None,
       'tensorboard_comment': '',
       'dict_maxexs': -1,
       'dict_include_valid': False,
       'dict_include_test': False,
       'log_every_n_secs': 15.0,
       'image_size': 256,
       'image_cropsize': 224,
       'init_model': None,
       'hiddensize': 1024,
       'embeddingsize': 300,
       'numlayers': 3,
       'learningrate': 1.0,
       'dropout': 0.0,
       'gradient_clip': 0.1,
       'bidirectional': False, 'attention': 'none',
       'attention_length': 48, 'attention_time': 'post',
       'no_cuda': False, 'gpu': -1,
       'rank_candidates': False, 'truncate': 150,
       'rnn_class': 'lstm', 'decoder': 'same',
       'lookuptable': 'enc_dec', 'optimizer': 'sgd',
       'momentum': 0.9, 'embedding_type': 'random',
       'numsoftmax': 1, 'report_freq': 0.001,
       'history_replies': 'label_else_model',
       'person_tokens': False,
       'dict_file': '',
       'dict_initpath': None,
       'dict_language': 'english',
       'dict_max_ngram_size': -1, 'dict_minfreq': 0,
       'dict_maxtokens': 30004, 'dict_nulltoken': '__NULL__',
       'dict_starttoken': '__START__', 'dict_endtoken': '__END__',
       'dict_unktoken': '__UNK__', 'dict_tokenizer': 're', 'dict_lower': True,
       'parlai_home': '/mnt/home/liuhaoc1/ParlAI',
       # 'override': {'task': 'twitter', 'max_train_time': '205200', 'model': 'seq2seq', 'numsoftmax': '1', 'hiddensize': '1024', 'embeddingsize': '300', 'attention': 'none', 'numlayers': '3', 'rnn_class': 'lstm', 'learningrate': '1',
       #              'dropout': '0.0',
       #              'gradient_clip': '0.1', 'lookuptable': 'enc_dec', 'optimizer': 'sgd', 'embedding_type': 'glove', 'momentum': '0.9', 'batchsize': '32', 'batch_sort': 'True', 'truncate': '150', 'validation_every_n_secs': '600', 'validation_metric': 'ppl', 'validation_metric_mode': 'min', 'validation_patience': '18', 'save_after_valid': 'True', 'load_from_checkpoint': 'True', 'dict_lower': 'True', 'dict_maxtokens': '30000', 'log_every_n_secs': '15', 'model_file': '/mnt/home/liuhaoc1/ParlAI/models/seq_twitter_aug/seq_twitter_aug'},
       'starttime': 'Jun15_16-53', 'show_advanced_args': False,
       'pytorch_teacher_task': None, 'pytorch_teacher_dataset': None,
       'pytorch_datapath': None, 'numworkers': 4, 'pytorch_preprocess': False,
       'pytorch_teacher_batch_sort': False, 'batch_sort_cache_type': 'pop',
       'batch_length_range': 5, 'shuffle': False, 'batch_sort_field': 'text',
       'pytorch_context_length': -1, 'pytorch_include_labels': True, 'num_examples': -1,
       'metrics': 'all', 'beam_size': 1, 'beam_log_freq': 0.0, 'topk': 1,
       'softmax_layer_bias': False, 'bpe_debug': False, 'dict_textfields': 'text,labels'}