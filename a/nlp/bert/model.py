from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths

model_path = get_pretrained( PretrainedList.chinese_base )
paths = get_checkpoint_paths(model_path)
print(paths.config, paths.checkpoint, paths.vocab)