import tensorflow as tf
from text_generation.utils.data_loader import load_data, path_to_file
from text_generation.utils.text_process import text_vectorization

# model config
embedding_dim = 256
units = 1024

# load dataset
targ, inp = load_data(path_to_file)
targ = targ[:100]
inp = inp[:100]
# create tf data set
BUFFER_SIZE = len(inp)
BATCH_SIZE = 256

dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)

# text to token
input_text_processor, output_text_processor = text_vectorization(targ, inp)

# get example input batch and target batch
for example_input_batch, example_target_batch in dataset.take(1):
    print(example_input_batch[:5])
    print()
    print(example_target_batch[:5])
    break

example_tokens = input_text_processor(example_input_batch)
