import tensorflow as tf
from text_generation.utils.data_loader import load_data, path_to_file
from text_generation.utils.text_process import text_vectorization
from text_generation.model.train import TrainTranslator
from text_generation.model.loss import MaskedLoss
from text_generation.model.log import BatchLogs

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

train_translator = TrainTranslator(
    embedding_dim, units,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor)
train_translator.use_tf_function = True

# Configure the loss and optimizer
train_translator.compile(
    optimizer=tf.optimizers.Adam(),
    loss=MaskedLoss(),
)
batch_loss = BatchLogs('batch_loss')
train_translator.fit(dataset, epochs=3,
                     callbacks=[batch_loss])
