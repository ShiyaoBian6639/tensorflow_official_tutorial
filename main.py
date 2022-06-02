import tensorflow as tf
import numpy as np
from text_generation.utils.data_loader import load_data, path_to_file
from text_generation.utils.text_process import text_vectorization
from text_generation.model.train import TrainTranslator
from text_generation.model.loss import MaskedLoss
from text_generation.model.log import BatchLogs
from text_generation.model.translate import Translator
from text_generation.utils.attention_plots import plot_attention
from tensorflow.python.training import checkpoint_utils as cp

# model config
embedding_dim = 256
units = 1024

# load dataset
targ, inp = load_data(path_to_file)

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

# translate
translator = Translator(
    encoder=train_translator.encoder,
    decoder=train_translator.decoder,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor,
)

# sample translation
input_text = tf.constant([
    'hace mucho frio aqui.',  # "It's really cold here."
    'Esta es mi vida.',  # "This is my life.""
])

result = translator.tf_translate(input_text=input_text)

print(result['text'][0].numpy().decode())
print(result['text'][1].numpy().decode())
print()

a = result['attention'][0]

print(np.sum(a, axis=-1))
i = 1
plot_attention(result['attention'][i], input_text[i], result['text'][i])


def get_config(self):
    config = super(TrainTranslator, self).get_config()
    config.update({"units": self.units})
    return config


def from_config(cls, config):
    return cls(**config)


TrainTranslator.get_config = get_config

# save model weights
train_translator.save("eng-spa")
train_translator.save("eng-spa1.hdf5", save_format="tf")

model_idx = cp.list_variables("./")
new_model = cp.load_variable("./", '_CHECKPOINTABLE_OBJECT_GRAPH')
