import re
import tensorflow as tf
import tensorflow_datasets as tfds

# Setting up the QA System as an object which can be used in chatbot.py
class QA_System:
    # Settings up lists of questions and answers
    questions = []
    answers = []
    # Default model parameneters
    MAX_LENGTH = 40
    NUM_LAYERS = 2
    D_MODEL = 256
    NUM_HEADS = 8
    UNITS = 512
    DROPOUT = 0.1

    def __init__(self):
        # Load the questions and answers from tbbt2.txt first
        self.load_data(filename='./qa_system/tbbt2.txt')
        # Create the vocabulary by tokenizing the given qa-pairs
        self.tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(self.questions + self.answers, target_vocab_size=2**13)
        self.START_TOKEN, self.END_TOKEN = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1]
        self.VOCAB_SIZE = self.tokenizer.vocab_size + 2

        # Creating the transformer model
        # Trabsformer model used from https://github.com/tensorflow/examples/blob/master/community/en/transformer_chatbot.ipynb
        self.model = self.transformer(
        vocab_size=self.VOCAB_SIZE,
        num_layers=self.NUM_LAYERS,
        units=self.UNITS,
        d_model=self.D_MODEL,
        num_heads=self.NUM_HEADS,
        dropout=self.DROPOUT)

        # Setting up the learning rate and adam optimiser
        learning_rate = CustomSchedule(self.D_MODEL)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.model.compile(optimizer=optimizer, loss=self.loss_function, metrics=[self.accuracy])
        # Loading saved model weights 
        self.model.load_weights('./qa_system/weights_tbbt2_3.h5')
        

    # Function to load the questions and answers from a text file
    # There's a QA pair in every text line
    # Questions and answers are separated by ' <<<>>> ' string
    def load_data(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                l = line.split(' <<<>>> ')
                self.questions.append(self.preprocess_sentence(l[0]))
                self.answers.append(self.preprocess_sentence(l[1]))

    # Preprocessing function adapted from https://github.com/tensorflow/examples/blob/master/community/en/transformer_chatbot.ipynb
    def preprocess_sentence(self, sentence):
        sentence = sentence.lower().strip()
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
        sentence = sentence.strip()
        # adding a start and an end token to the sentence
        return sentence

    # Creating a postprocessing funciton to detokenize the output and format the sentence
    # Function will properly capitalize all the output sentences, fix the punctuation spacings
    # and fix the apostrophes
    def postprocess_sentence(self, sentence):
        sentence = sentence.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
        sentence = sentence.replace(" ( ", " (").replace(" ) ", ") ")
        sentence = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", sentence)
        sentence = re.sub(r' ([.,:;?!%]+)$', r"\1", sentence)
        sentence = sentence.replace(" '", "'").replace(" n't", "n't").replace('i m', "I'm").replace('I m', "I'm").replace(' s ', "'s ").replace(' re ', "'re ").replace(
         "can not", "cannot").replace(' t ', "'t ")
        sentence = sentence.replace(" ` ", " '")
        split = re.split('([.!?] *)', sentence)
        sentence = ''.join([sen.capitalize() for sen in split])
        sentence = sentence.replace(" i ", " I ")
        return sentence.strip()


    # Evaluation function adapted from the above mentioned chatbot
    def evaluate(self, sentence):
        sentence = self.preprocess_sentence(sentence)
        sentence = tf.expand_dims(
            self.START_TOKEN + self.tokenizer.encode(sentence) + self.END_TOKEN, axis=0)
        output = tf.expand_dims(self.START_TOKEN, 0)
        for i in range(self.MAX_LENGTH):
            predictions = self.model(inputs=[sentence, output], training=False)
            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, self.END_TOKEN[0]):
                break
            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)

    # Predict function which uses the trained model to predict the output 
    # and then post-processes the output to make it viable to show in the Chatbot conversation
    def predict(self, sentence):
        prediction = self.evaluate(sentence)
        predicted_sentence = self.tokenizer.decode(
            [i for i in prediction if i < self.tokenizer.vocab_size])
        print(predicted_sentence)
        return self.postprocess_sentence(predicted_sentence)

    # Following Transformer layers functions are adapted from the above mentioned transformer model
    # Transformer layers have been improved with the get_config function which makes the transformer model 
    # easily save-able in future versions of tensorflow and the will be able to be saved with model.save() function and loaded with load_model() function
    # (saving subclassed models is planned to be integrated into tf)
    # However, this chatbot uses tensorflow 2.0 (in which subclassing serialization is unsupported),
    # and the model has to be recreated and loaded with saved weights 
    def create_padding_mask(self, x):
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        # (batch_size, 1, 1, sequence length)
        return mask[:, tf.newaxis, tf.newaxis, :]

    # Creating look ahead mask
    def create_look_ahead_mask(self, x):
        seq_len = tf.shape(x)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        padding_mask = self.create_padding_mask(x)
        return tf.maximum(look_ahead_mask, padding_mask)

    # Defining the transformer model
    def transformer(self, vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
        inputs = tf.keras.Input(shape=(None,), name="inputs")
        dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

        enc_padding_mask = tf.keras.layers.Lambda(
            self.create_padding_mask, output_shape=(1, 1, None),
            name='enc_padding_mask')(inputs)
        # mask the future tokens for decoder inputs at the 1st attention block
        look_ahead_mask = tf.keras.layers.Lambda(
            self.create_look_ahead_mask,
            output_shape=(1, None, None),
            name='look_ahead_mask')(dec_inputs)
        # mask the encoder outputs for the 2nd attention block
        dec_padding_mask = tf.keras.layers.Lambda(
            self.create_padding_mask, output_shape=(1, 1, None),
            name='dec_padding_mask')(inputs)

        enc_outputs = self.encoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )(inputs=[inputs, enc_padding_mask])

        dec_outputs = self.decoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

        outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

        return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

    # Creating the encoder layer
    def encoder_layer(self, units, d_model, num_heads, dropout, name="encoder_layer"):
        inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
        attention = MultiHeadAttention(
            d_model, num_heads, name="attention")({
                'query': inputs,
                'key': inputs,
                'value': inputs,
                'mask': padding_mask
            })
        attention = tf.keras.layers.Dropout(rate=dropout)(attention)
        attention = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(inputs + attention)
        outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
        outputs = tf.keras.layers.Dense(units=d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention + outputs)
        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=name)

    # Creating the  encoder
    def encoder(self, vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
        inputs = tf.keras.Input(shape=(None,), name="inputs")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
        embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
        outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

        for i in range(num_layers):
            outputs = self.encoder_layer(
                units=units,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                name="encoder_layer_{}".format(i),
            )([outputs, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=name)

    # Creating the decoder layer
    def decoder_layer(self, units, d_model, num_heads, dropout, name="decoder_layer"):
        inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
        enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name="look_ahead_mask")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

        attention1 = MultiHeadAttention(
            d_model, num_heads, name="attention_1")(inputs={
                'query': inputs,
                'key': inputs,
                'value': inputs,
                'mask': look_ahead_mask
            })
        attention1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention1 + inputs)

        attention2 = MultiHeadAttention(
            d_model, num_heads, name="attention_2")(inputs={
                'query': attention1,
                'key': enc_outputs,
                'value': enc_outputs,
                'mask': padding_mask
            })
        attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
        attention2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention2 + attention1)

        outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
        outputs = tf.keras.layers.Dense(units=d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(outputs + attention2)

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs,
            name=name)

    # Creating decoder
    def decoder(self, vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
        inputs = tf.keras.Input(shape=(None,), name='inputs')
        enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name='look_ahead_mask')
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
        embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
        outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

        for i in range(num_layers):
            outputs = self.decoder_layer(
                units=units,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                name='decoder_layer_{}'.format(i),
            )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs,
            name=name)

    def loss_function(self, y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, self.MAX_LENGTH - 1))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)
        return tf.reduce_mean(loss)

    def accuracy(self, y_true, y_pred):
        # ensure labels have shape (batch_size, MAX_LENGTH - 1)
        y_true = tf.reshape(y_true, shape=(-1, self.MAX_LENGTH - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)



# Defining multi-head attention model, positional encoding and custom schedule, adapted from the previously mentioned chatbot
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)
        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        # scaled dot-product attention
        scaled_attention = self.scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)
        return outputs

    def scaled_dot_product_attention(self, query, key, value, mask):
        """Calculate the attention weights. """
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        # scale matmul_qk
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        # add the mask to zero out padding tokens
        if mask is not None:
            logits += (mask * -1e9)
        # softmax is normalized on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, value)

        return output


class PositionalEncoding(tf.keras.layers.Layer):

  def __init__(self, position, d_model):
    self.position = position
    self.d_model = d_model
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)
  
  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'position' : self.position,
        'd_model' : self.d_model
    })
    return config

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)
    # apply sin to even index in the array
    sines = tf.math.sin(angle_rads[:, 0::2])
    # apply cos to odd index in the array
    cosines = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'd_model': self.d_model,
        'warmup_steps': self.warmup_steps,
    })
    return config 

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)