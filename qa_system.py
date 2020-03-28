import re
import tensorflow as tf

class qa_system:
    questions = []
    answers = []

    with open('/content/drive/My Drive/data2/dataset.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            l = line.split(' <<<>>> ')
            questions.append(preprocess_sentence(l[0]))
            answers.append(preprocess_sentence(l[1]))

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
    
    def evaluate(self, sentence):
        sentence = preprocess_sentence(sentence)

        sentence = tf.expand_dims(
            START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

        output = tf.expand_dims(START_TOKEN, 0)

        for i in range(MAX_LENGTH):
            predictions = model(inputs=[sentence, output], training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, END_TOKEN[0]):
                break

            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)


    def predict(self, sentence):
        prediction = evaluate(sentence)

        predicted_sentence = tokenizer.decode(
            [i for i in prediction if i < tokenizer.vocab_size])

        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))

        return predicted_sentence