import logging
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import TruncatedNormal
from transformers import BertTokenizerFast, TFBertForSequenceClassification, BertConfig, TFBertModel

train = pd.read_csv("mid_data/mid_train_data.csv", header=0)
test = pd.read_csv("mid_data/mid_trail_data.csv", header=0)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))

    train_texts1, train_texts2, test_texts1, test_texts2 = [], [], [], []
    train_total_marks, train_time_marks, train_Geography_marks, train_tone_marks, train_entities_marks = [], [], [], [], []
    train_narrative_marks, train_Style_marks = [], []

    for i in range(0, len(train)):
        train_texts1.append(str(train["text1"][i]))
        train_texts2.append(str(train["text2"][i]))
        train_total_marks.append(train["Overall"][i])
        train_time_marks.append(train["Time"][i])
        train_Geography_marks.append(train["Geography"][i])
        train_tone_marks.append(train["Tone"][i])
        train_entities_marks.append(train["Entities"][i])
        train_narrative_marks.append(train["Narrative"][i])
        train_Style_marks.append(train["Style"][i])

    for i in range(0, len(test)):
        test_texts1.append(str(test["text1"][i]))
        test_texts2.append(str(test["text2"][i]))

    model_name = 'bert-base-uncased'
    config = BertConfig.from_pretrained(model_name)
    config.output_hidden_states = False
    # max_length = 100;

    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=model_name, config=config)

    train_encodings = tokenizer(train_texts2, train_texts1, truncation=True, padding="max_length")
    # val_encodings = tokenizer(val_texts2, val_texts1, truncation=True, padding="max_length")
    test_encodings = tokenizer(test_texts2, test_texts1, truncation=True, padding="max_length")
    print(type(train_encodings['input_ids']))

    transformer_model = TFBertModel.from_pretrained(model_name, config=config)

    bert = transformer_model.layers[0]

    input_ids = Input(shape=(None,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(None,), name='attention_mask', dtype='int32')
    token_type_ids = Input(shape=(None,), name='token_type_ids', dtype='int32')
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

    bert_model = bert(inputs)[1]
    dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
    pooled_output = dropout(bert_model, training=False)

    overall = Dense(units=1, kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
                    name="Overall")(pooled_output)
    time = Dense(units=1, kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
                    name="Time")(pooled_output)
    geography = Dense(units=1, kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
                    name="Geography")(pooled_output)
    Tone = Dense(units=1, kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
                 name="Tone")(pooled_output)
    Entities = Dense(units=1, kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
                     name="Entities")(pooled_output)
    Narrative = Dense(units=1, kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
                      name="Narrative")(pooled_output)
    Style = Dense(units=1, kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
                  name="Style")(pooled_output)

    outputs = {'Overall': overall, 'Time': time, "Geography": geography, 'Tone': Tone, 'Entities': Entities,
               'Narrative': Narrative, 'Style': Style}
    model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel')
    model.summary()

    print(np.array(train_encodings['input_ids']))

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6, epsilon=1e-8)
    loss = tf.keras.losses.mean_squared_error
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    model.fit(x={'input_ids': np.array(train_encodings['input_ids']),
                 'attention_mask': np.array(train_encodings['attention_mask']),
                 'token_type_ids': np.array(train_encodings['token_type_ids'])},
              y={'Overall': np.array(train_total_marks), 'Time': np.array(train_time_marks),
                 "Geography": np.array(train_Geography_marks), 'Tone': np.array(train_tone_marks),
                 'Entities': np.array(train_entities_marks), 'Narrative': np.array(train_narrative_marks),
                 'Style': np.array(train_Style_marks)},
              validation_split=0.1,
              batch_size=8,
              epochs=3)
    y_pred = model.predict(x={'input_ids': np.array(test_encodings['input_ids']), 'attention_mask': np.array(test_encodings['attention_mask']), 'token_type_ids': np.array(test_encodings['token_type_ids'])})

    Overall = y_pred['Overall']
    Time = y_pred['Time']
    Geography = y_pred['Geography']
    Tone = y_pred['Tone']
    Entities = y_pred['Entities']
    Narrative = y_pred['Narrative']
    Style = y_pred['Style']

    Overall = list(np.ravel(Overall))
    Time = list(np.ravel(Time))
    Geography = list(np.ravel(Geography))
    Tone = list(np.ravel(Tone))
    Entities = list(np.ravel(Entities))
    Narrative = list(np.ravel(Narrative))
    Style = list(np.ravel(Style))

    result_output = pd.DataFrame(
        data={"pair_id": test["pair_id"], "Overall": Overall, "Time": Time, "Geography": Geography, "Tone": Tone,
              'Entities': Entities, 'Narrative': Narrative, 'Style': Style})
    result_output.to_csv("result/multi_bert.csv", index=False, quoting=3)



