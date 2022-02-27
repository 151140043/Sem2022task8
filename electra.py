import logging
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import ElectraTokenizerFast, TFElectraForSequenceClassification


train = pd.read_csv("mid_data/mid_train_data3.csv", header=0)
test = pd.read_csv("mid_data/mid_test_data.csv", header=0)


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))

    train_ratio = 0.9
    train_texts1, train_texts2, train_total_marks, val_texts1, val_texts2, val_total_marks, test_texts1, test_texts2 = [], [], [], [], [], [], [], []
    for i in range(0, len(train)):
        if np.random.rand() < train_ratio:
            train_texts1.append(str(train["text1"][i]))
            train_texts2.append(str(train["text2"][i]))
            train_total_marks.append(train["Overall"][i])
        else:
            val_texts1.append(str(train["text1"][i]))
            val_texts2.append(str(train["text2"][i]))
            val_total_marks.append(train["Overall"][i])

    for i in range(0, len(test)):
        test_texts1.append(str(test["text1"][i]))
        test_texts2.append(str(test["text2"][i]))

    print(val_total_marks)
    tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-base-discriminator')
    train_encodings = tokenizer(train_texts1, train_texts2, truncation=True, padding="max_length")
    val_encodings = tokenizer(val_texts1, val_texts2, truncation=True, padding="max_length")
    test_encodings = tokenizer(test_texts1, test_texts2, truncation=True, padding="max_length")

    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_total_marks
    ))

    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        val_total_marks
    ))

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
    ))

    model = TFElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator', finetuning_task='text-classification',
                                                            num_labels=1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6, epsilon=1e-8)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = tf.keras.losses.mean_squared_error
    # loss = tf.keras.losses.cosine_similarity()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # can also use any keras loss fn
    model.fit(train_dataset.shuffle(1000).batch(4), validation_data=val_dataset.batch(8), epochs=3, batch_size=8)

    y_pred = model.predict(test_dataset.batch(4))[0]

    y_pred = list(np.ravel(y_pred))

    result_output = pd.DataFrame(data={"id": test["pair_id"], "similarity": y_pred})
    result_output.to_csv("result/electra_new.csv", index=False, quoting=3)