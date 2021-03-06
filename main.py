import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import collections
import random
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
from tqdm import tqdm
import pickle

import shutil

from CNN_Encoder import CNN_Encoder
from RNN_Decoder import RNN_Decoder


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def create_training_data(top_k):

    # Download caption annotation files
    annotation_folder = '/annotations/'

    root_dir = "/database/wasp_mod_2"

    if not os.path.exists(os.path.abspath(root_dir) + annotation_folder):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                 cache_subdir=os.path.abspath(
                                                     root_dir),
                                                 origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                 extract=True)
        annotation_file = os.path.dirname(
            annotation_zip)+'/annotations/captions_train2014.json'
        os.remove(annotation_zip)

    else:
        annotation_file = os.path.join(
            root_dir, 'annotations/captions_train2014.json')

    # Download image files
    image_folder = '/train2014/'
    if not os.path.exists(os.path.abspath(root_dir) + image_folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath(
                                                root_dir),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        PATH = os.path.dirname(image_zip) + image_folder
        os.remove(image_zip)
    else:
        PATH = os.path.abspath(root_dir) + image_folder

    # subsample dataset

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
        image_path_to_caption[image_path].append(caption)

    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)

    # Select the first 6000 image_paths from the shuffled set.
    # Approximately each image id has 5 captions associated with it, so that will
    # lead to 30,000 examples.
    train_image_paths = image_paths[:6000]
    print(len(train_image_paths))

    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    hidden_layer = tf.keras.layers.MaxPool2D(
        pool_size=(8, 8), padding='valid')(hidden_layer)

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # Get unique images
    encode_train = sorted(set(img_name_vector))

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        # batch_features = tf.reshape(batch_features,
        #                            (batch_features.shape[0], batch_features.shape[1]*batch_features.shape[2]*batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

    # Find the maximum length of any caption in our dataset

    def calc_max_length(tensor):
        return max(len(t) for t in tensor)

    # Choose the top 5000 words from the vocabulary
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
        train_seqs, padding='post')

    max_length = calc_max_length(train_seqs)

    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    # Create training and validation sets using an 80-20 split randomly.
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys)*0.8)
    img_name_train_keys, img_name_val_keys = img_keys[:
                                                      slice_index], img_keys[slice_index:]

    img_name_train = []
    cap_train = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    img_name_val = []
    cap_val = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        img_name_val.extend([imgv] * capv_len)
        cap_val.extend(img_to_cap_vector[imgv])

    len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)

    return img_name_train, cap_train, img_name_val, cap_val, tokenizer, max_length


datafolder = "./data"
top_k = 5000

if not os.path.exists(datafolder):
    os.mkdir(datafolder)

    img_name_train, cap_train, img_name_val, cap_val, tokenizer, max_length = create_training_data(
        top_k)

    # Save a dictionary into a pickle file.

    dataset = {"img_name_train": img_name_train,
               "cap_train": cap_train,
               "img_name_val": img_name_val,
               "cap_val": cap_val,
               "tokenizer": tokenizer,
               "max_length": max_length}

    pickle.dump(dataset, open(datafolder+"/save.pickle", "wb"))

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    hidden_layer = tf.keras.layers.MaxPool2D(
        pool_size=(8, 8), padding='valid')(hidden_layer)

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

else:
    dataset = pickle.load(open(datafolder+"/save.pickle", "rb"))

    img_name_train = dataset["img_name_train"]
    cap_train = dataset["cap_train"]
    img_name_val = dataset["img_name_val"]
    cap_val = dataset["cap_val"]
    tokenizer = dataset["tokenizer"]
    max_length = dataset["max_length"]

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    hidden_layer = tf.keras.layers.MaxPool2D(
        pool_size=(8, 8), padding='valid')(hidden_layer)

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


# Feel free to change these parameters according to your system's configuration


BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 512
units = 512
vocab_size = top_k + 1
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64

# Load the numpy files


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
    map_func, [item1, item2], [tf.float32, tf.int32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []


@tf.function
def train_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    with tf.GradientTape() as tape:

        x = encoder(img_tensor)

        _, hidden = decoder(x, hidden)

        dec_input = tf.expand_dims(
            [tokenizer.word_index['<start>']] * target.shape[0], 1)

        x = decoder.embed(dec_input)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder

            predictions, hidden = decoder(x, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

            x = decoder.embed(dec_input)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


EPOCHS = 40

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    if epoch % 5 == 0:
        ckpt_manager.save()

    print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                        total_loss/num_steps))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
# plt.show()


def evaluate(image):

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(
        img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    result = []

    x = encoder(img_tensor_val)

    _, hidden = decoder(x, hidden)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

    result.append('<start>')

    x = decoder.embed(dec_input)

    for i in range(max_length):

        predictions, hidden = decoder(x, hidden)

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        #predictions = tf.nn.softmax(predictions)

        #predicted_id = np.argmax(predictions.numpy())

        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

        x = decoder.embed(dec_input)

    return result


def plot_attention(image, result):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    plt.imshow(temp_image)

    plt.tight_layout()

    # plt.show()


# captions on the validation set
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tokenizer.index_word[i]
                         for i in cap_val[rid] if i not in [0]])
result = evaluate(image)

print('Real Caption:', real_caption)
print('Prediction Caption:', ' '.join(result))
plot_attention(image, result)

for i in range(5):

    rid = np.random.randint(0, len(img_name_val))
    image = img_name_val[rid]
    real_caption = ' '.join([tokenizer.index_word[i]
                             for i in cap_val[rid] if i not in [0]])

    img = np.array(Image.open(image))

    result = evaluate(image)

    img = Image.fromarray(img)

    name = image.split("/")[-1]

    img.save("./exampels/"+name)

    f = open("./exampels/"+name+"_pred.txt", "w")
    f.write(' '.join(result))
    f.close()

    f = open("./exampels/"+name+"_gt.txt", "w")
    f.write(real_caption)
    f.close()
