import csv
import re
import emoji
import sys
import os
from datetime import datetime
import logging
import torch
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification, AdamW, BertConfig
from tqdm import tqdm, trange
import numpy as np
import time
import random
import datetime
from _datetime import datetime as dt
from sklearn.metrics import classification_report
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

# Logger stuff
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger(__name__)


def load_train_data(input_file_path, tokenizer):
    '''
    Function to load training data. The input format is one tweet per line:
    tweet_id \t tweet text \t OFF/NOT
    :param language: the tweets language, it is used just for paths, can be removed
    :param tokenizer: BERT tokenizer, output of the training code
    :return: the list of
        train_input_ids, train_labels, train_attention_masks
        which stand for tokenized tweet texts, labels and computed attention mask for training data
    '''

    # List of all tweets text
    train_tweets = []
    # List of all labels
    train_labels = []

    # -----------------------------------------------------------------
    # Parse Training Set
    with open(input_file_path) as input_file:
        # For each tweet
        count = 0
        for line in csv.reader(input_file, delimiter="\t"):
            if line[0] != 'id' and len(line) == 3:
                full_line = line[1]
                full_line = re.sub(r'#([^ ]*)', r'\1', full_line)
                full_line = re.sub(r'https.*[^ ]', 'URL', full_line)
                full_line = emoji.demojize(full_line)
                full_line = re.sub(r'(:.*?:)', r' \1 ', full_line)
                full_line = re.sub(' +', ' ', full_line)

                if line[2] == 'OFF':
                    label = 1
                else:
                    label = 0

                # Save tweet's text and label
                train_tweets.append(full_line)
                train_labels.append(label)

    # List of all tokenized tweets
    train_input_ids = []

    # For every tweet in the training set
    for sent in train_tweets:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=512,

            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )

        # Add the encoded tweet to the list.
        train_input_ids.append(encoded_sent)

    # # Pad our input tokens with value 0.
    # # "post" indicates that we want to pad and truncate at the end of the sequence,
    # # as opposed to the beginning.
    train_input_ids = pad_sequences(train_input_ids, maxlen=512, dtype="long",
                          value=tokenizer.pad_token_id, truncating="pre", padding="pre")

    # Create attention masks
    # The attention mask simply makes it explicit which tokens are actual words versus which are padding
    train_attention_masks = []

    # For each tweet in the training set
    for sent in train_input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        train_attention_masks.append(att_mask)

    # Return the list of encoded tweets, the list of labels and the list of attention masks
    return train_input_ids, train_labels, train_attention_masks


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# ======================================================================================================================
# Part of the code comes from https://mccormickml.com/2019/07/22/BERT-fine-tuning/
# ======================================================================================================================
# ---------------------------- Main ----------------------------

if len(sys.argv) < 4:
    print("Missing arguments!")
    print("Usage:")
    print("python fine_tune_language_model_no_validation.py /path/to/pre-traines/model /path/to/output/dir /path/to/input/training/file")
    print("Example:")
    print("python fine_tune_language_model_no_validation.py ../data/en/a/pretrained_lm ../data/en/a/finetuned_text_classification ../data/en/a/fine-tuning/fine-tuning_data.tab")


# Pre-trained model path
model_dir = sys.argv[1]

# Output model path
output_model_dir = sys.argv[2]
if not output_model_dir.endswith("/"):
    output_model_dir += "/"

# Input training file
input_file_path = sys.argv[3]

# Make dir for model serializations
os.makedirs(os.path.dirname(output_model_dir), exist_ok=True)

# Log stuff: print logger on file in output_model_dir/log.log
logging.basicConfig(filename=output_model_dir + 'log.log', level=logging.DEBUG)

# Log stuff: print logger also on stderr
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

# -----------------------------
# Load Pre-trained BERT model
# -----------------------------
config_class, model_class, tokenizer_class = (BertConfig, BertForSequenceClassification, BertTokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load a trained model and vocabulary pre-trained for specific language
logger.info("Loading model from: '" + model_dir + "', it may take a while...")

# Load pre-trained Tokenizer from directory, change this to load a tokenizer from ber package
tokenizer = tokenizer_class.from_pretrained(model_dir)

# Load Bert for classification 'container'
model = BertForSequenceClassification.from_pretrained(
    model_dir, # Use pre-trained model from its directory, change this to use a pre-trained model from bert
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Set the model to work on CPU if no GPU is present
model.to(device)
logger.info("Bert for classification model has been loaded!")

# --------------------------------------------------------------------
# ---------- Print BERT model list of parameters and layers ----------
# --------------------------------------------------------------------
# The list of prints can be safely removed

# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

logger.info('The BERT model has {:} different named parameters.\n'.format(len(params)))

logger.info('==== Embedding Layer ====\n')

for p in params[0:5]:
    logger.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

logger.info('\n==== First Transformer ====\n')

for p in params[5:21]:
    logger.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

logger.info('\n==== Output Layer ====\n')

for p in params[-4:]:
    logger.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# --------------------------------------------------------------------
# -------------------------- Load test data --------------------------
# --------------------------------------------------------------------

# The loading eval data return:
# - input_ids:         the list of all tweets already tokenized and ready for bert (with [CLS] and [SEP])
# - labels:            the list of labels, the i-th index corresponds to the i-th position in input_ids
# - attention_masks:   a list of [0,1] for every input_id that represent which token is a padding token and which is not

# Load Offenseval 2018, Train,Test already splitted into Train/Test set
train_inputs, train_labels, train_masks = load_train_data(input_file_path, tokenizer)
# --------------------------------------------------------------------
# -------------------- Split train and validation --------------------
# --------------------------------------------------------------------
#
# If the dataset is not partitioned into Train/Test we have to split it
# Split train and validation
# Use 90% for training and 10% for validation.
# train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=42, test_size=0.1, stratify=labels)

# Do the same for the masks.
# train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state=42, test_size=0.1, stratify=labels)

# Convert all inputs and labels into torch tensors, the required datatype for our model.

# Tweets
train_inputs = torch.tensor(train_inputs)

# Labels
train_labels = torch.tensor(train_labels)

# Attention masks
train_masks = torch.tensor(train_masks)

# We will use a DataLoader, it helps save on memory during training because, unlike a for loop, with an iterator
# the entire dataset does not need to be loaded into memory
# The DataLoader needs to know our batch size for training, so we specify it here.
# For fine-tuning BERT on a specific task, the authors recommend a batch size of
# 16 or 32.

batch_size = 32

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# --------------------------------------------------------------------
# -------------- Optimizer and Learning Rate Scheduler ---------------
# --------------------------------------------------------------------
# For the purposes of fine-tuning, the authors recommend choosing from the following values:

# Batch size: 16, 32 (We chose 32 when creating our DataLoaders).
# Learning rate (Adam): 5e-5, 3e-5, 2e-5 (We’ll use 2e-5).
# Number of epochs: 2, 3, 4 (We’ll use 4).
#
#
# Note: AdamW is a class from the HuggingFace library (as opposed to PyTorch)
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                )

# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

# --------------------------------------------------------------------
# Now we are ready to prepare and run the training/evaluation
# --------------------------------------------------------------------
#
# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []

# For each epoch...
for epoch_i in tqdm(range(0, epochs), desc="Training"):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    logger.info("")
    logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    logger.info('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    # the tqdm instruction mess with prints on terminal but it can be useful to understand what is the current
    # batch at any time
    for step, batch in tqdm(enumerate(train_dataloader), desc="Batch"):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            logger.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple.
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    logger.info("")
    logger.info("  Average training loss: {0:.2f}".format(avg_train_loss))
    logger.info("  Training epcoh took: {:}".format(format_time(time.time() - t0)))


logger.info("")
logger.info("Training complete!")

logger.info("Saving model to: " + output_model_dir)
logger.info("# Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()")
# Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = (
    model.module if hasattr(model, "module") else model
)  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
