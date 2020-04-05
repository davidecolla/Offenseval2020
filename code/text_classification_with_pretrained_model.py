import re
import emoji
import os
import sys
from datetime import datetime
import logging
import torch
from sklearn import preprocessing
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification, AdamW, BertConfig
import numpy as np
import datetime
from _datetime import datetime as dt
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


def load_unlabeled_data(test_file_path, tokenizer):

    '''
    Function to load unlabeled data. The input format is one tweet per line:
    tweet_id \t tweet text
    :param test_file_path: the file with test data
    :param tokenizer: BERT tokenizer, output of the training code
    :return: the list of
        test_input_ids, test_attention_masks, test_tweet_ids
        which stand for the list of tokenized tweet texts, the list of attention masks, and the list of input tweet ids respectively
        note that the list test_tweet_ids will be used for prediction output
    '''

    # List of all tweets text
    test_tweets = []
    test_tweet_ids = []
    # Test Set
    with open(test_file_path) as input_file:
        # For each tweet
        # for line in csv.reader(input_file, delimiter="\t"):
        for line in input_file:
            line = line.split("\t")
            if line[0] != 'id':
                full_line = line[1]
                full_line = re.sub(r'#([^ ]*)', r'\1', full_line)
                full_line = re.sub(r'https.*[^ ]', 'URL', full_line)
                full_line = emoji.demojize(full_line)
                full_line = re.sub(r'(:.*?:)', r' \1 ', full_line)
                full_line = re.sub(' +', ' ', full_line)

                # Save tweet's text and label
                test_tweets.append(full_line)
                test_tweet_ids.append(line[0])

    # List of all tokenized tweets
    test_input_ids = []

    # For every tweet in the test set
    for sent in test_tweets:
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
        test_input_ids.append(encoded_sent)

    # # Pad our input tokens with value 0.
    # # "post" indicates that we want to pad and truncate at the end of the sequence,
    # # as opposed to the beginning.
    test_input_ids = pad_sequences(test_input_ids, maxlen=512, dtype="long",
                                    value=tokenizer.pad_token_id, truncating="pre", padding="pre")

    # Create attention masks
    # The attention mask simply makes it explicit which tokens are actual words versus which are padding
    test_attention_masks = []

    # For each tweet in the test set
    for sent in test_input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        test_attention_masks.append(att_mask)

    # Return the list of encoded tweets, the list of labels and the list of attention masks
    return test_input_ids, test_attention_masks, test_tweet_ids



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
    print("python text_classification_with_pretrained_model.py /path/to/fine-tuned/model /path/to/output/dir /path/to/input/eval/file")
    print("Example:")
    print("python text_classification_with_pretrained_model.py ../data/en/a/finetuned_text_classification ../data/en/a/finetuned_text_classification/eval ../data/en/a/eval_data/test_data.tab")


# Pre-trained model path
model_dir = sys.argv[1]

# Output model path
output_dir = sys.argv[2]
if not output_dir.endswith("/"):
    output_dir += "/"

# Input training file
input_file_path = sys.argv[3]

# Make dir for model serializations
os.makedirs(os.path.dirname(output_dir), exist_ok=True)

# Log stuff: print logger on file in output_dir/log.log
logging.basicConfig(filename=output_dir + 'log.log', level=logging.DEBUG)

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
# -------------------------- Load test data --------------------------
# --------------------------------------------------------------------

# The loading eval data return:
# - input_ids:         the list of all tweets already tokenized and ready for bert (with [CLS] and [SEP)
# - labels:            the list of labels, the i-th index corresponds to the i-th position in input_ids
# - attention_masks:   a list of [0,1] for every input_id that represent which token is a padding token and which is not
# input_ids, labels, attention_masks = load_eval_data(language, tokenizer)


prediction_inputs, prediction_masks, tweet_ids = load_unlabeled_data(input_file_path, tokenizer)

# Tweets
prediction_inputs = torch.tensor(prediction_inputs)

# Attention masks
prediction_masks = torch.tensor(prediction_masks)


label_encoder = preprocessing.LabelEncoder()
targets = label_encoder.fit_transform(tweet_ids)
# targets: array([0, 1, 2, 3])

prediction_ids = torch.as_tensor(targets)
# targets: tensor([0, 1, 2, 3])

# Set the batch size.
batch_size = 32

# Create the DataLoader.
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_ids)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# Put model in evaluation mode
model.eval()

# Tracking variables
predictions = []

# Predict
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_tweet_ids = batch

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)

    logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()

    flat_logits = np.argmax(logits, axis=1).flatten()
    # Get tweet ids for prediction output
    ids = label_encoder.inverse_transform(b_tweet_ids.numpy())
    # Store predictions and true labels
    predictions.extend(list(zip(ids, flat_logits)))

output_file = output_dir + "evaluation_results.csv"
# Print the list of prediction
with open(output_file, 'w') as out_file:
    # Get each tweet id
    for tweet_id_prediction in predictions:
        # Print the prediction todo: debug to remove
        print(str(tweet_id_prediction[0]) + "\t" + str(tweet_id_prediction[1]))

        # Append the tweet id along with predicted label on file
        label = 'OFF'
        if str(tweet_id_prediction[1]) == '0':
            label = 'NOT'
        out_file.write(str(tweet_id_prediction[0]) + "," + label + '\n')


print('    DONE.')