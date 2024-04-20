DATA_PATH = "../data/"
DATASET_FILE = "../data/mozilla_firefox.csv"
TEST_DATASET_FILE = "../data/test.csv"
TRAIN_DATASET_FILE = "../data/train.csv"
DATASET_PICKLE_FILE = "../data/mozilla_firefox.pkl"
DATASET_NAME = "mozilla_firefox"


MODEL_NAMES = ["sentence-transformers/multi-qa-mpnet-base-dot-v1", "sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-distilroberta-v1", "sentence-transformers/all-MiniLM-L12-v2", "mixedbread-ai/mxbai-embed-large-v1", "intfloat/multilingual-e5-large-instruct", "avsolatorio/GIST-large-Embedding-v0", "llmrails/ember-v1", "microsoft/codebert-base"]
MODEL_USED = MODEL_NAMES[0]