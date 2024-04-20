import argparse, os, string, sys
import torch
import sacrebleu
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
import logging
from peft import PrefixTuningConfig, TaskType, get_peft_model, PeftModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class 