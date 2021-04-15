import argparse
import logging
import os
from datetime import datetime

from model import train_save

logging.basicConfig(filename=os.path.join("logs", f'{datetime.now()}.txt'), filemode='w+',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S', level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='Train Model using given dataset')
parser.add_argument('--training_file_path', type=str, dest="training_file_path", help="Training file path(.csv)",
                    required=True)
parser.add_argument('--output_dir', type=str, dest="output_dir", help="Model is saved into this directory",
                    default="output")
parser.add_argument('--epochs', type=int, dest="epochs", help="Number of epochs to train for",
                    default=3)
args = parser.parse_args()

logging.info("Starting to Train & save the model")
train_save.train_save_model(args.training_file_path, args.output_dir, args.epochs)
logging.info("Model successfully saved")
