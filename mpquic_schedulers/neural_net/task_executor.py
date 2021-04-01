import argparse
import logging
import os
import time
from datetime import datetime

from mpquic_schedulers.neural_net.train_save import train_save_model


def train_update_model():
    project_home_dir = os.getenv("PROJECT_HOME_DIR")
    
    parser = argparse.ArgumentParser(description='Train & Save Model')
    parser.add_argument('--epochs', type=int, dest="epochs", help="Num of epochs to run", default=3)
    parser.add_argument('--training_file', type=str, dest="training_file", help="Absolute Path to training file",
                        required=True)
    parser.add_argument('--output_dir', type=str, dest="output_dir", help="Absolute Path to output dir", required=True)
    args = parser.parse_args()
    
    logs_dir = os.path.join(project_home_dir, "nn_logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    logging.basicConfig(filename=os.path.join(logs_dir, f'{datetime.now()}.txt'), filemode='w+',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)
    
    time.sleep(30)
    
    while True:
        if not os.path.isdir(os.path.join(project_home_dir, "model_output")) or not os.path.isfile(os.path.join(
                project_home_dir, "online_training", "train.txt")):
            logging.warning("Skipping training for 2 mins, as required data is not available yet!")
            time.sleep(2 * 60)
            continue
        logging.info("Starting to Train & save the model")
        train_save_model(args.training_file, args.epochs, args.output_dir)
        time.sleep(5 * 60)

