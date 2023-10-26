import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" # defines the nomenclature of the log file name
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE) #defines the path where the log will be created, here it is current working directory(cwd) i.e. 'src'
os.makedirs(logs_path,exist_ok=True) #defines that the files will get appended if the same file is present

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

#You can test the code to check whether logging is done or not using:

#if __name__ == "__main__":
#   logging.info("Log testing, it is working fine")