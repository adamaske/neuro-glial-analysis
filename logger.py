import logging 
import datetime

# Logging
def setup_logger():
    log_filepath = "logs/log_" + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".txt"
    logging.basicConfig(
        level=logging.DEBUG,  # Set the lowest level to capture all messages
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),  # Log to a text file
            logging.StreamHandler()  # Log to the console
        ]
    )
    file_handler = logging.getLogger().handlers[0]
    file_handler.level = logging.DEBUG

    stream_handler = logging.getLogger().handlers[1] # Dont print debug info to console
    stream_handler.level = logging.INFO

setup_logger()

logging.info("testing info")

logging.debug("warning debug")