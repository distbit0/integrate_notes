import pysnooper
from loguru import logger
from dotenv import load_dotenv


def main():
    load_dotenv()
    logger.info("Starting the application")
    with pysnooper.snoop():
        # Your main application logic here
        print("Hello from integrate_notes!")


if __name__ == "__main__":
    main()
