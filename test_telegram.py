import requests
import logging

# Configure logging
logger = logging.getLogger("TelegramNotify")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

# Test notification
def send_test_notification():
    url = "https://api.telegram.org"
    response = requests.get(url)
    logger.debug(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        send_message("Hello, Telegram!")
    else:
        logger.error(f"Failed to access Telegram API: {response.status_code}")

# Function to send a message
def send_message(message):
    TELEGRAM_BOT_TOKEN = "8570114716:AAEamaAT1ZEtu6FVurhJyB0EnjgX7X6LGq0"
    TELEGRAM_CHAT_ID = "7819044079"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage?TELEGRAM_CHAT_ID={TELEGRAM_CHAT_ID}&text={message}"
    response = requests.post(url)
    if response.status_code == 200:
        logger.debug(f"Message sent successfully: {message}")
    else:
        logger.error(f"Failed to send message: {response.status_code}")

# Run the test notification
send_test_notification()
