from telegram import Bot
from telegram.error import TelegramError

# Replace with your bot token and chat ID
bot_token = '8570114716:AAEamaAT1ZEtu6FVurhJyB0EnjgX7X6LGq0'
chat_id = '7819044079'

try:
    # Create a bot instance
    bot = Bot(token=bot_token)

    # Add the bot to the chat
    bot.send_message(chat_id=chat_id, text="Hello, I am the bot!")
    print("Bot added successfully.")
except TelegramError as e:
    print(f"Failed to add bot: {e}")