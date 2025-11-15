import os
import logging
import tempfile
import asyncio
import warnings
import aiohttp
from urllib.parse import urljoin
import whisper
from moviepy.editor import VideoFileClip

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TelegramBotAPI:
    def __init__(self, token: str, base_url: str = "https://api.telegram.org", local_server: bool = False):
        self.token = token
        self.base_url = base_url
        self.local_server = local_server

        if local_server:
            self.api_url = f"{base_url}/bot{token}/"

    async def _make_request(self, method: str, data: dict = None, files: dict = None, timeout: int = 60):
        url = urljoin(self.api_url, method)

        try:
            if files:
                async with aiohttp.ClientSession() as session:
                    form_data = aiohttp.FormData()

                    for key, value in (data or {}).items():
                        form_data.add_field(key, str(value))

                    for key, file_info in files.items():
                        form_data.add_field(
                            key,
                            file_info['content'],
                            filename=file_info.get('filename', 'file'),
                            content_type=file_info.get('content_type', 'application/octet-stream')
                        )

                    async with session.post(url, data=form_data, timeout=timeout) as response:
                        return await response.json()
            else:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data, timeout=timeout) as response:
                        return await response.json()
        except asyncio.TimeoutError:
            logger.error(f"API request timeout for {method}")
            return None
        except Exception as e:
            logger.error(f"API request error for {method}: {str(e)}")
            return None

    async def send_message(self, chat_id: int, text: str, reply_to_message_id: int = None):
        data = {
            "chat_id": chat_id,
            "text": text
        }
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id

        return await self._make_request("sendMessage", data)

    async def edit_message_text(self, chat_id: int, message_id: int, text: str):
        data = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text
        }
        return await self._make_request("editMessageText", data)

    async def delete_message(self, chat_id: int, message_id: int):
        data = {
            "chat_id": chat_id,
            "message_id": message_id
        }
        return await self._make_request("deleteMessage", data)

    async def send_document(self, chat_id: int, document_path: str, caption: str = None, filename: str = None):
        try:
            with open(document_path, 'rb') as f:
                file_content = f.read()

            files = {
                "document": {
                    "content": file_content,
                    "filename": filename or os.path.basename(document_path)
                }
            }

            data = {
                "chat_id": chat_id
            }

            if caption:
                data["caption"] = caption

            return await self._make_request("sendDocument", data, files)
        except Exception as e:
            logger.error(f"Error sending document: {str(e)}")
            return None

    async def get_file(self, file_id: str):
        data = {
            "file_id": file_id
        }
        result = await self._make_request("getFile", data, timeout=120)

        if result:
            logger.info(f"getFile response received")
        else:
            logger.error("getFile returned None")

        return result

    async def download_file(self, file_path: str, destination: str, file_size: int = 0):
        try:
            if self.local_server:
                logger.info(f"Local server file path: {file_path}")

                if os.path.exists(file_path):
                    import shutil
                    shutil.copy2(file_path, destination)
                    actual_size = os.path.getsize(destination)
                    logger.info(f"File copied directly: {actual_size} bytes")
                    return True
                else:
                    logger.warning(f"Local file not found at {file_path}, trying HTTP download")

                    return await self._try_alternative_local_urls(file_path, destination, file_size)
            else:
                file_url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
                logger.info(f"Downloading from: {file_url}")

                async with aiohttp.ClientSession() as session:
                    timeout = aiohttp.ClientTimeout(total=3600)
                    async with session.get(file_url, timeout=timeout) as response:
                        if response.status == 200:
                            total_downloaded = 0
                            with open(destination, 'wb') as f:
                                async for chunk in response.content.iter_chunked(64 * 1024):
                                    if chunk:
                                        f.write(chunk)
                                        total_downloaded += len(chunk)
                                        if file_size > 50 * 1024 * 1024 and total_downloaded % (10 * 1024 * 1024) == 0:
                                            progress = (total_downloaded / file_size) * 100
                                            logger.info(
                                                f"Download progress: {progress:.1f}% ({total_downloaded}/{file_size} bytes)")

                            logger.info(f"Download completed: {total_downloaded} bytes")
                            return True
                        else:
                            logger.error(f"HTTP error {response.status} while downloading file")
                            return False

        except asyncio.TimeoutError:
            logger.error("Download timeout")
            return False
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            return False

    async def _try_alternative_local_urls(self, file_path: str, destination: str, file_size: int = 0):
        # Extract just the filename
        filename = os.path.basename(file_path)

        alternatives = [
            f"{self.base_url}/file/bot{self.token}/{filename}",
            f"{self.base_url}/api/file/bot{self.token}/{filename}",
            f"{self.base_url}/api/bot{self.token}/file/{filename}",
            f"{self.base_url}/bot{self.token}/file/{filename}",
            # Try with the relative path from the file_path
            f"{self.base_url}/file/bot{self.token}/videos/{filename}",
            f"{self.base_url}/api/file/bot{self.token}/videos/{filename}",
        ]

        for file_url in alternatives:
            try:
                logger.info(f"Trying alternative URL: {file_url}")
                async with aiohttp.ClientSession() as session:
                    timeout = aiohttp.ClientTimeout(total=3600)
                    async with session.get(file_url, timeout=timeout) as response:
                        if response.status == 200:
                            total_downloaded = 0
                            with open(destination, 'wb') as f:
                                async for chunk in response.content.iter_chunked(64 * 1024):
                                    if chunk:
                                        f.write(chunk)
                                        total_downloaded += len(chunk)

                            logger.info(f"Download successful via alternative URL: {total_downloaded} bytes")

                            if os.path.exists(destination) and os.path.getsize(destination) > 0:
                                return True
                            else:
                                logger.error("Downloaded file is empty")
                                continue
            except Exception as e:
                logger.error(f"Alternative URL failed: {str(e)}")
                continue

        logger.error("All download methods failed")
        return False


class TranscriptionBot:
    def __init__(self, token: str, model_size: str = "base", local_server_url: str = None):
        self.token = token
        self.model_size = model_size
        self.model = None

        if local_server_url:
            self.api = TelegramBotAPI(token, base_url=local_server_url, local_server=True)
            logger.info(f"Using local Bot API server: {local_server_url}")
        else:
            self.api = TelegramBotAPI(token)
            logger.info("Using official Telegram Bot API")

    def load_model(self):
        try:
            logger.info(f"Loading Whisper model ({self.model_size})...")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            return False

    def convert_to_wav(self, input_path: str, output_wav_path: str) -> bool:
        try:
            warnings.filterwarnings("ignore")
            clip = VideoFileClip(input_path)
            clip.audio.write_audiofile(output_wav_path, verbose=False, logger=None)
            clip.close()
            logger.info(f"Successfully converted to WAV: {output_wav_path}")
            return True
        except Exception as e:
            logger.error(f"Error converting to WAV: {str(e)}")
            return False

    def transcribe_german_audio(self, audio_path: str) -> str:
        try:
            logger.info("Transcribing audio in German...")
            result = self.model.transcribe(audio_path, language="de")
            logger.info("Transcription completed successfully")
            return result["text"]
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            return None

    async def process_update(self, update: dict):
        try:
            if "message" in update:
                message = update["message"]
                chat_id = message["chat"]["id"]
                message_id = message["message_id"]

                if "text" in message and message["text"].startswith("/start"):
                    await self.handle_start(chat_id, message_id)
                    return

                if "video" in message or "audio" in message or "voice" in message:
                    await self.handle_media(message, chat_id, message_id)

        except Exception as e:
            logger.error(f"Error processing update: {str(e)}")

    async def handle_start(self, chat_id: int, message_id: int):
        welcome_text = (
            "Добро пожаловать к твоему персональному боту!\n\n"
            "Я могу распознать текст по видео или аудиофайлу "
            "Отправь мне файл в соответствующем формате "
            "и я переведу его в текст.\n\n"
            "Поддерживаемые форматы:\n"
            "• Видео: MP4, MOV, AVI, MKV.\n"
            "• Аудио: MP3, WAV, M4A, FLAC.\n\n"
        )
        await self.api.send_message(chat_id, welcome_text)

    async def handle_media(self, message: dict, chat_id: int, message_id: int):
        processing_msg = None
        try:
            processing_result = await self.api.send_message(chat_id, "Обрабатываю файл...")
            if processing_result and processing_result.get("ok"):
                processing_msg = processing_result["result"]["message_id"]
            else:
                logger.error("Failed to send processing message")
                return

            with tempfile.TemporaryDirectory() as temp_dir:
                if "video" in message:
                    file_info = message["video"]
                    file_size = file_info.get("file_size", 0)
                    file_type = "video"
                elif "audio" in message:
                    file_info = message["audio"]
                    file_size = file_info.get("file_size", 0)
                    file_type = "audio"
                elif "voice" in message:
                    file_info = message["voice"]
                    file_size = file_info.get("file_size", 0)
                    file_type = "voice"
                else:
                    await self.api.edit_message_text(chat_id, processing_msg, "Неподдерживаемый тип файла")
                    return

                file_id = file_info["file_id"]
                original_file_path = os.path.join(temp_dir, "original_file")

                logger.info(f"Processing {file_type} file, size: {file_size} bytes")

                if not self.api.local_server and file_size > 20 * 1024 * 1024:
                    await self.api.edit_message_text(
                        chat_id,
                        processing_msg,
                        f"Файл слишком большой ({file_size // (1024 * 1024)}MB). "
                        "Для файлов больше 20MB используйте локальный Bot API сервер."
                    )
                    return

                await self.api.edit_message_text(chat_id, processing_msg, "Скачиваю файл...")

                file_result = await self.api.get_file(file_id)
                if not file_result or not file_result.get("ok"):
                    await self.api.edit_message_text(chat_id, processing_msg, "Ошибка получения информации о файле")
                    return

                file_data = file_result["result"]
                file_path = file_data.get("file_path")

                logger.info(f"File data received: {file_data}")
                logger.info(f"File path: {file_path}")
                logger.info(f"File size from API: {file_data.get('file_size')}")

                if not file_path:
                    await self.api.edit_message_text(chat_id, processing_msg, "Не удалось получить путь к файлу")
                    return

                download_success = await self.api.download_file(file_path, original_file_path, file_size)

                if not download_success:
                    await self.api.edit_message_text(chat_id, processing_msg, "Ошибка скачивания файла")
                    return

                if not os.path.exists(original_file_path) or os.path.getsize(original_file_path) == 0:
                    await self.api.edit_message_text(chat_id, processing_msg, "Файл не был загружен или пустой")
                    return

                actual_size = os.path.getsize(original_file_path)
                logger.info(f"File downloaded successfully: {actual_size} bytes")

                if "video" in message:
                    wav_path = os.path.join(temp_dir, "audio.wav")

                    await self.api.edit_message_text(chat_id, processing_msg, "Конвертирую в аудиофайл...")

                    loop = asyncio.get_event_loop()
                    conversion_success = await loop.run_in_executor(
                        None, self.convert_to_wav, original_file_path, wav_path
                    )

                    if not conversion_success:
                        await self.api.edit_message_text(chat_id, processing_msg, "Ошибка при конвертации в аудиофайл")
                        return

                    audio_file_path = wav_path
                else:
                    audio_file_path = original_file_path


                await self.api.edit_message_text(chat_id, processing_msg, "Расшифровываю...")

                loop = asyncio.get_event_loop()
                transcribed_text = await loop.run_in_executor(
                    None, self.transcribe_german_audio, audio_file_path
                )

                if transcribed_text is None:
                    await self.api.edit_message_text(chat_id, processing_msg, "Ошибка в процессе расшифровки")
                    return

                if not transcribed_text.strip():
                    await self.api.edit_message_text(chat_id, processing_msg, "В аудио нет текста")
                    return

                if len(transcribed_text) <= 4096:
                    await self.api.edit_message_text(chat_id, processing_msg, f"Расшифровка:\n\n{transcribed_text}")
                else:
                    await self.api.edit_message_text(chat_id, processing_msg, "Многа букав, отправляю файлом")

                    text_file_path = os.path.join(temp_dir, "transcription.txt")
                    with open(text_file_path, 'w', encoding='utf-8') as f:
                        f.write(transcribed_text)

                    await self.api.send_document(
                        chat_id,
                        text_file_path,
                        caption="Расшифровка:",
                        filename="transcription.txt"
                    )
                    await self.api.delete_message(chat_id, processing_msg)

                logger.info("Transcription completed successfully")

        except Exception as e:
            logger.error(f"Error processing media: {str(e)}")
            if processing_msg:
                try:
                    await self.api.edit_message_text(chat_id, processing_msg, "Какая-то ошибка, напиши Егору")
                except:
                    pass


class BotServer:
    def __init__(self, bot: TranscriptionBot):
        self.bot = bot
        self.running = False

    async def process_updates_async(self):
        logger.info("Starting bot in async polling mode...")
        offset = 0
        self.running = True

        while self.running:
            try:
                if self.bot.api.local_server:
                    url = f"{self.bot.api.base_url}/bot{self.bot.token}/getUpdates"
                else:
                    url = f"https://api.telegram.org/bot{self.bot.token}/getUpdates"

                params = {"offset": offset, "timeout": 30}

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=60) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("ok"):
                                for update in data["result"]:
                                    offset = update["update_id"] + 1
                                    await self.bot.process_update(update)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Polling error: {str(e)}")
                await asyncio.sleep(5)

    def stop(self):
        self.running = False


async def main_async():
    BOT_TOKEN = '<токен>'
    LOCAL_BOT_API_SERVER = 'http://localhost:8081' # none, если сервер не скачан

    bot = TranscriptionBot(
        token=BOT_TOKEN,
        model_size="base",
        local_server_url=LOCAL_BOT_API_SERVER
    )

    if not bot.load_model():
        logger.error("Failed to load Whisper model. Exiting.")
        return

    server = BotServer(bot)
    await server.process_updates_async()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()