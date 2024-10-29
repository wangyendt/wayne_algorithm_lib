# author: wangye(Wayne)
# license: Apache Licence
# file: tts.py
# time: 2024-10-30-01:38:42
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import subprocess
import platform
from gtts import gTTS
import shutil
import tempfile
import os


def text_to_speech_output_opus(text, opus_filename, use_say=True):
    """
    Converts text to speech and saves the output as an Opus file.

    Parameters:
    - text (str): The text to be converted to speech.
    - opus_filename (str): The name of the output .opus file.
    - use_say (bool): If True and running on macOS, uses the `say` command;
                      otherwise, uses gTTS for text-to-speech.

    Notes:
    - Requires ffmpeg for audio conversion.
    - Automatically cleans up temporary files.
    """
    # Check if ffmpeg is installed
    if not shutil.which("ffmpeg"):
        system_platform = platform.system()
        if system_platform == "Darwin":
            print("请安装 ffmpeg，可以通过 Homebrew 安装：`brew install ffmpeg`")
        elif system_platform == "Windows":
            print("请安装 ffmpeg，并将其添加到系统 PATH 中。可以从 https://ffmpeg.org 下载 Windows 版本。")
        elif system_platform == "Linux":
            print("请安装 ffmpeg，可以使用系统包管理器，例如：`sudo apt install ffmpeg`")
        return

    system_platform = platform.system()

    # Create a temporary directory for storing the .aiff file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_aiff = os.path.join(temp_dir, "temp_output.aiff")

        # macOS: Use 'say' if specified, otherwise use gTTS
        if system_platform == "Darwin" and use_say:
            subprocess.run(["say", text, "-o", temp_aiff], check=True)
        else:
            # Use gTTS for Windows, Linux, and optionally macOS
            tts = gTTS(text=text, lang="zh")
            tts.save(temp_aiff)

        # Convert to Opus
        subprocess.run(["ffmpeg", "-i", temp_aiff, "-acodec", "libopus", "-ac", "1", "-ar", "16000", opus_filename, "-y"], check=True)

    # Temporary directory and its contents (temp_aiff) are automatically cleaned up


def text_to_speech_output_mp3(text, mp3_filename, use_say=True):
    """
    Converts text to speech and saves the output as an MP3 file.

    Parameters:
    - text (str): The text to be converted to speech.
    - mp3_filename (str): The name of the output .mp3 file.
    - use_say (bool): If True and running on macOS, uses the `say` command;
                      otherwise, uses gTTS for text-to-speech.

    Notes:
    - Automatically cleans up temporary files.
    """
    # Check if ffmpeg is installed
    if not shutil.which("ffmpeg"):
        system_platform = platform.system()
        if system_platform == "Darwin":
            print("请安装 ffmpeg，可以通过 Homebrew 安装：`brew install ffmpeg`")
        elif system_platform == "Windows":
            print("请安装 ffmpeg，并将其添加到系统 PATH 中。可以从 https://ffmpeg.org 下载 Windows 版本。")
        elif system_platform == "Linux":
            print("请安装 ffmpeg，可以使用系统包管理器，例如：`sudo apt install ffmpeg`")
        return

    system_platform = platform.system()

    # Create a temporary directory for storing the .aiff file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_aiff = os.path.join(temp_dir, "temp_output.aiff")

        # macOS: Use 'say' if specified, otherwise use gTTS
        if system_platform == "Darwin" and use_say:
            subprocess.run(["say", text, "-o", temp_aiff], check=True)
        else:
            # Use gTTS for Windows, Linux, and optionally macOS
            tts = gTTS(text=text, lang="zh")
            tts.save(temp_aiff)

        # Convert to MP3
        subprocess.run(["ffmpeg", "-i", temp_aiff, mp3_filename, "-y"], check=True)

    # Temporary directory and its contents (temp_aiff) are automatically cleaned up


if __name__ == '__main__':
    text_to_speech_output_opus('你好，Chandler', 'test.opus')
