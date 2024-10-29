# author: wangye(Wayne)
# license: Apache Licence
# file: lark_custom_bot.py
# time: 2024-10-16-15:39:44
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import requests
import hashlib
import base64
import hmac
import time
import logging
import os
import numpy as np
from typing import List, Dict, Optional, Union
from requests_toolbelt.multipart.encoder import MultipartEncoder


class LarkCustomBot:
    """
    A class to interact with the Lark (Feishu) Bot API.

    This class provides methods to send various types of messages
    (text, rich text, images, etc.) to Lark channels using webhooks.
    It also supports image uploading (including from cv2.Mat objects) and signature verification for enhanced security.
    """

    def __init__(self, webhook: str, secret: str = '', bot_app_id: str = '', bot_secret: str = '') -> None:
        """
        Initialize the LarkCustomBot instance.

        Args:
            webhook (str): The webhook URL for the Lark bot.
            secret (str, optional): The signing secret for request verification. Defaults to ''.
            bot_app_id (str, optional): The bot's app ID for authentication. Defaults to ''.
            bot_secret (str, optional): The bot's app secret for authentication. Defaults to ''.
        """
        self._setup_logging()
        self.webhook = webhook
        self.secret = secret
        self.bot_app_id = bot_app_id
        self.bot_secret = bot_secret
        self.timestamp = str(int(time.time()))

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.DEBUG
        )

    def _get_tenant_access_token(self) -> str:
        """
        Retrieve the tenant access token for API authentication.

        Returns:
            str: The tenant access token.

        Raises:
            requests.RequestException: If the API request fails.
        """
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        headers = {"Content-Type": "application/json; charset=utf-8"}
        payload = {
            "app_id": self.bot_app_id,
            "app_secret": self.bot_secret
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["tenant_access_token"]
        except requests.RequestException as e:
            logging.error(f"Failed to get tenant access token: {e}")
            raise

    def upload_image(self, file_path: str) -> str:
        """
        Upload an image to the Lark server from a file path.

        Args:
            file_path (str): The path to the image file.

        Returns:
            str: The image key of the uploaded image, or an empty string if upload fails.
        """
        if not os.path.getsize(file_path):
            logging.error("Image file is empty")
            return ''

        if not self.bot_app_id or not self.bot_secret:
            logging.error("Bot app ID or secret is missing")
            return ''

        try:
            tenant_access_token = self._get_tenant_access_token()
            url = "https://open.feishu.cn/open-apis/im/v1/images"

            with open(file_path, 'rb') as file:
                form = MultipartEncoder(fields={
                    'image_type': 'message',
                    'image': (os.path.basename(file_path), file, 'image/jpeg')
                })
                headers = {
                    'Authorization': f'Bearer {tenant_access_token}',
                    'Content-Type': form.content_type
                }
                response = requests.post(url, headers=headers, data=form)
                response.raise_for_status()
                return response.json()["data"]["image_key"]
        except (IOError, requests.RequestException) as e:
            logging.error(f"Failed to upload image: {e}")
            return ''

    def upload_image_from_cv2(self, cv2_image: np.ndarray) -> str:
        """
        Upload an image to the Lark server from a cv2.Mat object.

        Args:
            cv2_image (np.ndarray): The cv2.Mat object containing the image.

        Returns:
            str: The image key of the uploaded image, or an empty string if upload fails.
        """
        if cv2_image.size == 0:
            logging.error("Image is empty")
            return ''

        if not self.bot_app_id or not self.bot_secret:
            logging.error("Bot app ID or secret is missing")
            return ''

        try:
            # Import cv2 only when this method is called
            import cv2

            tenant_access_token = self._get_tenant_access_token()
            url = "https://open.feishu.cn/open-apis/im/v1/images"

            # Convert cv2 image to bytes
            _, img_encoded = cv2.imencode('.jpg', cv2_image)
            img_bytes = img_encoded.tobytes()

            form = MultipartEncoder(fields={
                'image_type': 'message',
                'image': ('image.jpg', img_bytes, 'image/jpeg')
            })
            headers = {
                'Authorization': f'Bearer {tenant_access_token}',
                'Content-Type': form.content_type
            }
            response = requests.post(url, headers=headers, data=form)
            response.raise_for_status()
            return response.json()["data"]["image_key"]
        except requests.RequestException as e:
            logging.error(f"Failed to upload image: {e}")
            return ''

    def _generate_signature(self, timestamp: str, secret: str) -> str:
        """
        Generate a signature for request verification.

        Args:
            timestamp (str): The current timestamp.
            secret (str): The signing secret.

        Returns:
            str: The generated signature.
        """
        string_to_sign = f'{timestamp}\n{secret}'
        hmac_code = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
        return base64.b64encode(hmac_code).decode('utf-8')

    def _send_request(self, data: Dict) -> None:
        """
        Send a POST request to the Lark webhook.

        Args:
            data (Dict): The data to be sent in the request body.

        Raises:
            requests.RequestException: If the request fails.
        """
        headers = {'Content-Type': 'application/json; charset=utf-8'}
        if self.secret:
            data['timestamp'] = self.timestamp
            data['sign'] = self._generate_signature(self.timestamp, self.secret)

        logging.info(f"Sending data: {data}")

        try:
            response = requests.post(self.webhook, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

            if 'code' in result:
                logging.error(f"Message sending failed: {result}")
            elif result.get('StatusCode') == 0:
                logging.debug('Message sent successfully')
        except requests.RequestException as e:
            logging.error(f"Failed to send message: {e}")
            raise

    def send_text(self, text: str, mention_all: bool = False) -> None:
        """
        Send a text message.

        Args:
            text (str): The text content of the message.
            mention_all (bool, optional): Whether to mention all users in the channel. Defaults to False.
        """
        if mention_all:
            text += ' <at user_id="all">所有人</at>'

        data = {
            "msg_type": "text",
            "content": {"text": text}
        }
        self._send_request(data)

    def send_post(self, content: List[List[Dict]], title: Optional[str] = None) -> None:
        """
        Send a rich text message.

        Args:
            content (List[List[Dict]]): The content of the rich text message, organized in lines and elements.
            title (Optional[str], optional): The title of the message. Defaults to None.
        """
        data = {
            "msg_type": "post",
            "content": {
                "post": {
                    "zh_cn": {
                        "title": title,
                        "content": content
                    }
                }
            }
        }
        self._send_request(data)

    def send_share_chat(self, share_chat_id: str) -> None:
        """
        Share a chat.

        Args:
            share_chat_id (str): The ID of the chat to be shared.
        """
        data = {
            "msg_type": "share_chat",
            "content": {
                "share_chat_id": share_chat_id
            }
        }
        self._send_request(data)

    def send_image(self, image_key: str) -> None:
        """
        Send an image message.

        Args:
            image_key (str): The key of the image to be sent.
        """
        data = {
            "msg_type": "image",
            "content": {"image_key": image_key}
        }
        self._send_request(data)

    def send_interactive(self, card: Dict) -> None:
        """
        Send an interactive message card.

        Args:
            card (Dict): The content of the interactive message card.
        """
        data = {
            "msg_type": "interactive",
            "card": card
        }
        self._send_request(data)


def create_text_content(text: str, unescape: bool = False) -> Dict:
    """
    Create a text content element for rich text messages.

    Args:
        text (str): The text content.
        unescape (bool, optional): Whether to unescape the text. Defaults to False.

    Returns:
        Dict: A dictionary representing the text content element.
    """
    return {
        "tag": "text",
        "text": text,
        "un_escape": unescape
    }


def create_link_content(href: str, text: str) -> Dict:
    """
    Create a link content element for rich text messages.

    Args:
        href (str): The URL of the link.
        text (str): The display text for the link.

    Returns:
        Dict: A dictionary representing the link content element.
    """
    return {
        "tag": "a",
        "href": href,
        "text": text
    }


def create_at_content(user_id: str, user_name: str) -> Dict:
    """
    Create an @mention content element for rich text messages.

    Args:
        user_id (str): The ID of the user to be mentioned.
        user_name (str): The name of the user to be mentioned.

    Returns:
        Dict: A dictionary representing the @mention content element.
    """
    return {
        "tag": "at",
        "user_id": user_id,
        "user_name": user_name
    }


def create_image_content(image_key: str, width: Optional[int] = None, height: Optional[int] = None) -> Dict:
    """
    Create an image content element for rich text messages.

    Args:
        image_key (str): The key of the image.
        width (Optional[int], optional): The width of the image. Defaults to None.
        height (Optional[int], optional): The height of the image. Defaults to None.

    Returns:
        Dict: A dictionary representing the image content element.
    """
    content = {
        "tag": "img",
        "image_key": image_key
    }
    if width:
        content["width"] = width
    if height:
        content["height"] = height
    return content


if __name__ == '__main__':
    app_id = ''
    app_secret = ''
    hook = 'https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxx'
    bot = LarkCustomBot(hook, bot_app_id=app_id, bot_secret=app_secret)
    bot.send_text("hello world")
    # image_key = bot.upload_image('./test.png')
    # image_key = bot.upload_image_from_cv2(cv2.imread('./test.png'))
    bot.send_image(image_key='img_v3_02fn_beb6b288-ac80-4279-a6b6-e52145f7c70g')
    bot.send_post([
        [create_text_content('www.baidu.com\n'), create_link_content(href='baidu.com', text='百度')],
        [create_image_content('img_v3_02fn_beb6b288-ac80-4279-a6b6-e52145f7c70g')]
    ])
