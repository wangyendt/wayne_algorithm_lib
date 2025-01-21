# author: wangye(Wayne)
# license: Apache Licence
# file: lark_bot.py
# time: 2024-10-30-01:08:13
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import json
import platform
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import lark_oapi as lark
from lark_oapi.api.contact.v3 import *
from lark_oapi.api.im.v1 import *


class TextContent:
    """
    Helper class for creating various text patterns used in Feishu messages.
    Provides static methods for creating mentions, text formatting, and links.
    """

    @staticmethod
    def make_at_all_pattern() -> str:
        """
        Create an @all mention pattern.

        Returns:
            str: The @all mention string.
        """
        return "<at user_id=\"all\"></at>"

    @staticmethod
    def make_at_someone_pattern(someone_open_id: str, username: str, id_type: str) -> str:
        """
        Create a pattern to mention a specific user.

        Args:
            someone_open_id (str): User's identifier.
            username (str): Display name for the mention.
            id_type (str): Type of ID ('open_id', 'union_id', or 'user_id').

        Returns:
            str: Formatted mention string.
        """
        id_type_mapping = {
            'open_id': 'user_id',  # In @mentions, open_id is specified as user_id
            'union_id': 'union_id',
            'user_id': 'user_id'
        }

        mention_type = id_type_mapping.get(id_type, 'user_id')
        return f"<at {mention_type}=\"{someone_open_id}\">{username}</at>"

    @staticmethod
    def make_bold_pattern(content: str) -> str:
        """
        Make text bold using HTML-like tags.

        Args:
            content (str): The text content to be bolded.

        Returns:
            str: Bolded text.
        """
        return f"<b>{content}</b>"

    @staticmethod
    def make_italian_pattern(content: str) -> str:
        """
        Make text italic using HTML-like tags.

        Args:
            content (str): The text content to be italicized.

        Returns:
            str: Italicized text.
        """
        return f"<i>{content}</i>"

    @staticmethod
    def make_underline_pattern(content: str) -> str:
        """
        Make text underlined using HTML-like tags.

        Args:
            content (str): The text content to be underlined.

        Returns:
            str: Underlined text.
        """
        return f"<u>{content}</u>"

    @staticmethod
    def make_delete_line_pattern(content: str) -> str:
        """
        Make text strikethrough using HTML-like tags.

        Args:
            content (str): The text content to be struck through.

        Returns:
            str: Strikethrough text.
        """
        return f"<s>{content}</s>"

    @staticmethod
    def make_url_pattern(url: str, text: str) -> str:
        """
        Create a markdown-style link.

        Args:
            url (str): The URL to link to.
            text (str): Display text for the link.

        Returns:
            str: Formatted markdown link.
        """
        return f'[{text}]({url})'


class PostContent:
    """
    Helper class for creating rich post content in Feishu messages.
    Supports various content types including text, links, mentions, images, and more.
    """

    def __init__(self, title: str = ''):
        """
        Initialize a new post content with an optional title.

        Args:
            title (str, optional): Post title. Defaults to ''.
        """
        self.content = {
            "zh_cn": {
                "title": title,
                "content": []
            },
        }

    def get_content(self) -> Dict:
        """
        Return the complete post content structure.

        Returns:
            Dict: The post content.
        """
        return self.content

    def set_title(self, title: str) -> None:
        """
        Set or update the post title.

        Args:
            title (str): The new title for the post.
        """
        self.content["zh_cn"]["title"] = title

    @staticmethod
    def list_text_styles() -> List[str]:
        """
        Return available text styles.

        Returns:
            List[str]: List of available text styles.
        """
        return ['bold', 'underline', 'lineThrough', 'italic']

    def make_text_content(self, text: str, styles: Optional[List[str]] = None, unescape: bool = False) -> Dict:
        """
        Create text content with optional styles.

        Args:
            text (str): The text content.
            styles (Optional[List[str]], optional): List of styles to apply. Defaults to None.
            unescape (bool, optional): Whether to unescape the text. Defaults to False.

        Returns:
            Dict: Dict containing the formatted text content.
        """
        return {
            "tag": "text",
            "text": text,
            "style": styles or [],
            "unescape": unescape
        }

    def make_link_content(self, text: str, link: str, styles: Optional[List[str]] = None) -> Dict:
        """
        Create a hyperlink content element.

        Args:
            text (str): Display text for the link.
            link (str): URL for the link.
            styles (Optional[List[str]], optional): List of text styles to apply. Defaults to None.

        Returns:
            Dict: Dict containing the formatted link content.
        """
        return {
            "tag": "a",
            "text": text,
            "href": link,
            "style": styles or []
        }

    def make_at_content(self, at_user_id: str, styles: Optional[List[str]] = None) -> Dict:
        """
        Create a mention content element.

        Args:
            at_user_id (str): The user ID to mention.
            styles (Optional[List[str]], optional): List of styles to apply. Defaults to None.

        Returns:
            Dict: Dict containing the mention content.
        """
        return {
            "tag": "at",
            "user_id": at_user_id,
            "style": styles or []
        }

    def make_image_content(self, image_key: str) -> Dict:
        """
        Create an image content element.

        Args:
            image_key (str): The key of the image to include.

        Returns:
            Dict: Dict containing the image content.
        """
        return {
            "tag": "img",
            "image_key": image_key
        }

    def make_media_content(self, file_key: str, image_key: str = '') -> Dict:
        """
        Create a media content element with an optional thumbnail.

        Args:
            file_key (str): The key of the media file.
            image_key (str, optional): The key of the thumbnail image. Defaults to ''.

        Returns:
            Dict: Dict containing the media content.
        """
        return {
            "tag": "media",
            "image_key": image_key,
            "file_key": file_key
        }

    def make_emoji_content(self, emoji_type: str) -> Dict:
        """
        Create an emoji content element.

        Args:
            emoji_type (str): The type of emoji.

        Returns:
            Dict: Dict containing the emoji content.
        """
        return {
            "tag": "emotion",
            "emoji_type": emoji_type
        }

    def make_hr_content(self) -> Dict:
        """
        Create a horizontal rule content element.

        Returns:
            Dict: Dict containing the horizontal rule.
        """
        return {
            "tag": "hr"
        }

    def make_code_block_content(self, language: str, text: str) -> Dict:
        """
        Create a code block content element.

        Args:
            language (str): The programming language of the code block.
            text (str): The code text.

        Returns:
            Dict: Dict containing the code block content.
        """
        return {
            "tag": "code_block",
            "language": language,
            "text": text
        }

    def make_markdown_content(self, md_text: str) -> Dict:
        """
        Create a Markdown content element.

        Args:
            md_text (str): The Markdown-formatted text.

        Returns:
            Dict: Dict containing the Markdown content.
        """
        return {
            "tag": "md",
            "text": md_text
        }

    def add_content_in_line(self, content: Dict) -> None:
        """
        Add content to the current line.

        If there are no existing lines, it initializes the first line.

        Args:
            content (Dict): The content to add.
        """
        if not self.content["zh_cn"]["content"]:
            self.content["zh_cn"]["content"].append([])
        self.content["zh_cn"]["content"][-1].append(content)

    def add_contents_in_line(self, contents: List[Dict]) -> None:
        """
        Add multiple content items to the current line.

        If there are no existing lines, it initializes the first line.

        Args:
            contents (List[Dict]): The list of contents to add.
        """
        if not self.content["zh_cn"]["content"]:
            self.content["zh_cn"]["content"].append([])
        self.content["zh_cn"]["content"][-1].extend(contents)

    def add_content_in_new_line(self, content: Dict) -> None:
        """
        Add content in a new line.

        Args:
            content (Dict): The content to add in the new line.
        """
        self.content["zh_cn"]["content"].append([content])

    def add_contents_in_new_line(self, contents: List[Dict]) -> None:
        """
        Add multiple content items in a new line.

        Args:
            contents (List[Dict]): The list of contents to add in the new line.
        """
        self.content["zh_cn"]["content"].append(contents)

    def list_emoji_types(self) -> None:
        """
        Open Feishu emoji documentation in the default browser.

        This method detects the operating system and attempts to open the documentation URL.
        If the operating system is unsupported or an error occurs, it logs an error message.
        """
        url = "https://open.feishu.cn/document/server-docs/im-v1/message-reaction/emojis-introduce"
        try:
            system_name = platform.system()
            if system_name == "Windows":
                subprocess.run(["start", url], shell=True)
            elif system_name == "Darwin":  # macOS
                subprocess.run(["open", url])
            elif system_name == "Linux":
                subprocess.run(["xdg-open", url])
            else:
                print(f"Unsupported operating system: {system_name}")
        except Exception as e:
            print(f"Failed to open webpage: {e}")


class LarkBot:
    """
    Main class for interacting with Feishu API.
    Provides methods for sending messages, managing files, and interacting with groups.
    """

    def __init__(self, app_id: str, app_secret: str):
        """
        Initialize the Feishu bot with app credentials.

        Args:
            app_id (str): Feishu application ID.
            app_secret (str): Feishu application secret.
        """
        self.client = lark.Client.builder() \
            .app_id(app_id) \
            .app_secret(app_secret) \
            .log_level(lark.LogLevel.DEBUG) \
            .build()

    def get_user_info(self, emails: List[str], mobiles: List[str]) -> Optional[Dict]:
        """
        Get user information by email and mobile numbers.

        Args:
            emails (List[str]): List of email addresses.
            mobiles (List[str]): List of mobile numbers.

        Returns:
            Optional[Dict]: Dict containing user information or None if request fails.
        """
        request = BatchGetIdUserRequest.builder() \
            .user_id_type("open_id") \
            .request_body(BatchGetIdUserRequestBody.builder()
                          .emails(emails)
                          .mobiles(mobiles)
                          .include_resigned(True)
                          .build()) \
            .build()

        response = self.client.contact.v3.user.batch_get_id(request)

        if not response.success():
            lark.logger.error(
                f"Failed to get user info: {response.code}, {response.msg}, "
                f"log_id: {response.get_log_id()}"
            )
            return None

        return json.loads(lark.JSON.marshal(response.data, indent=4))["user_list"]

    def get_group_list(self) -> List[Dict]:
        """
        Get the list of chat groups the bot is a member of.

        Returns:
            List[Dict]: List of chat group information.
        """
        request = ListChatRequest.builder().build()
        response = self.client.im.v1.chat.list(request)

        if not response.success():
            lark.logger.error(
                f"Failed to get group list: {response.code}, {response.msg}, "
                f"log_id: {response.get_log_id()}"
            )
            return []

        response_data = json.loads(lark.JSON.marshal(response.data, indent=4))
        return response_data.get('items', [])

    @lru_cache
    def get_group_chat_id_by_name(self, group_name: str) -> List[str]:
        """
        Get chat IDs for groups matching the given name.

        Args:
            group_name (str): The name of the group to search for.

        Returns:
            List[str]: List of chat IDs matching the group name.
        """
        return [group['chat_id'] for group in self.get_group_list()
                if group.get('name') == group_name]

    def get_members_in_group_by_group_chat_id(self, group_chat_id: str) -> List[Dict]:
        """
        Get the list of members in a specific group chat.

        Args:
            group_chat_id (str): The chat ID of the group.

        Returns:
            List[Dict]: List of member information in the group.
        """
        request = GetChatMembersRequest.builder().chat_id(group_chat_id).build()
        response = self.client.im.v1.chat_members.get(request)

        if not response.success():
            lark.logger.error(
                f"Failed to get chat members: {response.code}, {response.msg}, "
                f"log_id: {response.get_log_id()}"
            )
            return []

        response_data = json.loads(lark.JSON.marshal(response.data, indent=4))
        return response_data.get('items', [])

    @lru_cache
    def get_member_open_id_by_name(self, group_chat_id: str, member_name: str) -> List[str]:
        """
        Get open IDs for members matching the given name in a group chat.

        Args:
            group_chat_id (str): The chat ID of the group.
            member_name (str): The name of the member to search for.

        Returns:
            List[str]: List of open IDs matching the member name.
        """
        return [member['member_id']
                for member in self.get_members_in_group_by_group_chat_id(group_chat_id)
                if member.get('name') == member_name]

    def _send_message(self,
                      receive_id_type: str,
                      receive_id: str,
                      msg_type: str,
                      content: str) -> Dict:
        """
        Internal method to send messages through Feishu API.

        Args:
            receive_id_type (str): Type of receiver ID ('open_id' or 'chat_id').
            receive_id (str): ID of the message receiver.
            msg_type (str): Type of message to send.
            content (str): JSON-encoded message content.

        Returns:
            Dict: Dict containing API response data.
        """
        request = CreateMessageRequest.builder() \
            .receive_id_type(receive_id_type) \
            .request_body(CreateMessageRequestBody.builder()
                          .receive_id(receive_id)
                          .msg_type(msg_type)
                          .content(content)
                          .build()) \
            .build()

        response = self.client.im.v1.message.create(request)

        if not response.success():
            lark.logger.error(
                f"Failed to send message: {response.code}, {response.msg}, "
                f"log_id: {response.get_log_id()}"
            )
            return {}

        response_data = json.loads(lark.JSON.marshal(response.data, indent=4))
        lark.logger.info(response_data)
        return response_data

    def send_text_to_user(self, user_open_id: str, text: str = '') -> Dict:
        """
        Send a text message to a specific user.

        Args:
            user_open_id (str): The open ID of the user.
            text (str, optional): The text message to send. Defaults to ''.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'open_id',
            user_open_id,
            'text',
            json.dumps({"text": text}, ensure_ascii=False)
        )

    def send_text_to_chat(self, chat_id: str, text: str = '') -> Dict:
        """
        Send a text message to a specific chat group.

        Args:
            chat_id (str): The chat ID of the group.
            text (str, optional): The text message to send. Defaults to ''.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'chat_id',
            chat_id,
            'text',
            json.dumps({"text": text}, ensure_ascii=False)
        )

    def send_image_to_user(self, user_open_id: str, image_key: str) -> Dict:
        """
        Send an image to a specific user.

        Args:
            user_open_id (str): The open ID of the user.
            image_key (str): The key of the image to send.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'open_id',
            user_open_id,
            'image',
            json.dumps({"image_key": image_key}, ensure_ascii=False)
        )

    def send_image_to_chat(self, chat_id: str, image_key: str) -> Dict:
        """
        Send an image to a specific chat group.

        Args:
            chat_id (str): The chat ID of the group.
            image_key (str): The key of the image to send.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'chat_id',
            chat_id,
            'image',
            json.dumps({"image_key": image_key}, ensure_ascii=False)
        )

    def send_interactive_to_user(self, user_open_id: str, interactive: Dict) -> Dict:
        """
        Send an interactive message to a specific user.

        Args:
            user_open_id (str): The open ID of the user.
            interactive (Dict): The interactive message content.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'open_id',
            user_open_id,
            'interactive',
            json.dumps(interactive, ensure_ascii=False)
        )

    def send_interactive_to_chat(self, chat_id: str, interactive: Dict) -> Dict:
        """
        Send an interactive message to a specific chat group.

        Args:
            chat_id (str): The chat ID of the group.
            interactive (Dict): The interactive message content.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'chat_id',
            chat_id,
            'interactive',
            json.dumps(interactive, ensure_ascii=False)
        )

    def send_shared_chat_to_user(self, user_open_id: str, shared_chat_id: str) -> Dict:
        """
        Share a chat to a specific user.

        Args:
            user_open_id (str): The open ID of the user.
            shared_chat_id (str): The chat ID to share.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'open_id',
            user_open_id,
            'share_chat',
            json.dumps({"chat_id": shared_chat_id}, ensure_ascii=False)
        )

    def send_shared_chat_to_chat(self, chat_id: str, shared_chat_id: str) -> Dict:
        """
        Share a chat to a specific chat group.

        Args:
            chat_id (str): The chat ID of the target group.
            shared_chat_id (str): The chat ID to share.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'chat_id',
            chat_id,
            'share_chat',
            json.dumps({"chat_id": shared_chat_id}, ensure_ascii=False)
        )

    def send_shared_user_to_user(self, user_open_id: str, shared_user_id: str) -> Dict:
        """
        Share a user to a specific user.

        Args:
            user_open_id (str): The open ID of the target user.
            shared_user_id (str): The user ID to share.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'open_id',
            user_open_id,
            'share_user',
            json.dumps({"user_id": shared_user_id}, ensure_ascii=False)
        )

    def send_shared_user_to_chat(self, chat_id: str, shared_user_id: str) -> Dict:
        """
        Share a user to a specific chat group.

        Args:
            chat_id (str): The chat ID of the target group.
            shared_user_id (str): The user ID to share.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'chat_id',
            chat_id,
            'share_user',
            json.dumps({"user_id": shared_user_id}, ensure_ascii=False)
        )

    def send_audio_to_user(self, user_open_id: str, file_key: str) -> Dict:
        """
        Send an audio message to a specific user.

        Args:
            user_open_id (str): The open ID of the user.
            file_key (str): The file key of the audio to send.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'open_id',
            user_open_id,
            'audio',
            json.dumps({"file_key": file_key}, ensure_ascii=False)
        )

    def send_audio_to_chat(self, chat_id: str, file_key: str) -> Dict:
        """
        Send an audio message to a specific chat group.

        Args:
            chat_id (str): The chat ID of the group.
            file_key (str): The file key of the audio to send.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'chat_id',
            chat_id,
            'audio',
            json.dumps({"file_key": file_key}, ensure_ascii=False)
        )

    def send_media_to_user(self, user_open_id: str, file_key: str) -> Dict:
        """
        Send a media message to a specific user.

        Args:
            user_open_id (str): The open ID of the user.
            file_key (str): The file key of the media to send.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'open_id',
            user_open_id,
            'media',
            json.dumps({"file_key": file_key}, ensure_ascii=False)
        )

    def send_media_to_chat(self, chat_id: str, file_key: str) -> Dict:
        """
        Send a media message to a specific chat group.

        Args:
            chat_id (str): The chat ID of the group.
            file_key (str): The file key of the media to send.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'chat_id',
            chat_id,
            'media',
            json.dumps({"file_key": file_key}, ensure_ascii=False)
        )

    def send_file_to_user(self, user_open_id: str, file_key: str) -> Dict:
        """
        Send a file to a specific user.

        Args:
            user_open_id (str): The open ID of the user.
            file_key (str): The file key of the file to send.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'open_id',
            user_open_id,
            'file',
            json.dumps({"file_key": file_key}, ensure_ascii=False)
        )

    def send_file_to_chat(self, chat_id: str, file_key: str) -> Dict:
        """
        Send a file to a specific chat group.

        Args:
            chat_id (str): The chat ID of the group.
            file_key (str): The file key of the file to send.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'chat_id',
            chat_id,
            'file',
            json.dumps({"file_key": file_key}, ensure_ascii=False)
        )

    def send_system_msg_to_user(self, user_open_id: str, system_msg_text: str) -> Dict:
        """
        Send a system message to a specific user.

        Args:
            user_open_id (str): The open ID of the user.
            system_msg_text (str): The system message text.

        Returns:
            Dict: Dict containing API response data.
        """
        system_message = {
            "type": "divider",
            "params": {
                "divider_text": {
                    "text": system_msg_text,
                    "i18n_text": {
                        "zh_CN": system_msg_text,
                    }
                }

            },
            "options": {
                "need_rollup": True
            }
        }
        return self._send_message(
            'open_id',
            user_open_id,
            'system',
            json.dumps(system_message, ensure_ascii=False)
        )

    def send_post_to_user(self, user_open_id: str, post_content: Dict[str, str]) -> Dict:
        """
        Send a rich post message to a specific user.

        Args:
            user_open_id (str): The open ID of the user.
            post_content (Dict[str, str]): The post content.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'open_id',
            user_open_id,
            'post',
            json.dumps(post_content, ensure_ascii=False)
        )

    def send_post_to_chat(self, chat_id: str, post_content: Dict[str, dict]) -> Dict:
        """
        Send a rich post message to a specific chat group.

        Args:
            chat_id (str): The chat ID of the group.
            post_content (Dict[str, dict]): The post content.

        Returns:
            Dict: Dict containing API response data.
        """
        return self._send_message(
            'chat_id',
            chat_id,
            'post',
            json.dumps(post_content, ensure_ascii=False)
        )

    def upload_image(self, image_path: str) -> str:
        """
        Upload an image to Feishu.

        Args:
            image_path (str): The local path to the image file.

        Returns:
            str: The key of the uploaded image, or an empty string if upload fails.
        """
        try:
            with open(image_path, "rb") as file:
                request = CreateImageRequest.builder() \
                    .request_body(CreateImageRequestBody.builder()
                                  .image_type("message")
                                  .image(file)
                                  .build()) \
                    .build()

                # Send request
                response: CreateImageResponse = self.client.im.v1.image.create(request)

                # Handle failure
                if not response.success():
                    lark.logger.error(
                        f"client.im.v1.image.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
                    return ''

                # Handle success
                response_data = json.loads(lark.JSON.marshal(response.data, indent=4))
                lark.logger.info(response_data)
                return response_data.get('image_key', '')
        except Exception as e:
            lark.logger.error(f"Exception occurred while uploading image: {e}")
            return ''

    def download_image(self, image_key: str, image_save_path: str) -> None:
        """
        Download an image from Feishu.

        Args:
            image_key (str): The key of the image to download.
            image_save_path (str): The local path to save the downloaded image.
        """
        try:
            request = GetImageRequest.builder() \
                .image_key(image_key) \
                .build()

            # Send request
            response: GetImageResponse = self.client.im.v1.image.get(request)

            # Handle failure
            if not response.success():
                lark.logger.error(
                    f"client.im.v1.image.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
                return

            # 确保目标文件夹存在
            save_path = Path(image_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the image
            save_path.write_bytes(response.file.read())
            
        except Exception as e:
            lark.logger.error(f"Exception occurred while downloading image: {e}")

    def upload_file(self, file_path: str, file_type: str = 'stream') -> str:
        """
        Upload a file to Feishu.

        Args:
            file_path (str): The local path to the file.
            file_type (str, optional): The type of the file. Defaults to 'stream'.
                                       Possible values: 'stream', 'opus', 'mp4', 'pdf', 'doc', 'xls', 'ppt'.

        Returns:
            str: The key of the uploaded file, or an empty string if upload fails.
        """
        try:
            with open(file_path, "rb") as file:
                request = CreateFileRequest.builder() \
                    .request_body(CreateFileRequestBody.builder()
                                  .file_type(file_type)
                                  .file_name(Path(file_path).name)
                                  .file(file)
                                  .build()) \
                    .build()

                # Send request
                response: CreateFileResponse = self.client.im.v1.file.create(request)

                # Handle failure
                if not response.success():
                    lark.logger.error(
                        f"client.im.v1.file.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
                    return ''

                # Handle success
                response_data = json.loads(lark.JSON.marshal(response.data, indent=4))
                lark.logger.info(response_data)
                return response_data.get('file_key', '')
        except Exception as e:
            lark.logger.error(f"Exception occurred while uploading file: {e}")
            return ''

    def download_file(self, file_key: str, file_save_path: str) -> None:
        """
        Download a file from Feishu.

        Args:
            file_key (str): The key of the file to download.
            file_save_path (str): The local path to save the downloaded file.
        """
        try:
            request = GetFileRequest.builder() \
                .file_key(file_key) \
                .build()

            # Send request
            response: GetFileResponse = self.client.im.v1.file.get(request)

            # Handle failure
            if not response.success():
                lark.logger.error(
                    f"client.im.v1.file.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
                return

            # Determine the save path
            save_path = Path(file_save_path)
            if save_path.is_dir():
                save_path = save_path / Path(response.file_name)

            # 确保目标文件夹存在
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the file
            save_path.write_bytes(response.file.read())
            lark.logger.info(f"File downloaded successfully and saved to {save_path}")
            
        except Exception as e:
            lark.logger.error(f"Exception occurred while downloading file: {e}")

    def download_message_resource(self, message_id: str, resource_type: str, save_path: str, file_key: str = None) -> bool:
        """
        下载消息中的资源文件（图片、音频、视频、文件等）
        :param message_id: 消息ID
        :param resource_type: 资源类型，可选值：image、file、media、audio、video
        :param save_path: 保存路径
        :param file_key: 资源的key（image_key或file_key）
        :return: 是否下载成功
        """
        try:
            # 构造请求对象
            request = GetMessageResourceRequest.builder() \
                .message_id(message_id) \
                .type(resource_type)

            # 添加file_key（如果提供）
            if file_key:
                request = request.file_key(file_key)

            request = request.build()

            # 发起请求
            response: GetMessageResourceResponse = self.client.im.v1.message_resource.get(request)

            # 处理失败返回
            if not response.success():
                lark.logger.error(
                    f"下载消息资源失败, code: {response.code}, msg: {response.msg}, "
                    f"log_id: {response.get_log_id()}, resp: \n"
                    f"{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}"
                )
                return False

            # 确保目标文件夹存在
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存文件
            save_path.write_bytes(response.file.read())

            lark.logger.info(f"消息资源已下载到: {save_path}")
            return True

        except Exception as e:
            lark.logger.error(f"下载消息资源时发生错误: {e}")
            return False

    def download_message_resources(self, message_id: str, message_content: str, save_dir: str) -> Dict[str, str]:
        """
        下载消息中的所有资源文件
        :param message_id: 消息ID
        :param message_content: 消息内容（JSON字符串）
        :param save_dir: 保存目录
        :return: 资源类型到保存路径的映射
        """
        try:
            # 解析消息内容
            content = json.loads(message_content)
            result = {}

            # 确保保存目录存在
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # 检查并下载图片
            if "image_key" in content:
                image_path = save_dir / f"image_{content['image_key']}.png"
                if self.download_message_resource(message_id, "image", str(image_path), content['image_key']):
                    result["image"] = str(image_path)

            # 检查并下载文件
            if "file_key" in content:
                file_path = save_dir / f"file_{content['file_key']}"
                if self.download_message_resource(message_id, "file", str(file_path), content['file_key']):
                    result["file"] = str(file_path)

            # 检查并下载音频
            if "file_key" in content and content.get("type") == "audio":
                audio_path = save_dir / f"audio_{content['file_key']}.mp3"
                if self.download_message_resource(message_id, "audio", str(audio_path), content['file_key']):
                    result["audio"] = str(audio_path)

            # 检查并下载视频
            if "file_key" in content and content.get("type") == "video":
                video_path = save_dir / f"video_{content['file_key']}.mp4"
                if self.download_message_resource(message_id, "video", str(video_path), content['file_key']):
                    result["video"] = str(video_path)

            # 检查并下载媒体文件
            if "file_key" in content and content.get("type") == "media":
                media_path = save_dir / f"media_{content['file_key']}"
                if self.download_message_resource(message_id, "media", str(media_path), content['file_key']):
                    result["media"] = str(media_path)

            return result

        except Exception as e:
            lark.logger.error(f"下载消息资源时发生错误: {e}")
            return {}

    def get_chat_and_user_name(self, chat_id: str, user_id: str) -> Tuple[str, str]:
        """
        获取群聊名称和用户名称
        
        Args:
            chat_id: 会话ID
            user_id: 用户ID
            
        Returns:
            Tuple[str, str]: (群聊名称, 用户名称)，如果是私聊则群聊名称为空字符串
        """
        # 获取群聊信息
        chat_name = ""
        request = lark.api.im.v1.GetChatRequest.builder().chat_id(chat_id).build()
        response = self.client.im.v1.chat.get(request)
        if response.success():
            chat_name = response.data.name
        else:
            print(f"获取群聊信息失败: {response.code}, {response.msg}")
            
        # 获取用户信息
        user_name = ""
        request = lark.api.contact.v3.GetUserRequest.builder().user_id(user_id).build()
        response = self.client.contact.v3.user.get(request)
        if response.success():
            user_name = response.data.user.name
        else:
            print(f"获取用户信息失败: {response.code}, {response.msg}")
            
        return chat_name, user_name


if __name__ == '__main__':
    # 创建机器人实例
    bot = LarkBot(
        app_id="cli_a785d99779791013",
        app_secret="bt1JJe4iOy3L7ifsSZsOddDm5xV4xjAT"
    )

    try:
        # 1. 获取群组列表
        group_list = bot.get_group_list()
        print("群组列表:")
        print(json.dumps(group_list, indent=2, ensure_ascii=False))

        # 2. 获取特定群组的ID
        group_chat_ids = bot.get_group_chat_id_by_name("测试3")
        if not group_chat_ids:
            print("未找到群组")
            exit(1)
        group_chat_id = group_chat_ids[0]

        # 3. 获取群成员信息
        members = bot.get_members_in_group_by_group_chat_id(group_chat_id)
        print("群成员:")
        print(json.dumps(members, indent=2, ensure_ascii=False))

        # 4. 获取特定成员的 open_id
        member_open_ids = bot.get_member_open_id_by_name(group_chat_id, "王也")
        if not member_open_ids:
            print("未找到指定成员")
            exit(1)
        specific_member_user_open_id = member_open_ids[0]

        # 5. 获取用户信息
        user_infos = bot.get_user_info(emails=[], mobiles=["13267080069"])
        print("用户信息:")
        print(json.dumps(user_infos, indent=2, ensure_ascii=False))

        if not user_infos:
            print("未找到用户信息")
            exit(1)
        user_open_id = user_infos[0].get("user_id")

        # 6. 发送文本消息
        # 6.1 发送普通文本消息
        text_response = bot.send_text_to_user(user_open_id, "Hello, this is a single chat.\nYou know?")
        print("发送文本消息响应:", json.dumps(text_response, indent=2, ensure_ascii=False))

        # 6.2 发送带格式的文本消息
        some_text = TextContent.make_at_someone_pattern(specific_member_user_open_id, "hi", "open_id")
        some_text += TextContent.make_at_all_pattern()
        some_text += TextContent.make_bold_pattern("notice")
        some_text += TextContent.make_italian_pattern("italian")
        some_text += TextContent.make_underline_pattern("underline")
        some_text += TextContent.make_delete_line_pattern("delete line")
        some_text += TextContent.make_url_pattern("www.baidu.com", "百度")

        formatted_text_response = bot.send_text_to_chat(group_chat_id, f"Hi, this is a group.\n{some_text}")
        print("发送格式化文本消息响应:", json.dumps(formatted_text_response, indent=2, ensure_ascii=False))

        # 7. 上传和发送图片
        image_path = "/Users/wayne/Downloads/IMU标定和姿态结算.drawio.png"
        image_key = bot.upload_image(image_path)
        if image_key:
            image_to_user_response = bot.send_image_to_user(user_open_id, image_key)
            print("发送图片到用户响应:", json.dumps(image_to_user_response, indent=2, ensure_ascii=False))

            image_to_chat_response = bot.send_image_to_chat(group_chat_id, image_key)
            print("发送图片到群组响应:", json.dumps(image_to_chat_response, indent=2, ensure_ascii=False))

        # 8. 分享群组和用户
        share_chat_to_user_response = bot.send_shared_chat_to_user(user_open_id, group_chat_id)
        print("分享群组到用户响应:", json.dumps(share_chat_to_user_response, indent=2, ensure_ascii=False))

        share_chat_to_chat_response = bot.send_shared_chat_to_chat(group_chat_id, group_chat_id)
        print("分享群组到群组响应:", json.dumps(share_chat_to_chat_response, indent=2, ensure_ascii=False))

        share_user_to_user_response = bot.send_shared_user_to_user(user_open_id, user_open_id)
        print("分享用户到用户响应:", json.dumps(share_user_to_user_response, indent=2, ensure_ascii=False))

        share_user_to_chat_response = bot.send_shared_user_to_chat(group_chat_id, user_open_id)
        print("分享用户到群组响应:", json.dumps(share_user_to_chat_response, indent=2, ensure_ascii=False))

        # 9. 上传和发送文件
        file_path = "/Users/wayne/Downloads/test.txt"
        file_key = bot.upload_file(file_path)
        if file_key:
            file_to_user_response = bot.send_file_to_user(user_open_id, file_key)
            print("发送文件到用户响应:", json.dumps(file_to_user_response, indent=2, ensure_ascii=False))

            file_to_chat_response = bot.send_file_to_chat(group_chat_id, file_key)
            print("发送文件到群组响应:", json.dumps(file_to_chat_response, indent=2, ensure_ascii=False))

        # 10. 发送富文本消息
        post = PostContent(title="我是标题")

        # 添加文本内容
        line1 = post.make_text_content(text="这是第一行", styles=["bold"])
        post.add_content_in_new_line(line1)

        # 添加@提醒
        line3 = post.make_at_content(specific_member_user_open_id, styles=["bold", "italic"])
        post.add_content_in_new_line(line3)

        # 添加表情和Markdown
        line4_1 = post.make_emoji_content("OK")
        line4_2 = post.make_markdown_content("**helloworld**")
        post.add_content_in_new_line(line4_1)
        post.add_content_in_line(line4_2)

        # 添加代码块
        line6 = post.make_code_block_content(language="python", text='print("Hello, World!")')
        post.add_content_in_new_line(line6)

        # 发送富文本消息
        post_response = bot.send_post_to_chat(group_chat_id, post.get_content())
        print("发送富文本消息响应:", json.dumps(post_response, indent=2, ensure_ascii=False))

        print("所有示例执行完成")

    except Exception as e:
        print(f"错误: {str(e)}")
