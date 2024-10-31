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
from typing import Dict, List, Optional
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

            # Save the image
            with open(image_save_path, "wb") as f:
                f.write(response.file.read())
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

            # Save the file
            with open(save_path, "wb") as f:
                f.write(response.file.read())
            lark.logger.info(f"File downloaded successfully and saved to {save_path}")
        except Exception as e:
            lark.logger.error(f"Exception occurred while downloading file: {e}")
