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
import re
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterable, Dict, Iterable, List, Optional, Tuple, Union

import lark_oapi as lark
import requests
from lark_oapi.api.contact.v3 import *
from lark_oapi.api.im.v1 import *
from pywayne.tools import wayne_print

_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$")
_COMMON_CARD_HEADER_TEMPLATES = (
    "blue",
    "wathet",
    "turquoise",
    "green",
    "yellow",
    "orange",
    "red",
    "carmine",
    "violet",
    "purple",
    "indigo",
    "grey",
)


def _print_info(text: str) -> None:
    wayne_print(text, color="cyan")


def _print_success(text: str) -> None:
    wayne_print(text, color="green", bold=True)


def _print_warn(text: str) -> None:
    wayne_print(text, color="yellow", bold=True)


def _print_error(text: str) -> None:
    wayne_print(text, color="red", bold=True)


def _force_split_by_bytes(text: str, max_bytes: int, encoding: str = "utf-8") -> List[str]:
    """
    Force split a string by encoded byte length.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than 0")
    if not text:
        return []

    out: List[str] = []
    buf = ""
    for ch in text:
        candidate = buf + ch
        if len(candidate.encode(encoding)) <= max_bytes:
            buf = candidate
            continue

        if buf:
            out.append(buf)
        buf = ch

    if buf:
        out.append(buf)
    return out


def _chunk_by_bytes(text: str, max_bytes: int, encoding: str = "utf-8") -> List[str]:
    """
    Split text by paragraph first, then by line, ensuring each chunk's encoded bytes <= max_bytes.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than 0")
    if not text:
        return []
    if len(text.encode(encoding)) <= max_bytes:
        return [text]

    parts = re.split(r"\n{2,}", text)
    out: List[str] = []
    buf = ""

    for part in parts:
        part = part.strip("\n")
        if not part:
            continue

        candidate = f"{buf}\n\n{part}" if buf else part
        if len(candidate.encode(encoding)) <= max_bytes:
            buf = candidate
            continue

        if buf:
            out.append(buf)
            buf = ""

        lines = part.splitlines()
        tmp = ""
        for line in lines:
            candidate_line = f"{tmp}\n{line}" if tmp else line
            if len(candidate_line.encode(encoding)) <= max_bytes:
                tmp = candidate_line
                continue

            if tmp:
                out.append(tmp)
                tmp = ""

            if len(line.encode(encoding)) <= max_bytes:
                tmp = line
            else:
                out.extend(_force_split_by_bytes(line, max_bytes, encoding))

        if tmp:
            out.append(tmp)

    if buf:
        out.append(buf)

    return out


def _split_md_row(line: str) -> List[str]:
    """
    Split one markdown table row into cells.
    """
    content = line.strip().strip("|")
    if not content:
        return [""]
    return [cell.strip() for cell in content.split("|")]


def render_md_table_as_mono(table_md: str, max_col_width: int = 40) -> str:
    """
    Render a Markdown table as a fixed-width plain table.
    """
    lines = [ln.rstrip() for ln in table_md.splitlines() if ln.strip()]
    if len(lines) < 2 or not _TABLE_SEP_RE.match(lines[1]):
        return table_md

    rows = [_split_md_row(lines[0])]
    for line in lines[2:]:
        if "|" not in line:
            break
        rows.append(_split_md_row(line))

    ncol = max(len(row) for row in rows)
    for row in rows:
        if len(row) < ncol:
            row.extend([""] * (ncol - len(row)))

    def clip(value: str) -> str:
        value = value.replace("\t", "  ")
        if len(value) <= max_col_width:
            return value
        return value[: max_col_width - 1] + "..."

    clipped_rows = [[clip(cell) for cell in row] for row in rows]
    widths = [max(len(row[col]) for row in clipped_rows) for col in range(ncol)]

    def format_row(row: List[str]) -> str:
        return " | ".join(row[col].ljust(widths[col]) for col in range(ncol))

    header = format_row(clipped_rows[0])
    separator = "-+-".join("-" * width for width in widths)
    body = "\n".join(format_row(row) for row in clipped_rows[1:])

    if body:
        return f"{header}\n{separator}\n{body}"
    return f"{header}\n{separator}"


def _split_md_blocks(md: str) -> List[Dict[str, str]]:
    """
    Split markdown into generic blocks and table blocks.
    """
    lines = md.splitlines()
    out: List[Dict[str, str]] = []
    buf: List[str] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx]

        if "|" in line and idx + 1 < len(lines) and _TABLE_SEP_RE.match(lines[idx + 1]):
            if "".join(buf).strip():
                out.append({"kind": "md", "text": "\n".join(buf).strip("\n")})
            buf = []

            table_lines = [line, lines[idx + 1]]
            idx += 2
            while idx < len(lines):
                row = lines[idx]
                if not row.strip() or "|" not in row:
                    break
                table_lines.append(row)
                idx += 1

            out.append({"kind": "table", "text": "\n".join(table_lines).strip("\n")})
            continue

        buf.append(line)
        idx += 1

    if "".join(buf).strip():
        out.append({"kind": "md", "text": "\n".join(buf).strip("\n")})

    return out


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

    def add_markdown(self,
                     md: str,
                     *,
                     table_as: str = "code_block",
                     max_chunk_bytes: int = 8_000,
                     mono_max_col_width: int = 40) -> None:
        """
        Add markdown content with table-aware fallback and byte-size chunking.

        Args:
            md (str): Markdown text.
            table_as (str): Table render mode, either "code_block" or "md".
            max_chunk_bytes (int): Max encoded bytes per inserted block.
            mono_max_col_width (int): Max table column width when table_as is "code_block".
        """
        if table_as not in {"code_block", "md"}:
            raise ValueError("table_as must be either 'code_block' or 'md'")

        for block in _split_md_blocks(md):
            if block["kind"] == "table" and table_as == "code_block":
                mono = render_md_table_as_mono(block["text"], max_col_width=mono_max_col_width)
                for part in _chunk_by_bytes(mono, max_chunk_bytes):
                    self.add_content_in_new_line(self.make_code_block_content("text", part))
                continue

            for part in _chunk_by_bytes(block["text"], max_chunk_bytes):
                self.add_content_in_new_line(self.make_markdown_content(part))

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
                _print_warn(f"Unsupported operating system: {system_name}")
        except Exception as e:
            _print_error(f"Failed to open webpage: {e}")


class CardContentV2:
    """
    Helper class for constructing schema=2.0 interactive cards.
    """

    def __init__(self, title: str = "", template: str = "blue"):
        """
        Initialize a minimal schema=2.0 card structure.
        """
        self.card: Dict = {
            "schema": "2.0",
            "config": {
                "update_multi": True,
            },
            "body": {
                "direction": "vertical",
                "padding": "12px 12px 12px 12px",
                "elements": []
            }
        }

        if title:
            self.card["header"] = {
                "title": {
                    "tag": "plain_text",
                    "content": title
                },
                "template": template
            }

    def add_markdown(self, md: str, *, max_chunk_bytes: int = 18_000) -> None:
        """
        Add markdown elements to card body with byte-size chunking.
        """
        for part in _chunk_by_bytes(md, max_chunk_bytes):
            self.card["body"]["elements"].append({
                "tag": "markdown",
                "content": part
            })

    def add_hr(self) -> None:
        """
        Add a horizontal divider to card body.
        """
        self.card["body"]["elements"].append({"tag": "hr"})

    def add_image(self, img_key: str, *, size: str = "large", preview: bool = True) -> None:
        """
        Add an image element to card body.
        """
        self.card["body"]["elements"].append({
            "tag": "img",
            "img_key": img_key,
            "size": size,
            "preview": preview
        })

    def get_card(self) -> Dict:
        """
        Return the complete card payload.
        """
        return self.card

    @staticmethod
    def list_header_templates() -> List[str]:
        """
        Return commonly used Feishu card header template names.

        Note:
            This is a pragmatic helper list for common presets, not a strict
            validation source of truth. Feishu may add new templates over time.
        """
        return list(_COMMON_CARD_HEADER_TEMPLATES)


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
        self.app_id = app_id
        self.app_secret = app_secret
        self.client = lark.Client.builder() \
            .app_id(app_id) \
            .app_secret(app_secret) \
            .log_level(lark.LogLevel.DEBUG) \
            .build()

    @staticmethod
    def _dump_message_content(content: Union[str, Dict[str, Any], List[Any]]) -> str:
        """
        Normalize message content to Feishu API's JSON string format.
        """
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    @staticmethod
    def _coerce_stream_chunk(chunk: Any) -> str:
        """
        Normalize streamed chunks into plain text for card rendering.
        """
        if chunk is None:
            return ""
        if isinstance(chunk, bytes):
            return chunk.decode("utf-8", errors="ignore")
        if isinstance(chunk, str):
            return chunk
        return str(chunk)

    @staticmethod
    def _response_to_dict(response, action: str) -> Dict:
        """
        Convert an SDK response object to a plain dict with consistent error logging.
        """
        if not response.success():
            lark.logger.error(
                f"{action} failed, code: {response.code}, msg: {response.msg}, "
                f"log_id: {response.get_log_id()}"
            )
            return {}

        response_data_obj = getattr(response, "data", None)
        if response_data_obj is None:
            raw = getattr(getattr(response, "raw", None), "content", b"")
            if raw:
                try:
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    raw_json = json.loads(raw)
                    data = raw_json.get("data")
                    if data is None:
                        return raw_json if isinstance(raw_json, dict) else {}
                    lark.logger.info(data)
                    return data
                except Exception:
                    pass
            return {}

        response_data = json.loads(lark.JSON.marshal(response_data_obj, indent=4))
        lark.logger.info(response_data)
        return response_data

    def _get_tenant_access_token(self) -> str:
        """
        Fetch a tenant access token for APIs not exposed by the installed SDK.
        """
        response = requests.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            headers={"Content-Type": "application/json; charset=utf-8"},
            json={"app_id": self.app_id, "app_secret": self.app_secret},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("code", 0) != 0:
            raise RuntimeError(f"get tenant access token failed: {data}")
        return data["tenant_access_token"]

    def _post_open_api_json(self, path: str, payload: Dict[str, Any]) -> Dict:
        """
        Send a raw JSON POST request to Feishu OpenAPI.
        """
        token = self._get_tenant_access_token()
        response = requests.post(
            f"https://open.feishu.cn/open-apis{path}",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("code", 0) != 0:
            lark.logger.error(f"raw api request failed for {path}: {data}")
            return {}
        lark.logger.info(data)
        return data

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

    def send_markdown_to_chat(self,
                              chat_id: str,
                              md_text: str,
                              *,
                              title: str = "",
                              prefer: str = "card_v2",
                              table_fallback: str = "code_block",
                              max_message_bytes: Optional[int] = None) -> List[Dict]:
        """
        Send markdown content to chat with auto route and byte-size auto chunking.

        Args:
            chat_id (str): Chat ID.
            md_text (str): Markdown text.
            title (str): Optional message/card title.
            prefer (str): "card_v2" or "post".
            table_fallback (str): Table render mode for post route ("code_block" or "md").
            max_message_bytes (Optional[int]): Optional per-message byte limit.

        Returns:
            List[Dict]: API responses for all sent message chunks.
        """
        normalized_prefer = prefer.lower()
        if normalized_prefer not in {"card_v2", "post"}:
            raise ValueError("prefer must be either 'card_v2' or 'post'")
        if table_fallback not in {"code_block", "md"}:
            raise ValueError("table_fallback must be either 'code_block' or 'md'")
        if not md_text:
            return []

        default_limit = 18_000 if normalized_prefer == "card_v2" else 8_000
        chunk_limit = max_message_bytes or default_limit
        text_chunks = _chunk_by_bytes(md_text, chunk_limit)

        responses: List[Dict] = []
        total_chunks = len(text_chunks)

        for idx, chunk in enumerate(text_chunks, start=1):
            chunk_title = title
            if title and total_chunks > 1:
                chunk_title = f"{title} ({idx}/{total_chunks})"

            if normalized_prefer == "card_v2":
                card = CardContentV2(title=chunk_title)
                card.add_markdown(chunk, max_chunk_bytes=chunk_limit)
                responses.append(self.send_interactive_to_chat(chat_id, card.get_card()))
                continue

            post = PostContent(title=chunk_title)
            post.add_markdown(chunk, table_as=table_fallback, max_chunk_bytes=chunk_limit)
            responses.append(self.send_post_to_chat(chat_id, post.get_content()))

        return responses

    def build_streaming_card(self,
                             md_text: str,
                             *,
                             title: str = "",
                             template: str = "blue",
                             streaming: bool = True,
                             status_text: str = "",
                             max_chunk_bytes: int = 18_000) -> Dict[str, Any]:
        """
        Build a schema=2.0 card suitable for progressive in-place updates.

        Notes:
            - The generated card keeps `config.update_multi=true`, which is required
              when updating a shared message card for all viewers.
            - Feishu card updates are rate-limited; callers should throttle update
              frequency on their side.
        """
        card = CardContentV2(title=title, template=template)
        body = md_text or ""

        if streaming:
            footer = status_text or "Generating..."
            body = f"{body}\n\n---\n\n{footer}" if body else footer
        elif status_text:
            body = f"{body}\n\n---\n\n{status_text}" if body else status_text

        if not body:
            body = "..."

        card.add_markdown(body, max_chunk_bytes=max_chunk_bytes)
        return card.get_card()

    def reply_streaming_card(self,
                             message_id: str,
                             *,
                             title: str = "Streaming Reply",
                             template: str = "blue",
                             initial_md: str = "",
                             reply_in_thread: bool = False,
                             uuid: str = "",
                             status_text: str = "Generating...",
                             max_chunk_bytes: int = 18_000) -> Dict:
        """
        Reply with an initial interactive card intended for subsequent stream updates.
        """
        card = self.build_streaming_card(
            initial_md,
            title=title,
            template=template,
            streaming=True,
            status_text=status_text,
            max_chunk_bytes=max_chunk_bytes,
        )
        return self.reply_message(
            message_id,
            "interactive",
            card,
            reply_in_thread=reply_in_thread,
            uuid=uuid,
        )

    def update_streaming_card(self,
                              message_id: str,
                              md_text: str,
                              *,
                              title: str = "Streaming Reply",
                              template: str = "blue",
                              done: bool = False,
                              status_text: str = "",
                              max_chunk_bytes: int = 18_000) -> Dict:
        """
        Update a previously sent streaming card with the latest full markdown text.

        Args:
            message_id: Interactive card message ID to update.
            md_text: The full current markdown text, not a delta chunk.
            done: Whether streaming has finished.
            status_text: Optional footer line. Defaults to `Generating...` while
                         streaming and no footer when `done=True`.
        """
        card = self.build_streaming_card(
            md_text,
            title=title,
            template=template,
            streaming=not done,
            status_text=status_text if done else (status_text or "Generating..."),
            max_chunk_bytes=max_chunk_bytes,
        )
        return self.update_interactive_card(message_id, card)

    def recolor_streaming_card(self,
                               message_id: str,
                               md_text: str,
                               *,
                               title: str = "Streaming Reply",
                               template: str = "green",
                               status_text: str = "Done",
                               done: bool = True,
                               max_chunk_bytes: int = 18_000) -> Dict:
        """
        Convenience helper to switch a streaming card to another header template.

        Typical usage:
            - blue while generating
            - green when done
            - red when failed
            - orange when partially complete
        """
        return self.update_streaming_card(
            message_id,
            md_text,
            title=title,
            template=template,
            done=done,
            status_text=status_text,
            max_chunk_bytes=max_chunk_bytes,
        )

    def stream_reply_card(self,
                          source_message_id: str,
                          text_stream: Iterable[Any],
                          *,
                          title: str = "Streaming Reply",
                          template: str = "blue",
                          initial_md: str = "",
                          reply_in_thread: bool = False,
                          uuid: str = "",
                          update_interval: float = 0.25,
                          status_text: str = "Generating...",
                          final_status_text: str = "",
                          final_template: Optional[str] = "green",
                          max_chunk_bytes: int = 18_000) -> Dict[str, Any]:
        """
        Consume a sync text stream and keep one reply card updated in-place.
        """
        reply = self.reply_streaming_card(
            source_message_id,
            title=title,
            template=template,
            initial_md=initial_md,
            reply_in_thread=reply_in_thread,
            uuid=uuid,
            status_text=status_text,
            max_chunk_bytes=max_chunk_bytes,
        )
        card_message_id = reply.get("message_id", "")
        if not card_message_id:
            return {"reply": reply, "final": {}, "message_id": "", "text": initial_md}

        full_text = initial_md
        last_update = 0.0

        for chunk in text_stream:
            chunk_text = self._coerce_stream_chunk(chunk)
            if not chunk_text:
                continue
            full_text += chunk_text

            now = time.monotonic()
            if now - last_update < update_interval:
                continue

            self.update_streaming_card(
                card_message_id,
                full_text,
                title=title,
                template=template,
                done=False,
                status_text=status_text,
                max_chunk_bytes=max_chunk_bytes,
            )
            last_update = now

        final = self.update_streaming_card(
            card_message_id,
            full_text,
            title=title,
            template=final_template or template,
            done=True,
            status_text=final_status_text,
            max_chunk_bytes=max_chunk_bytes,
        )
        return {
            "reply": reply,
            "final": final,
            "message_id": card_message_id,
            "text": full_text,
        }

    async def astream_reply_card(self,
                                 source_message_id: str,
                                 text_stream: AsyncIterable[Any],
                                 *,
                                 title: str = "Streaming Reply",
                                 template: str = "blue",
                                 initial_md: str = "",
                                 reply_in_thread: bool = False,
                                 uuid: str = "",
                                 update_interval: float = 0.25,
                                 status_text: str = "Generating...",
                                 final_status_text: str = "",
                                 final_template: Optional[str] = "green",
                                 max_chunk_bytes: int = 18_000) -> Dict[str, Any]:
        """
        Async variant of `stream_reply_card` for async generators.
        """
        reply = self.reply_streaming_card(
            source_message_id,
            title=title,
            template=template,
            initial_md=initial_md,
            reply_in_thread=reply_in_thread,
            uuid=uuid,
            status_text=status_text,
            max_chunk_bytes=max_chunk_bytes,
        )
        card_message_id = reply.get("message_id", "")
        if not card_message_id:
            return {"reply": reply, "final": {}, "message_id": "", "text": initial_md}

        full_text = initial_md
        last_update = 0.0

        async for chunk in text_stream:
            chunk_text = self._coerce_stream_chunk(chunk)
            if not chunk_text:
                continue
            full_text += chunk_text

            now = time.monotonic()
            if now - last_update < update_interval:
                continue

            self.update_streaming_card(
                card_message_id,
                full_text,
                title=title,
                template=template,
                done=False,
                status_text=status_text,
                max_chunk_bytes=max_chunk_bytes,
            )
            last_update = now

        final = self.update_streaming_card(
            card_message_id,
            full_text,
            title=title,
            template=final_template or template,
            done=True,
            status_text=final_status_text,
            max_chunk_bytes=max_chunk_bytes,
        )
        return {
            "reply": reply,
            "final": final,
            "message_id": card_message_id,
            "text": full_text,
        }

    def reply_message(self,
                      message_id: str,
                      msg_type: str,
                      content: Union[str, Dict[str, Any], List[Any]],
                      *,
                      reply_in_thread: bool = False,
                      uuid: str = "") -> Dict:
        """
        Reply to a message, optionally in thread mode.
        """
        request_body_builder = ReplyMessageRequestBody.builder() \
            .msg_type(msg_type) \
            .content(self._dump_message_content(content)) \
            .reply_in_thread(reply_in_thread)
        if uuid:
            request_body_builder = request_body_builder.uuid(uuid)

        request = ReplyMessageRequest.builder() \
            .message_id(message_id) \
            .request_body(request_body_builder.build()) \
            .build()

        response = self.client.im.v1.message.reply(request)
        return self._response_to_dict(response, "reply_message")

    def forward_message(self,
                        message_id: str,
                        receive_id: str,
                        *,
                        receive_id_type: str = "chat_id",
                        uuid: str = "") -> Dict:
        """
        Forward an existing message to a user or chat.
        """
        request_builder = ForwardMessageRequest.builder() \
            .receive_id_type(receive_id_type) \
            .message_id(message_id) \
            .request_body(ForwardMessageRequestBody.builder().receive_id(receive_id).build())
        if uuid:
            request_builder = request_builder.uuid(uuid)

        response = self.client.im.v1.message.forward(request_builder.build())
        return self._response_to_dict(response, "forward_message")

    def recall_message(self, message_id: str) -> Dict:
        """
        Recall a message sent by the bot.
        """
        request = DeleteMessageRequest.builder().message_id(message_id).build()
        response = self.client.im.v1.message.delete(request)
        return self._response_to_dict(response, "recall_message")

    def get_message(self, message_id: str, *, user_id_type: str = "open_id") -> Dict:
        """
        Get message details by message_id.
        """
        request = GetMessageRequest.builder() \
            .message_id(message_id) \
            .user_id_type(user_id_type) \
            .build()
        response = self.client.im.v1.message.get(request)
        return self._response_to_dict(response, "get_message")

    def get_message_list(self,
                         chat_id: str,
                         start_time: str,
                         end_time: str,
                         *,
                         sort_type: str = "",
                         page_size: int = 50,
                         page_token: str = "") -> Dict:
        """
        List historical messages in a chat.
        """
        request_builder = ListMessageRequest.builder() \
            .container_id_type("chat") \
            .container_id(chat_id) \
            .start_time(start_time) \
            .end_time(end_time) \
            .page_size(page_size)
        if sort_type:
            request_builder = request_builder.sort_type(sort_type)
        if page_token:
            request_builder = request_builder.page_token(page_token)

        response = self.client.im.v1.message.list(request_builder.build())
        return self._response_to_dict(response, "get_message_list")

    def update_message(self,
                       message_id: str,
                       msg_type: str,
                       content: Union[str, Dict[str, Any], List[Any]]) -> Dict:
        """
        Replace the content of an existing message.
        """
        request = UpdateMessageRequest.builder() \
            .message_id(message_id) \
            .request_body(UpdateMessageRequestBody.builder()
                          .msg_type(msg_type)
                          .content(self._dump_message_content(content))
                          .build()) \
            .build()
        response = self.client.im.v1.message.update(request)
        return self._response_to_dict(response, "update_message")

    def patch_message(self,
                      message_id: str,
                      content: Union[str, Dict[str, Any], List[Any]]) -> Dict:
        """
        Partially update message content, commonly used for interactive cards.
        """
        request = PatchMessageRequest.builder() \
            .message_id(message_id) \
            .request_body(PatchMessageRequestBody.builder()
                          .content(self._dump_message_content(content))
                          .build()) \
            .build()
        response = self.client.im.v1.message.patch(request)
        return self._response_to_dict(response, "patch_message")

    def update_interactive_card(self, message_id: str, card: Dict[str, Any]) -> Dict:
        """
        Update an existing interactive card in-place.
        """
        return self.patch_message(message_id, card)

    def get_message_read_users(self,
                               message_id: str,
                               *,
                               user_id_type: str = "open_id",
                               page_size: int = 50,
                               page_token: str = "") -> Dict:
        """
        Get read-receipt users for a message.
        """
        request_builder = ReadUsersMessageRequest.builder() \
            .message_id(message_id) \
            .user_id_type(user_id_type) \
            .page_size(page_size)
        if page_token:
            request_builder = request_builder.page_token(page_token)

        response = self.client.im.v1.message.read_users(request_builder.build())
        return self._response_to_dict(response, "get_message_read_users")

    def urgent_message(self,
                       message_id: str,
                       urgent_type: str,
                       user_open_ids: List[str],
                       *,
                       user_id_type: str = "open_id") -> Dict:
        """
        Send an urgent notification for an existing message.
        """
        urgent_receivers = UrgentReceivers.builder().user_id_list(user_open_ids).build()
        normalized_type = urgent_type.lower()

        if normalized_type == "app":
            request = UrgentAppMessageRequest.builder() \
                .message_id(message_id) \
                .user_id_type(user_id_type) \
                .request_body(urgent_receivers) \
                .build()
            response = self.client.im.v1.message.urgent_app(request)
        elif normalized_type == "phone":
            request = UrgentPhoneMessageRequest.builder() \
                .message_id(message_id) \
                .user_id_type(user_id_type) \
                .request_body(urgent_receivers) \
                .build()
            response = self.client.im.v1.message.urgent_phone(request)
        elif normalized_type == "sms":
            request = UrgentSmsMessageRequest.builder() \
                .message_id(message_id) \
                .user_id_type(user_id_type) \
                .request_body(urgent_receivers) \
                .build()
            response = self.client.im.v1.message.urgent_sms(request)
        else:
            raise ValueError("urgent_type must be one of: app, phone, sms")

        return self._response_to_dict(response, f"urgent_message[{normalized_type}]")

    def add_reaction(self, message_id: str, emoji_type: str) -> Dict:
        """
        Add an emoji reaction to a message.

        Notes:
            - `emoji_type` uses Feishu's message-reaction emoji code, not the visual emoji glyph.
            - Commonly used values include: `THUMBSUP`, `OK`, `HEART`, `HAHA`.
            - The full supported list is maintained by Feishu and can be opened with
              `PostContent.list_emoji_types()`, which jumps to:
              https://open.feishu.cn/document/server-docs/im-v1/message-reaction/emojis-introduce
        """
        request = CreateMessageReactionRequest.builder() \
            .message_id(message_id) \
            .request_body(CreateMessageReactionRequestBody.builder()
                          .reaction_type(Emoji.builder().emoji_type(emoji_type).build())
                          .build()) \
            .build()
        response = self.client.im.v1.message_reaction.create(request)
        return self._response_to_dict(response, "add_reaction")

    def delete_reaction(self, message_id: str, reaction_id: str) -> Dict:
        """
        Delete a previously added reaction from a message.
        """
        request = DeleteMessageReactionRequest.builder() \
            .message_id(message_id) \
            .reaction_id(reaction_id) \
            .build()
        response = self.client.im.v1.message_reaction.delete(request)
        return self._response_to_dict(response, "delete_reaction")

    def list_reactions(self,
                       message_id: str,
                       *,
                       reaction_type: str = "",
                       user_id_type: str = "open_id",
                       page_size: int = 50,
                       page_token: str = "") -> Dict:
        """
        List reactions on a message.
        """
        request_builder = ListMessageReactionRequest.builder() \
            .message_id(message_id) \
            .user_id_type(user_id_type) \
            .page_size(page_size)
        if reaction_type:
            request_builder = request_builder.reaction_type(reaction_type)
        if page_token:
            request_builder = request_builder.page_token(page_token)

        response = self.client.im.v1.message_reaction.list(request_builder.build())
        return self._response_to_dict(response, "list_reactions")

    def pin_message(self, message_id: str) -> Dict:
        """
        Pin a message in chat.
        """
        request = CreatePinRequest.builder() \
            .request_body(CreatePinRequestBody.builder().message_id(message_id).build()) \
            .build()
        response = self.client.im.v1.pin.create(request)
        return self._response_to_dict(response, "pin_message")

    def unpin_message(self, message_id: str) -> Dict:
        """
        Unpin a message in chat.
        """
        request = DeletePinRequest.builder().message_id(message_id).build()
        response = self.client.im.v1.pin.delete(request)
        return self._response_to_dict(response, "unpin_message")

    def list_pinned_messages(self,
                             chat_id: str,
                             *,
                             start_time: str = "",
                             end_time: str = "",
                             page_size: int = 50,
                             page_token: str = "") -> Dict:
        """
        List pinned messages in a chat.
        """
        request_builder = ListPinRequest.builder() \
            .chat_id(chat_id) \
            .page_size(page_size)
        if start_time:
            request_builder = request_builder.start_time(start_time)
        if end_time:
            request_builder = request_builder.end_time(end_time)
        if page_token:
            request_builder = request_builder.page_token(page_token)

        response = self.client.im.v1.pin.list(request_builder.build())
        return self._response_to_dict(response, "list_pinned_messages")

    def create_chat(self,
                    name: str,
                    user_open_ids: List[str],
                    description: str = "",
                    *,
                    avatar: str = "",
                    owner_open_id: str = "",
                    bot_ids: Optional[List[str]] = None,
                    set_bot_manager: bool = False,
                    uuid: str = "") -> Dict:
        """
        Create a new chat and optionally invite users/bots.
        """
        request_body_builder = CreateChatRequestBody.builder() \
            .name(name) \
            .description(description) \
            .user_id_list(user_open_ids)
        if avatar:
            request_body_builder = request_body_builder.avatar(avatar)
        if owner_open_id:
            request_body_builder = request_body_builder.owner_id(owner_open_id)
        if bot_ids:
            request_body_builder = request_body_builder.bot_id_list(bot_ids)

        request_builder = CreateChatRequest.builder() \
            .user_id_type("open_id") \
            .set_bot_manager(set_bot_manager) \
            .request_body(request_body_builder.build())
        if uuid:
            request_builder = request_builder.uuid(uuid)

        response = self.client.im.v1.chat.create(request_builder.build())
        return self._response_to_dict(response, "create_chat")

    def delete_chat(self, chat_id: str) -> Dict:
        """
        Delete a chat.
        """
        request = DeleteChatRequest.builder().chat_id(chat_id).build()
        response = self.client.im.v1.chat.delete(request)
        return self._response_to_dict(response, "delete_chat")

    def update_chat(self,
                    chat_id: str,
                    *,
                    name: str = "",
                    description: str = "",
                    avatar: str = "",
                    owner_open_id: str = "") -> Dict:
        """
        Update basic chat information.
        """
        request_body_builder = UpdateChatRequestBody.builder()
        if avatar:
            request_body_builder = request_body_builder.avatar(avatar)
        if name:
            request_body_builder = request_body_builder.name(name)
        if description:
            request_body_builder = request_body_builder.description(description)
        if owner_open_id:
            request_body_builder = request_body_builder.owner_id(owner_open_id)

        request = UpdateChatRequest.builder() \
            .user_id_type("open_id") \
            .chat_id(chat_id) \
            .request_body(request_body_builder.build()) \
            .build()
        response = self.client.im.v1.chat.update(request)
        return self._response_to_dict(response, "update_chat")

    def add_members_to_chat(self,
                            chat_id: str,
                            user_open_ids: List[str],
                            *,
                            succeed_type: int = 0) -> Dict:
        """
        Add members to a chat.
        """
        request = CreateChatMembersRequest.builder() \
            .member_id_type("open_id") \
            .succeed_type(succeed_type) \
            .chat_id(chat_id) \
            .request_body(CreateChatMembersRequestBody.builder().id_list(user_open_ids).build()) \
            .build()
        response = self.client.im.v1.chat_members.create(request)
        return self._response_to_dict(response, "add_members_to_chat")

    def remove_members_from_chat(self, chat_id: str, user_open_ids: List[str]) -> Dict:
        """
        Remove members from a chat.
        """
        request = DeleteChatMembersRequest.builder() \
            .member_id_type("open_id") \
            .chat_id(chat_id) \
            .request_body(DeleteChatMembersRequestBody.builder().id_list(user_open_ids).build()) \
            .build()
        response = self.client.im.v1.chat_members.delete(request)
        return self._response_to_dict(response, "remove_members_from_chat")

    def set_chat_admin(self, chat_id: str, user_open_ids: List[str], *, is_admin: bool = True) -> Dict:
        """
        Add or remove chat administrators.
        """
        if is_admin:
            request = AddManagersChatManagersRequest.builder() \
                .member_id_type("open_id") \
                .chat_id(chat_id) \
                .request_body(AddManagersChatManagersRequestBody.builder().manager_ids(user_open_ids).build()) \
                .build()
            response = self.client.im.v1.chat_managers.add_managers(request)
            return self._response_to_dict(response, "set_chat_admin[add]")

        request = DeleteManagersChatManagersRequest.builder() \
            .member_id_type("open_id") \
            .chat_id(chat_id) \
            .request_body(DeleteManagersChatManagersRequestBody.builder().manager_ids(user_open_ids).build()) \
            .build()
        response = self.client.im.v1.chat_managers.delete_managers(request)
        return self._response_to_dict(response, "set_chat_admin[remove]")

    def transfer_chat_owner(self, chat_id: str, new_owner_open_id: str) -> Dict:
        """
        Transfer chat ownership to another member.
        """
        return self.update_chat(chat_id, owner_open_id=new_owner_open_id)

    def get_chat_announcement(self, chat_id: str) -> Dict:
        """
        Get the current chat announcement.
        """
        request = GetChatAnnouncementRequest.builder() \
            .user_id_type("open_id") \
            .chat_id(chat_id) \
            .build()
        response = self.client.im.v1.chat_announcement.get(request)
        return self._response_to_dict(response, "get_chat_announcement")

    def set_chat_announcement(self,
                              chat_id: str,
                              *,
                              requests: Union[str, List[str]],
                              revision: str = "") -> Dict:
        """
        Patch chat announcement with raw request operations from Feishu's announcement API.
        """
        normalized_requests = [requests] if isinstance(requests, str) else requests
        request = PatchChatAnnouncementRequest.builder() \
            .chat_id(chat_id) \
            .request_body(PatchChatAnnouncementRequestBody.builder()
                          .revision(revision)
                          .requests(normalized_requests)
                          .build()) \
            .build()
        response = self.client.im.v1.chat_announcement.patch(request)
        return self._response_to_dict(response, "set_chat_announcement")

    def batch_send_message(self,
                           msg_type: str,
                           *,
                           content: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
                           card: Optional[Dict[str, Any]] = None,
                           user_open_ids: Optional[List[str]] = None,
                           department_ids: Optional[List[str]] = None,
                           user_ids: Optional[List[str]] = None,
                           union_ids: Optional[List[str]] = None) -> Dict:
        """
        Batch-send a message to users or departments.

        Notes:
            - This uses Feishu's `/message/v4/batch_send/` endpoint.
            - Batch messages cannot be updated or replied to.
            - The endpoint only supports user/department targets, not chats.
        """
        normalized_type = msg_type.lower()
        payload: Dict[str, Any] = {
            "msg_type": normalized_type,
            "open_ids": user_open_ids or [],
            "department_ids": department_ids or [],
            "user_ids": user_ids or [],
            "union_ids": union_ids or [],
        }

        if normalized_type == "interactive":
            if not card:
                raise ValueError("card is required when msg_type='interactive'")
            payload["card"] = card
        else:
            if content is None:
                raise ValueError("content is required when msg_type is not 'interactive'")
            if normalized_type == "text" and isinstance(content, str):
                payload["content"] = {"text": content}
            elif isinstance(content, str):
                try:
                    payload["content"] = json.loads(content)
                except json.JSONDecodeError:
                    raise ValueError("content must be a dict/list or valid JSON string for this msg_type")
            else:
                payload["content"] = content

        if not any([payload["open_ids"], payload["department_ids"], payload["user_ids"], payload["union_ids"]]):
            raise ValueError("at least one target list must be provided")

        return self._post_open_api_json("/message/v4/batch_send/", payload)

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
            _print_error(f"获取群聊信息失败: {response.code}, {response.msg}")
            
        # 获取用户信息
        user_name = ""
        request = lark.api.contact.v3.GetUserRequest.builder().user_id(user_id).build()
        response = self.client.contact.v3.user.get(request)
        if response.success():
            user_name = response.data.user.name
        else:
            _print_error(f"获取用户信息失败: {response.code}, {response.msg}")
            
        return chat_name, user_name


if __name__ == '__main__':
    # 创建机器人实例
    bot = LarkBot(
        app_id="cli_xxx",
        app_secret="xxxxxx"
    )

    try:
        # 1. 获取群组列表
        group_list = bot.get_group_list()
        _print_info("群组列表:")
        _print_info(json.dumps(group_list, indent=2, ensure_ascii=False))

        # 2. 获取特定群组的ID
        group_chat_ids = bot.get_group_chat_id_by_name("测试3")
        if not group_chat_ids:
            _print_warn("未找到群组")
            exit(1)
        group_chat_id = group_chat_ids[0]

        # 3. 获取群成员信息
        members = bot.get_members_in_group_by_group_chat_id(group_chat_id)
        _print_info("群成员:")
        _print_info(json.dumps(members, indent=2, ensure_ascii=False))

        # 4. 获取特定成员的 open_id
        member_open_ids = bot.get_member_open_id_by_name(group_chat_id, "王也")
        if not member_open_ids:
            _print_warn("未找到指定成员")
            exit(1)
        specific_member_user_open_id = member_open_ids[0]

        # 5. 获取用户信息
        user_infos = bot.get_user_info(emails=[], mobiles=["13267080069"])
        _print_info("用户信息:")
        _print_info(json.dumps(user_infos, indent=2, ensure_ascii=False))

        if not user_infos:
            _print_warn("未找到用户信息")
            exit(1)
        user_open_id = user_infos[0].get("user_id")

        # 6. 发送文本消息
        # 6.1 发送普通文本消息
        text_response = bot.send_text_to_user(user_open_id, "Hello, this is a single chat.\nYou know?")
        _print_success(f"发送文本消息响应: {json.dumps(text_response, indent=2, ensure_ascii=False)}")

        # 6.2 发送带格式的文本消息
        some_text = TextContent.make_at_someone_pattern(specific_member_user_open_id, "hi", "open_id")
        some_text += TextContent.make_at_all_pattern()
        some_text += TextContent.make_bold_pattern("notice")
        some_text += TextContent.make_italian_pattern("italian")
        some_text += TextContent.make_underline_pattern("underline")
        some_text += TextContent.make_delete_line_pattern("delete line")
        some_text += TextContent.make_url_pattern("www.baidu.com", "百度")

        formatted_text_response = bot.send_text_to_chat(group_chat_id, f"Hi, this is a group.\n{some_text}")
        _print_success(f"发送格式化文本消息响应: {json.dumps(formatted_text_response, indent=2, ensure_ascii=False)}")

        # 7. 上传和发送图片
        image_path = "/Users/wayne/Downloads/IMU标定和姿态结算.drawio.png"
        image_key = bot.upload_image(image_path)
        if image_key:
            image_to_user_response = bot.send_image_to_user(user_open_id, image_key)
            _print_success(f"发送图片到用户响应: {json.dumps(image_to_user_response, indent=2, ensure_ascii=False)}")

            image_to_chat_response = bot.send_image_to_chat(group_chat_id, image_key)
            _print_success(f"发送图片到群组响应: {json.dumps(image_to_chat_response, indent=2, ensure_ascii=False)}")

        # 8. 分享群组和用户
        share_chat_to_user_response = bot.send_shared_chat_to_user(user_open_id, group_chat_id)
        _print_success(f"分享群组到用户响应: {json.dumps(share_chat_to_user_response, indent=2, ensure_ascii=False)}")

        share_chat_to_chat_response = bot.send_shared_chat_to_chat(group_chat_id, group_chat_id)
        _print_success(f"分享群组到群组响应: {json.dumps(share_chat_to_chat_response, indent=2, ensure_ascii=False)}")

        share_user_to_user_response = bot.send_shared_user_to_user(user_open_id, user_open_id)
        _print_success(f"分享用户到用户响应: {json.dumps(share_user_to_user_response, indent=2, ensure_ascii=False)}")

        share_user_to_chat_response = bot.send_shared_user_to_chat(group_chat_id, user_open_id)
        _print_success(f"分享用户到群组响应: {json.dumps(share_user_to_chat_response, indent=2, ensure_ascii=False)}")

        # 9. 上传和发送文件
        file_path = "/Users/wayne/Downloads/test.txt"
        file_key = bot.upload_file(file_path)
        if file_key:
            file_to_user_response = bot.send_file_to_user(user_open_id, file_key)
            _print_success(f"发送文件到用户响应: {json.dumps(file_to_user_response, indent=2, ensure_ascii=False)}")

            file_to_chat_response = bot.send_file_to_chat(group_chat_id, file_key)
            _print_success(f"发送文件到群组响应: {json.dumps(file_to_chat_response, indent=2, ensure_ascii=False)}")

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
        _print_success(f"发送富文本消息响应: {json.dumps(post_response, indent=2, ensure_ascii=False)}")

        _print_success("所有示例执行完成")

    except Exception as e:
        _print_error(f"错误: {str(e)}")
