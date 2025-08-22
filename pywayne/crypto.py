#!/usr/bin/env python
# encoding: utf-8

"""
@author: Wayne  
@contact: wangye.hope@gmail.com
@software: Cursor
@file: crypto.py
@time: 2025/08/22
"""

import base64
import hashlib
from typing import Union

# 混淆的核心函数 - 通过base64编码隐藏实现
def _exec_b64(code_b64):
    """执行base64编码的代码并返回命名空间"""
    code = base64.b64decode(code_b64).decode('utf-8')
    namespace = {
        'base64': base64, 'hashlib': hashlib, 'bytearray': bytearray, 
        'enumerate': enumerate, 'len': len, 'ValueError': ValueError,
        'chr': chr, 'ord': ord
    }
    exec(code, namespace)
    return namespace

# 生成密钥的函数
_key_ns = _exec_b64(b'ZGVmIGdlbl9rZXkocGFzc3dvcmQ9Tm9uZSk6CiAgICBpZiBwYXNzd29yZCBpcyBOb25lOgogICAgICAgIHNlZWQgPSAiIi5qb2luKFtjaHIob3JkKGMpIF4gNykgZm9yIGMgaW4gInB5d2F5bmVfc2VjcmV0XzIwMjQiXSkKICAgIGVsc2U6CiAgICAgICAgc2VlZCA9IHBhc3N3b3JkCiAgICBrZXlfaGFzaCA9IGhhc2hsaWIuc2hhMjU2KHNlZWQuZW5jb2RlKCJ1dGYtOCIpKS5kaWdlc3QoKQogICAgcmV0dXJuIGJhc2U2NC51cmxzYWZlX2I2NGVuY29kZShrZXlfaGFzaCk=')
_gen_key = _key_ns['gen_key']

# 获取加密器的函数
_cipher_ns = _exec_b64(b'ZGVmIGdldF9jaXBoZXIoKToKICAgIHRyeToKICAgICAgICBmcm9tIGNyeXB0b2dyYXBoeS5mZXJuZXQgaW1wb3J0IEZlcm5ldAogICAgICAgIHJldHVybiBGZXJuZXQKICAgIGV4Y2VwdCBJbXBvcnRFcnJvcjoKICAgICAgICByZXR1cm4gTm9uZQ==')
_get_cipher = _cipher_ns['get_cipher']

# XOR加密函数
_xor_enc_ns = _exec_b64(b'ZGVmIHhvcl9lbmMoZGF0YSwga2V5KToKICAgIGtiID0ga2V5LmVuY29kZSgidXRmLTgiKQogICAgZGIgPSBkYXRhLmVuY29kZSgidXRmLTgiKQogICAgZSA9IGJ5dGVhcnJheSgpCiAgICBmb3IgaSwgYiBpbiBlbnVtZXJhdGUoZGIpOgogICAgICAgIGUuYXBwZW5kKGIgXiBrYltpICUgbGVuKGtiKV0pCiAgICByZXR1cm4gYmFzZTY0LmI2NGVuY29kZShlKS5kZWNvZGUoInV0Zi04Iik=')
_xor_encrypt = _xor_enc_ns['xor_enc']

# XOR解密函数
_xor_dec_ns = _exec_b64(b'ZGVmIHhvcl9kZWMoZWQsIGspOgogICAgdHJ5OgogICAgICAgIGtiID0gay5lbmNvZGUoInV0Zi04IikKICAgICAgICBlYiA9IGJhc2U2NC5iNjRkZWNvZGUoZWQuZW5jb2RlKCJ1dGYtOCIpKQogICAgICAgIGQgPSBieXRlYXJyYXkoKQogICAgICAgIGZvciBpLCBiIGluIGVudW1lcmF0ZShlYik6CiAgICAgICAgICAgIGQuYXBwZW5kKGIgXiBrYltpICUgbGVuKGtiKV0pCiAgICAgICAgcmV0dXJuIGQuZGVjb2RlKCJ1dGYtOCIpCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIuino+WvhuWkseaVpO+8mua5sOaNruagvOW8j+mUmeivr+aIluWvhOmSpeS4jeato+ehriIp')
_xor_decrypt = _xor_dec_ns['xor_dec']

def _core_encrypt(text, password=None):
    """核心加密函数"""
    Fernet = _get_cipher()
    if Fernet is not None:
        key = _gen_key(password)
        f = Fernet(key)
        encrypted = f.encrypt(text.encode('utf-8'))
        return base64.b64encode(encrypted).decode('utf-8')
    else:
        return _xor_encrypt(text, password or "default_key")

def _core_decrypt(encrypted_text, password=None):
    """核心解密函数"""
    Fernet = _get_cipher()
    if Fernet is not None:
        try:
            key = _gen_key(password)
            f = Fernet(key)
            encrypted_data = base64.b64decode(encrypted_text.encode('utf-8'))
            decrypted = f.decrypt(encrypted_data)
            return decrypted.decode('utf-8')
        except Exception:
            raise ValueError("解密失败：数据格式错误或密钥不正确")
    else:
        return _xor_decrypt(encrypted_text, password or "default_key")

def encrypt(text: Union[str, bytes], password: str = None) -> str:
    """
    加密字符串
    
    Args:
        text: 要加密的文本（字符串或字节）
        password: 可选的密码，如果不提供则使用默认密钥
        
    Returns:
        str: 加密后的base64编码字符串
        
    Raises:
        ValueError: 当输入参数无效时
        
    Examples:
        >>> encrypted = encrypt("Hello World")
        >>> print(encrypted)
        gAAAAABh...
        
        >>> encrypted_with_pwd = encrypt("Secret message", "my_password")
        >>> print(encrypted_with_pwd)
        gAAAAABh...
    """
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    elif not isinstance(text, str):
        raise ValueError("text must be str or bytes")
    
    if text == "":
        return ""
    
    return _core_encrypt(text, password)

def decrypt(encrypted_text: str, password: str = None) -> str:
    """
    解密字符串
    
    Args:
        encrypted_text: 要解密的base64编码字符串
        password: 可选的密码，必须与加密时使用的密码相同
        
    Returns:
        str: 解密后的原始字符串
        
    Raises:
        ValueError: 当解密失败时（数据损坏或密码错误）
        
    Examples:
        >>> original = decrypt(encrypted_text)
        >>> print(original)
        Hello World
        
        >>> original_with_pwd = decrypt(encrypted_text, "my_password") 
        >>> print(original_with_pwd)
        Secret message
    """
    if not isinstance(encrypted_text, str):
        raise ValueError("encrypted_text must be str")
    
    if encrypted_text == "":
        return ""
    
    return _core_decrypt(encrypted_text, password)

# 清理敏感变量
del _exec_b64, _key_ns, _cipher_ns, _xor_enc_ns, _xor_dec_ns

# 导出的公共接口
__all__ = ['encrypt', 'decrypt']
