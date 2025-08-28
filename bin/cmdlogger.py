#!/usr/bin/env python3

import sys
import subprocess
import threading
import argparse
import os

class CommandLogger:
    """
    執行一個子命令，並將其 stdin, stdout, stderr 記錄到檔案中，
    同時在控制台進行實時的 I/O 轉發。
    """
    def __init__(self, command, log_file):
        """
        初始化 CommandLogger。
        :param command: 一個包含命令及其參數的列表 (list)。
        :param log_file: 用於儲存 I/O 記錄的檔案路徑。
        """
        if not isinstance(command, list) or not command:
            raise ValueError("命令必須是一個非空的列表。")
        self.command = command
        self.log_file = log_file
        self.process = None
        self._log_file_handle = None
        self._threads = []
        self.exit_code = None

    def _forward_stream(self, src_stream, dest_stream, log_prefix):
        """
        一個通用的資料流轉發和記錄函數。
        :param src_stream: 來源資料流 (e.g., process.stdout)
        :param dest_stream: 目標資料流 (e.g., sys.stdout.buffer)
        :param log_prefix: 在日誌中使用的前綴 (e.g., "輸出")
        """
        try:
            # 對於 stdin 處理，需要特殊邏輯來避免卡死
            if log_prefix == "輸入":
                self._forward_stdin(src_stream, dest_stream, log_prefix)
            else:
                # 處理 stdout 和 stderr
                while True:
                    line_bytes = src_stream.readline()
                    if not line_bytes:
                        break
                    
                    try:
                        line_str = line_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        line_str = f"[非UTF-8資料, {len(line_bytes)} bytes]\n"

                    if self._log_file_handle and not self._log_file_handle.closed:
                        self._log_file_handle.write(f"{log_prefix}: {line_str}")
                        self._log_file_handle.flush()

                    if dest_stream:
                        dest_stream.write(line_bytes)
                        dest_stream.flush()
        except Exception as e:
            try:
                if self._log_file_handle and not self._log_file_handle.closed:
                    log_msg = f"!!! {log_prefix} 資料流轉發錯誤: {e}\n"
                    self._log_file_handle.write(log_msg)
                    self._log_file_handle.flush()
            except:
                pass

    def _forward_stdin(self, src_stream, dest_stream, log_prefix):
        """
        專門處理 stdin 轉發的函數，避免在子進程結束後繼續阻塞
        """
        import select
        try:
            while True:
                # 檢查子進程是否還在運行
                if self.process.poll() is not None:
                    # 子進程已結束，關閉 stdin 並退出
                    break
                
                # 使用 select 檢查是否有輸入可讀（僅在 Unix 系統上）
                if hasattr(select, 'select'):
                    ready, _, _ = select.select([src_stream], [], [], 0.1)
                    if not ready:
                        continue
                
                line_bytes = src_stream.readline()
                if not line_bytes:
                    break
                
                try:
                    line_str = line_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    line_str = f"[非UTF-8資料, {len(line_bytes)} bytes]\n"

                if self._log_file_handle and not self._log_file_handle.closed:
                    self._log_file_handle.write(f"{log_prefix}: {line_str}")
                    self._log_file_handle.flush()

                if dest_stream and not dest_stream.closed:
                    dest_stream.write(line_bytes)
                    dest_stream.flush()
                    
        except Exception as e:
            try:
                if self._log_file_handle and not self._log_file_handle.closed:
                    log_msg = f"!!! {log_prefix} 資料流轉發錯誤: {e}\n"
                    self._log_file_handle.write(log_msg)
                    self._log_file_handle.flush()
            except:
                pass
        finally:
            # 關閉子進程的 stdin
            if self.process and self.process.stdin and not self.process.stdin.closed:
                try:
                    self.process.stdin.close()
                    if self._log_file_handle and not self._log_file_handle.closed:
                        self._log_file_handle.write("--- 輸入流已關閉 ---\n")
                        self._log_file_handle.flush()
                except:
                    pass

    def run(self):
        """
        執行命令並開始記錄。
        返回子程序的退出碼。
        """
        try:
            # 以附加模式打開日誌檔案，因為它已在 main 函數中被清空
            self._log_file_handle = open(self.log_file, 'a', encoding='utf-8')
            
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )

            # 創建並啟動 I/O 轉發線程
            stdin_thread = threading.Thread(
                target=self._forward_stream,
                args=(sys.stdin.buffer, self.process.stdin, "輸入"),
                daemon=False
            )
            stdout_thread = threading.Thread(
                target=self._forward_stream,
                args=(self.process.stdout, sys.stdout.buffer, "輸出"),
                daemon=False
            )
            stderr_thread = threading.Thread(
                target=self._forward_stream,
                args=(self.process.stderr, sys.stderr.buffer, "錯誤"),
                daemon=False
            )

            self._threads = [stdin_thread, stdout_thread, stderr_thread]
            for t in self._threads:
                t.start()
            
            # 等待子進程結束
            self.exit_code = self.process.wait()

            # 等待所有線程完成
            # stdout 和 stderr 線程會在子進程結束後自然結束
            # stdin 線程需要特殊處理，因為它可能還在等待用戶輸入
            for t in self._threads:
                if t.is_alive():
                    t.join(timeout=2.0)  # 增加超時時間，給線程更多時間完成
                    if t.is_alive():
                        # 如果線程仍然活著（通常是 stdin 線程），說明它可能被阻塞
                        # 由於我們已經關閉了子進程，這些線程應該很快結束
                        pass

        except FileNotFoundError:
            print(f"錯誤: 命令 '{self.command[0]}' 未找到。", file=sys.stderr)
            self.exit_code = 127
        except Exception as e:
            print(f"啟動代理時發生錯誤: {e}", file=sys.stderr)
            if self._log_file_handle and not self._log_file_handle.closed:
                try:
                    self._log_file_handle.write(f"!!! 主程序錯誤: {e}\n")
                except: pass
            self.exit_code = 1
        finally:
            self._cleanup()
        
        return self.exit_code

    def _cleanup(self):
        """清理資源，如終止進程和關閉檔案。"""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
                if self.process.poll() is None:
                    self.process.kill()
            except:
                pass
        
        if self._log_file_handle and not self._log_file_handle.closed:
            self._log_file_handle.close()

def main():
    """程序入口點，處理命令行參數並執行 CommandLogger。"""
    parser = argparse.ArgumentParser(
        description="執行一個命令，並將其標準輸入、輸出和錯誤記錄到檔案中。"
    )
    parser.add_argument(
        '--log-path',
        help='指定日誌檔案的路徑。如果未提供，將在腳本目錄下創建 io_log.log。'
    )
    
    # 解析腳本自身的參數，其餘未識別的參數將被視為要執行的命令
    args, command_to_run = parser.parse_known_args()

    # 決定日誌檔案的路徑
    if args.log_path:
        log_file_path = args.log_path
    else:
        # 預設行為：在腳本檔案所在的目錄下創建日誌
        try:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            log_file_path = os.path.join(script_dir, "io_log.log")
        except NameError:
            # 如果在某些非標準環境中 __file__ 未定義，則退回到目前目錄
            log_file_path = "io_log.log"

    # 檢查是否提供了要執行的命令
    if not command_to_run:
        parser.print_help(sys.stderr)
        print("\n錯誤：未提供要執行的命令。", file=sys.stderr)
        print("用法示例: python3 your_script.py --log-path /tmp/my.log <command> [args...]", file=sys.stderr)
        return 1

    # 在啟動前，清空日誌檔案並確保目錄存在
    try:
        log_dir = os.path.dirname(os.path.abspath(log_file_path))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_file_path, 'w', encoding='utf-8') as f:
            pass # 僅用於清空檔案
    except IOError as e:
        print(f"錯誤：無法寫入日誌檔案 {log_file_path}: {e}", file=sys.stderr)
        return 1

    # 創建實例並執行
    logger = CommandLogger(command=command_to_run, log_file=log_file_path)
    return logger.run()

if __name__ == '__main__':
    sys.exit(main())
