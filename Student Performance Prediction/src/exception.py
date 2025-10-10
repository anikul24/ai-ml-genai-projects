import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = f"Error occurred in script: {file_name} at line number: {line_number} error message: {str(error)}"
    else:
        error_message = f"Error: {str(error)} (no traceback available)"
    return error_message


class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        super().__init__(error)
        self.error = error
        self.error_detail = error_detail

    def __str__(self):
        return error_message_detail(self.error, self.error_detail)