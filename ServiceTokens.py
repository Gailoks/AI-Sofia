"""Module with service tokens for nn module"""
SERVICE_INPUT_SIZE = 0
SERVICE_OUTPUT_SIZE = 1

STO_SWITCH = 0

class ServiceTokens:
    """Provides service tokens for nn module"""
    def __init__(self, vocabulary_size: int):
        self.__vocabulary_size = vocabulary_size

    def get(self, service_index: int) -> int:
        """Gets service token by its index"""
        return self.__vocabulary_size + service_index

    def check(self, service_token: int, service_index: int) -> bool:
        """Checks if given token is associated with given index"""
        return self.get(service_index) == service_token
