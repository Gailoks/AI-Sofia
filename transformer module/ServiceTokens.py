SERVICE_INPUT_SIZE = 0
SERVICE_OUTPUT_SIZE = 1

STO_SWITCH = 0

class ServiceTokens:
    def __init__(self, vocabulary_size:int):
        self.__vocabulary_size = vocabulary_size

    def get(self, service_index:int):
        return self.__vocabulary_size + service_index