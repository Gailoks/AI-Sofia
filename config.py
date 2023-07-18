"""Provides functional to work with application configuration"""
from __future__ import annotations
from typing import TypeVar, Type, Any, Iterable
import json

T = TypeVar('T')


def _extract_local_keys(dictionay: dict[str, Any]) -> Iterable[str]:
    return filter(lambda x: not isinstance(dictionay[x], dict), dictionay)


class ConfigurationScope:
    """Represents application configuration scope"""
    def __init__(self, root_dictionary: dict[str, Any]):
        self.__root_dictionary = root_dictionary

        global_scope_keys = _extract_local_keys(self.__root_dictionary)
        self.__global_scope: dict[str, Any] = {}
        for key in global_scope_keys:
            self.__global_scope[key] = self.__root_dictionary[key]

    def get(self, path: str) -> dict[str, Any]:
        """Gets accumulated configuration by path"""
        accumulated, _ = self.__get_internal(path)
        return accumulated

    def scope(self, path: str) -> ConfigurationScope:
        """Creates new scope by path"""
        accumulated, scope_dict = self.__get_internal(path)
        for pair in accumulated.items():
            scope_dict[pair[0]] = pair[1]

        return ConfigurationScope(scope_dict)

    def resolve(self, type_d: Type[T], path: str) -> T:
        """Creates new instance of T using configuration"""
        return type_d(**self.get(path))

    def this(self) -> dict[str, Any]:
        """Gets global configuration of scope"""
        return self.__global_scope

    def __get_internal(self, path: str):
        result = self.__global_scope.copy()
        path_segments = path.split(':')

        current_scope: dict[str, Any] = self.__root_dictionary

        for path_segment in path_segments:
            current_scope = current_scope[path_segment]
            current_scope_keys = _extract_local_keys(current_scope)

            for key in current_scope_keys:
                result[key] = current_scope[key]

        return result, current_scope


class Configuration(ConfigurationScope):
    """Represents application configuration"""
    def __init__(self, json_content: str):
        super().__init__(json.loads(json_content))


def config() -> Configuration:
    """Load Configuration form default config path"""
    with open("config.json", 'r', encoding="utf-8") as config_file:
        return Configuration(config_file.read())
    