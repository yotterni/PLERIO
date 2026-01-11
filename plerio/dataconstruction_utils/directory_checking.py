import os
from pathlib import Path


class DirectoryChecker:
    """
    A class that creates needed directories. If the directory exists,
    will try to delete it first and create a new one after that.
    """
    def __init__(self, keep_existing: bool = False) -> None:
        """
        :param keep_existing: if `True` and if the directory exists,
        the class will keep the directory instead of deleting it and
        recreating after that.
        """
        self.keep_existing = keep_existing

    def handle(self, path: str | Path):
        """
        Hihg-level API to check if a directory exists
        :param path:
        :return:
        """
        path = Path(path)

        if self.keep_existing:
            if not path.exists():
                self.create_dir_(path)
        else:
            self.delete_dir_(path)
            self.create_dir_(path)

    @staticmethod
    def delete_dir_(path: Path) -> None:
        """
        Deletes a directory if it exists.
        :param path: path to delete
        :return: `None`
        """
        try:
            os.rmdir(path)
        except (FileNotFoundError, OSError):
            pass
            # TODO: replace path with smth meaningful, maybe logging

    @staticmethod
    def create_dir_(path: Path) -> None:
        """
        Recursively creates a directory if it doesn't exist.
        E.g., if the 'a/b/c' is passed and none of directories
        of this path exist, then they all will be created.
        :param path: needed path to create.
        :return: `None`.
        """
        path.mkdir(parents=True, exist_ok=True)

    def __str__(self):
        return f'DirectoryChecker(keep_existing={self.keep_existing})'

    def __repr__(self):
        return self.__str__()
