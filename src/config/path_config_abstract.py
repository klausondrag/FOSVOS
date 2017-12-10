from pathlib import Path


class PathConfigAbstract(object):
    @staticmethod
    def db_root_dir():
        raise NotImplementedError

    @staticmethod
    def save_root_dir() -> Path:
        return Path('models')

    @staticmethod
    def exp_dir():
        raise NotImplementedError

    @staticmethod
    def is_custom_pytorch():
        raise NotImplementedError

    @staticmethod
    def custom_pytorch():
        raise NotImplementedError

    @staticmethod
    def is_custom_opencv():
        raise NotImplementedError

    @staticmethod
    def custom_opencv():
        raise NotImplementedError

    @staticmethod
    def models_dir():
        raise NotImplementedError
        # @staticmethod
        # def matlab_code():
        #    raise NotImplementedError

    @staticmethod
    def _get_model_file_name(model_name, epoch: int) -> str:
        return '{}_epoch-{}.pth'.format(model_name, str(epoch))

    @classmethod
    def _get_parent_folder(cls) -> Path:
        return cls.save_root_dir() / 'parent'

    @classmethod
    def get_parent_file(cls, model_name: str, epoch: int) -> Path:
        return cls._get_parent_folder() / PathConfigAbstract._get_model_file_name(model_name, epoch)

    @classmethod
    def _get_online_folder(cls) -> Path:
        return cls.save_root_dir() / 'online'

    @classmethod
    def get_online_file(cls, seq_name: str, epoch: int) -> Path:
        return cls._get_online_folder() / PathConfigAbstract._get_model_file_name(seq_name, epoch)
