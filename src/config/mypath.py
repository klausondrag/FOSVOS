from config.path_abstract import PathAbstract


class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return '/Users/lealtaixe/Datasets/DAVIS/'

    @staticmethod
    def save_root_dir():
        return './'

    @staticmethod
    def exp_dir():
        return './'

    @staticmethod
    def is_custom_pytorch():
        return False

    @staticmethod
    def custom_pytorch():
        return None

    @staticmethod
    def is_custom_opencv():
        return False

    @staticmethod
    def custom_opencv():
        return None

    @staticmethod
    def models_dir():
        return "./models"
