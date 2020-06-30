from util.path_abstract import PathAbstract


class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return '/home/xinyi/Dataset/SYN/DATA/mtp'

    @staticmethod
    def save_root_dir():
        return './model'

    @staticmethod
    def models_dir():
        return "./model"
