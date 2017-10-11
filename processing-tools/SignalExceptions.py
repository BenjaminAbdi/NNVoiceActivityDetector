class NotAvalidWavFileException(Exception):
    def __init__(self, filename):
        self.message = "\nLe nom de fichier \"{}\" est soit erroné, soit le fichier associé n'est pas un fichier wav valide.".format(filename)
        super().__init__(self.message)
