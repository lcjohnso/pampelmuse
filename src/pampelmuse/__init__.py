__version__ = "1.3.1"


# define the exception that is raised when the parameters in the PampelMuse-configuration are such that they do not
# allow for a proper analysis of the provided input
class ConfigurationError(EnvironmentError):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)