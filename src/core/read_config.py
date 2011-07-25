import asciidata



"""
Read keywords from an ASCII configuration file (CF).
CF structure: KEY  <value>  ## comment\n
UPDATED 2010/02/02: Code can now properly handle left-out parameters and
                    empty lines
"""
class ReadConfig:

    def __init__(self, config_file):
        self.data = open(config_file)

    
    """
    Method reads all the keywords defined in the CF and writes
    them into dictionary <instance>.keys
    """
    def read_keys(self, verbose=0):
        params = []
        for line in self.data:
            params.append(line.split("##")[0])

        self.keys = {}
        for param in params:
            candidate = param.split()
            if len(candidate) == 2:
                key = candidate[0]
                value = candidate[1]
                try:
                    self.keys[key] = float(value)
                except ValueError:
                    self.keys[key] = value
                if verbose > 0:
                    print "\t"+key+"\t"+str(self.keys[key])


    """
    Method to call a specific keyword from the dictionary.
    Note that either a string or a float (where possible) is returned.
    """
    def call_key(self, key, default='xxx'):
        try:
            value = self.keys[key]
        except KeyError:
            if default != 'xxx':
                value = default
            else:
                raise KeyError, "Parameter '%s' is not defined!" %key
        return value
