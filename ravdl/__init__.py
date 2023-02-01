from .utils.misc import isLatestVersion
import sys

print('Checking version of RavDL...')
if not isLatestVersion('ravdl'):
    print('Please update RavDL to the latest version.')
    sys.exit(1)
