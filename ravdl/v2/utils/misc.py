import json
import urllib.request
from pip._internal.operations.freeze import freeze

def isLatestVersion(pkgName):
    # Get the currently installed version
    current_version = ''
    for requirement in freeze(local_only=False):
        pkg = requirement.split('==')
        if pkg[0] == pkgName:
            current_version = pkg[1]

    # Check pypi for the latest version number
    contents = urllib.request.urlopen('https://pypi.org/pypi/'+pkgName+'/json').read()
    data = json.loads(contents)
    latest_version = data['info']['version']

    # print('Current version of '+pkgName+' is '+current_version)
    # print('Latest version of '+pkgName+' is '+latest_version)

    return latest_version == current_version
