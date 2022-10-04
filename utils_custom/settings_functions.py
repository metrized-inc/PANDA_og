import hjson
import os
import torchvision.transforms as T
import wandb


def create_directory(path):
    """
    Create a directory if it does not exist yet
    path: path to directory
    """
    if not os.path.exists(os.path.dirname(path)):
        # Create the dir
        os.makedirs(os.path.dirname(path))

def dict2json(fp, d):
    """
    Write a dict to an .hjson file
    fp: filename
    d: the dict
    """
    with open(fp, "w") as json_file:
        # get json text
        text_formatted = hjson.dumps(d, indent=4, sort_keys=True)
        json_file.write(text_formatted)

def json2dict(fp):
    """
    Load an .hjson into a dict
    fp: filename
    """
    with open(fp, "r") as f:
        d = hjson.load(f)
    return d

def load_settings(fp):
    """
    Return a dictionary from an existing .hjson file if it exists
    otherwise return an empty dictionary
    """
    if os.path.exists(fp):
        return json2dict(fp)
    else:
        print("Error: {} does not exist".format(fp))
        return dict()

def clean_tf_arguments(s):
    """
    Remove string elements that cause transform definition errors
    """
    # Remove interpolation bilinear
    s = s.replace("interpolation=bilinear", "")
    return s


def tf2str(tf_list):
    """
    Change a torchvision transform into a string for portability
    """

    tf_str = ""

    for tf in tf_list:
        json_str = "\nT." + str(tf) + ","
        json_str = clean_tf_arguments(json_str)
        tf_str += json_str

    # Wrap in compose
    tf_compose = "T.Compose([" + tf_str + "])"
    return tf_compose


def str2transform(tf_string):
    """
    Get a torchvision transform object from a string representation
    """
    if tf_string:
        return eval(tf_string)
    else:
        return None

"""
Uploads the settings file to wandb
log_dir: the directory to save to (pass wandb.run.dir to log to wandb)
settings_path: the path to the settings file (to get filename)
settings: the settings dictionary object
"""
def logSettingsFile(log_dir, settings_path, settings):
    settings_wandb = os.path.join(log_dir, os.path.split(settings_path)[1])
    dict2json(settings_wandb, settings)

"""
Downloads and uses the artifact in the run, returning the local directory path to the artifact
artifact_name: API of the artifact to download from wandb
run: wandb run that called this function or None if not using a wandb run
"""
def getArtifact(artifact_name, run):
    # download artifact outside of a run
    if run is None:
        api = wandb.Api()
        artifact = api.artifact(artifact_name)
        directory = artifact.checkout()
    # download artifact inside of a run
    else:
        artifact = run.use_artifact(artifact_name)
        directory = artifact.download()

    return directory

"""
Returns a local directory given a local directory path or wandb artifact API
dir_path: the local directory path or wandb artifact API
run: wandb run that called this function or None if not using a wandb run
"""
def getDataDirectory(dir_path, run=None):
    return_dir = dir_path
    try:
        # CHeck null first
        if dir_path:
            if os.path.exists(dir_path):
                return return_dir
            else:
                return_dir = getArtifact(dir_path, run)
                return return_dir
    except:
        raise Exception('{} is not a local directory or wandb artifact'.format(dir_path))