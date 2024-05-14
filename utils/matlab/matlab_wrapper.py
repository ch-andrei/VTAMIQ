from utils.misc.logging import log_warn

try:
    import matlab
    import matlab.engine
except ImportError:
    log_warn("MATLAB is missing. Running MATLAB implementations will result in errors.")

import numpy as np
import scipy.io
import io, os, uuid

matlab_eng = None
matlab_io_out = None
matlab_io_err = None

matlab_wrapper_temp_folder = "./utils/matlab/matlab_temp_folder"


def init_instance_matlab():
    """
    This function must be called at least once before using any Matlab function
    :return:
    """

    if not os.path.exists(matlab_wrapper_temp_folder):
        os.makedirs(matlab_wrapper_temp_folder)

    global matlab_eng, matlab_io_out, matlab_io_err

    # configure Matlab instance
    matlab_eng = matlab.engine.start_matlab()
    matlab_io_out = io.StringIO()
    matlab_io_err = io.StringIO()

    cwd = os.getcwd()  # assumes ".../iqa-tool" (or ".../{repo_name}") as runtime entry point
    matlab_eng.addpath(r'{}\utils\matlab'.format(cwd), nargout=0)

    print("Initialized Matlab instance (wrapper).")


def quit_instance_matlab():
    global matlab_eng

    if matlab_eng is not None:
        matlab_eng.quit()


def get_io_streams():
    global matlab_io_out, matlab_io_err
    return matlab_io_out, matlab_io_err


def get_matlab_instance():
    global matlab_eng

    if matlab_eng is None:
        init_instance_matlab()

    return matlab_eng


def imgs_to_unique_mat(img1, img2=None, num_chars=20, dict_tag="tag", extra_identifier="id"):
    uuid_str = str(uuid.uuid4())
    random_inds = np.array(np.random.rand(num_chars) * len(uuid_str), np.uint8)
    final_str = ''.join([uuid_str[i] for i in random_inds])

    path1 = matlab_wrapper_temp_folder + "/" + extra_identifier + "-" + final_str + "_1.mat"
    path2 = matlab_wrapper_temp_folder + "/" + extra_identifier + "-" + final_str + "_2.mat"

    scipy.io.savemat(path1, mdict={dict_tag: img1})
    # image 2 is not always provided
    if img2 is not None:
        scipy.io.savemat(path2, mdict={dict_tag: img2})

    return (path1, path2, dict_tag) if img2 is not None else (path1, dict_tag)


def mat_to_img(path1, tag):
    return scipy.io.loadmat(path1)[tag]


def remove_matlab_unique_mat(path1, path2=None):
    os.remove(path1)
    if path2 is not None:
        os.remove(path2)
