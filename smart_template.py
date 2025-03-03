import glob
import os
import pickle
import tarfile
import warnings
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import wget
import yaml
from astropy import constants
from astropy.io import fits
from astropy.table import Table
from scipy.constants import c
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.optimize import curve_fit
from tqdm import tqdm
import sys

# user-defined parameters for the code
params = dict()

# -- DO NOT EDIT THE LINE BELOW --
# --- THIS IS DONE WITHIN THE CODE ---
params['working_directory'] = '/Users/eartigau/gaucho' # UPDATED BY THE CODE

if not os.path.exists(params['working_directory']):
    print('\n\nYou are running the code for the first time')
    print('\nPlease enter the path to the working directory.')
    print('\t\tIf you plan to put your data for target "GL1234"')
    print('\t\tin the folder :   /Users/yourself/smart_template/GL1234')
    print('\t\tthen here you should enter :  /Users/yourself/smart_template')
    working_dir = input('\nEnter a valid path: ')

    if len(working_dir) == 0:
        raise FileNotFoundError('The path you entered is not valid.')

    if not os.path.exists(working_dir):
        raise FileNotFoundError('The path you entered is not valid.')

    dir = os.path.realpath(working_dir)
    print('\t We will modify {}'.format(__file__))
    lines = []
    print(__file__)
    with open(__file__, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if len(lines[i]) != 0:
            if lines[i].startswith("params['working_directory']"):
                lines[i] = "params['working_directory'] = '{}' # UPDATED BY THE CODE\n".format(working_dir)

    with open(__file__, 'w') as f:
        f.writelines(lines)

    print('\n\nThe working directory has been set to:\n\t\t\t ** {} **'.format(dir))
    print('\nThere is no need to restart the code, but you will not be asked again')
    print('for a working directory.\n\n')
    params['working_directory'] = dir


def rot_broad(wvl: np.ndarray, flux: np.ndarray, epsilon: float, vsini: float,
              eff_wvl: Optional[float] = None) -> np.ndarray:
    """
    **********************************************************************
    ***** THIS FUNCTION IS COPIED FROM PyAstronomy/pyasl/rotBroad.py *****
    ***** AND MODIFIED TO USE THE SAME CONVENTIONS AS THE LBL CODE.  *****
    ***** AND AVOID DEPENDENCIES ON OTHER PyAstronomy MODULES.       *****
    **********************************************************************

    Apply rotational broadening using a single broadening kernel.
    The effect of rotational broadening on the spectrum is
    wavelength dependent, because the Doppler shift depends
    on wavelength. This function neglects this dependence, which
    is weak if the wavelength range is not too large.
    .. note:: numpy.convolve is used to carry out the convolution
              and "mode = same" is used. Therefore, the output
              will be of the same size as the input, but it
              will show edge effects.
    Parameters
    ----------
    wvl : array
        The wavelength
    flux : array
        The flux
    epsilon : float
        Linear limb-darkening coefficient
    vsini : float
        Projected rotational velocity in km/s.
    eff_wvl : float, optional
        The wavelength at which the broadening
        kernel is evaluated. If not specified,
        the mean wavelength of the input will be
        used.
    Returns
    -------
    Broadened spectrum : array
        The rotationally broadened output spectrum.
    """
    # Wavelength binsize
    dwl = wvl[1] - wvl[0]
    # deal with no effective wavelength
    if eff_wvl is None:
        eff_wvl = np.mean(wvl)
    # The number of bins needed to create the broadening kernel
    binn_half = int(np.floor(((vsini / (c / 1000)) * eff_wvl / dwl))) + 1
    gwvl = (np.arange(4 * binn_half) - 2 * binn_half) * dwl + eff_wvl
    # Create the broadening kernel
    dl = gwvl - eff_wvl
    # -------------------------------------------------------------------------
    # this bit is from _Gdl.gdl
    #    Calculates the broadening profile.
    # -------------------------------------------------------------------------
    # set vc
    vc = vsini / (c / 1000)
    # set eps (make sure it is a float)
    eps = float(epsilon)
    # calculate the max vc
    dlmax = vc * eff_wvl
    # generate the c1 and c2 parameters
    c1 = 2 * (1 - eps) / (np.pi * dlmax * (1 - eps / 3))
    c2 = eps / (2 * dlmax * (1 - eps / 3))
    # storage for the output
    bprof = np.zeros(len(dl))
    # Calculate the broadening profile
    xvec = dl / dlmax
    indi0 = np.where(np.abs(xvec) < 1.0)[0]
    bprof[indi0] = c1 * np.sqrt(1 - xvec[indi0] ** 2) + c2 * (1 - xvec[indi0] ** 2)
    # Correct the normalization for numeric accuracy
    # The integral of the function is normalized, however, especially in the
    # case of mild broadening (compared to the wavelength resolution), the
    # discrete  broadening profile may no longer be normalized, which leads to
    # a shift of the output spectrum, if not accounted for.
    bprof /= (np.sum(bprof) * dwl)
    # -------------------------------------------------------------------------
    # Remove the zero entries
    indi = np.where(bprof > 0.0)[0]
    bprof = bprof[indi]
    # -------------------------------------------------------------------------
    result = np.convolve(flux, bprof, mode="same") * dwl
    return result


def linear_minimization(vector, sample, mm, v, sz_sample, case, recon, amps):
    # raise ValueError(emsg.format(func_name))
    # ​
    # vector of N elements
    # sample: matrix N * M each M column is adjusted in amplitude to minimize
    # the chi2 according to the input vector
    # output: vector of length M gives the amplitude of each column
    #
    if case == 1:
        # fill-in the co-variance matrix
        for i in range(sz_sample[0]):
            for j in range(i, sz_sample[0]):
                mm[i, j] = np.sum(sample[i, :] * sample[j, :])
                # we know the matrix is symetric, we fill the other half
                # of the diagonal directly
                mm[j, i] = mm[i, j]
            # dot-product of vector with sample columns
            v[i] = np.sum(vector * sample[i, :])
        # if the matrix cannot we inverted because the determinant is zero,
        # then we return a NaN for all outputs
        if np.linalg.det(mm) == 0:
            amps = np.zeros(sz_sample[0]) + np.nan
            recon = np.zeros_like(v)
            return amps, recon

        # invert coveriance matrix
        inv = np.linalg.inv(mm)
        # retrieve amplitudes
        for i in range(len(v)):
            for j in range(len(v)):
                amps[i] += inv[i, j] * v[j]

        # reconstruction of the best-fit from the input sample and derived
        # amplitudes
        for i in range(sz_sample[0]):
            recon += amps[i] * sample[i, :]
        return amps, recon

    if case == 2:
        # same as for case 1 but with axis flipped
        for i in range(sz_sample[1]):
            for j in range(i, sz_sample[1]):
                mm[i, j] = np.sum(sample[:, i] * sample[:, j])
                mm[j, i] = mm[i, j]
            v[i] = np.sum(vector * sample[:, i])

        if np.linalg.det(mm) == 0:
            return amps, recon

        inv = np.linalg.inv(mm)
        for i in range(len(v)):
            for j in range(len(v)):
                amps[i] += inv[i, j] * v[j]

        for i in range(sz_sample[1]):
            recon += amps[i] * sample[:, i]
        return amps, recon


def color(message, color):
    COLOURS = dict()
    COLOURS['BLACK'] = '\033[90;1m'
    COLOURS['RED'] = '\033[1;91;1m'
    COLOURS['GREEN'] = '\033[92;1m'
    COLOURS['YELLOW'] = '\033[1;93;1m'
    COLOURS['BLUE'] = '\033[94;1m'
    COLOURS['MAGENTA'] = '\033[1;95;1m'
    COLOURS['CYAN'] = '\033[1;96;1m'
    COLOURS['WHITE'] = '\033[97;1m'
    COLOURS['ENDC'] = '\033[0;0m'

    return COLOURS[color.upper()] + message + COLOURS['ENDC']


def printc(message, msg_type='', print_time=True):
    """
    Print a message with a color
    :param message:
    :param msg_type:
        -> info = green
        -> bad1 = yellow
        -> bad2 = red
        -> bad3 = magenta
        -> number = blue
        -> (other) = white
    :param print_time:
    :return: nothing
    """

    msg_color = "black"

    if msg_type == 'info':
        msg_color = 'green'

    if msg_type == 'bad1':
        msg_color = 'cyan'

    if msg_type == 'bad2':
        msg_color = 'red'

    if msg_type == 'bad3':
        msg_color = 'magenta'

    if msg_type == 'number':
        msg_color = 'blue'

    if print_time:
        time = datetime.now().strftime('%H:%M:%S.%f')[:-4] + '│'
    else:
        time = ''

    if len(message) == 1:
        # get terminal width
        try:
            w = os.get_terminal_size().columns - len(time)
        except:
            w = 80 - len(time)
        message = message[0] * w

    print(color(time + message, msg_color))


def save_pickle(filename, variable):
    with open(filename, 'wb') as handle:
        pickle.dump(variable, handle)


def read_pickle(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b


def art(word, color1='MAGENTA', color2='red'):
    letter = \
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z']
    length = \
        [3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4, 3, 4, 3, 3]

    low1 = "┌─┐┌┐ ┌─┐┌┬┐┌─┐┌─┐┌─┐┬ ┬┬  ┬┬┌─┬  ┌┬┐┌┐┌┌─┐┌─┐┌─┐ ┬─┐┌─┐┌┬┐┬ ┬┬  ┬┬ ┬─┐ ┬┬ ┬┌─┐"
    low2 = "├─┤├┴┐│   ││├┤ ├┤ │ ┬├─┤│  │├┴┐│  │││││││ │├─┘│─┼┐├┬┘└─┐ │ │ │└┐┌┘│││┌┴┬┘└┬┘┌─┘"
    low3 = "┴ ┴└─┘└─┘─┴┘└─┘└  └─┘┴ ┴┴└─┘┴ ┴┴─┘┴ ┴┘└┘└─┘┴  └─┘└┴└─└─┘ ┴ └─┘ └┘ └┴┘┴ └─ ┴ └─┘"
    up1 = "╔═╗╔╗ ╔═╗╔╦╗╔═╗╔═╗╔═╗╦ ╦╦  ╦╦╔═╦  ╔╦╗╔╗╔╔═╗╔═╗╔═╗ ╦═╗╔═╗╔╦╗╦ ╦╦  ╦╦ ╦═╗ ╦╦ ╦╔═╗"
    up2 = "╠═╣╠╩╗║   ║║║╣ ╠╣ ║ ╦╠═╣║  ║╠╩╗║  ║║║║║║║ ║╠═╝║═╬╗╠╦╝╚═╗ ║ ║ ║╚╗╔╝║║║╔╩╦╝╚╦╝╔═╝"
    up3 = "╩ ╩╚═╝╚═╝═╩╝╚═╝╚  ╚═╝╩ ╩╩╚═╝╩ ╩╩═╝╩ ╩╝╚╝╚═╝╩  ╚═╝╚╩╚═╚═╝ ╩ ╚═╝ ╚╝ ╚╩╝╩ ╚═ ╩ ╚═╝"

    letter = letter + ['-', ' ', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', '?', '!']
    length = length + [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    low1 = low1 + "         ┌─┐ ┐ ┌─┐┌─┐┌ ┐┌─┐┌─┐┌─┐┌─┐┌─┐ ┌  ┐ ┌─┐ ┐ "
    low2 = low2 + "───      │ │ │ ┌─┘ ─┤└─┤└─┐├─┐  │├─┤└─┤ │  │  ┌┘ │ "
    low3 = low3 + "      ·  └─┘─┴─└─┘└─┘  ┘└─┘└─┘  ┴└─┘└─┘ └  ┘  o  o "

    low_1 = ""
    low_2 = ""
    low_3 = ""

    letter = np.array([l.lower() for l in letter])

    l1 = np.array([(np.cumsum(length))[l.lower() == letter][0] for l in word])
    l2 = np.array([np.array(length)[l.lower() == letter][0] for l in word])
    l0 = l1 - l2

    for i in range(len(l1)):
        if word[i] == word[i].lower():
            low_1 += low1[l0[i]:l1[i]]
            low_2 += low2[l0[i]:l1[i]]
            low_3 += low3[l0[i]:l1[i]]
        else:
            low_1 += up1[l0[i]:l1[i]]
            low_2 += up2[l0[i]:l1[i]]
            low_3 += up3[l0[i]:l1[i]]

    low_0 = color('╔' + '═' * (len(low_1) + 2) + '╗', color1)
    low_4 = color('╚' + '═' * (len(low_1) + 2) + '╝', color1)

    low_1 = color('║ ', color1) + color(low_1, color2) + color(' ║', color1)
    low_2 = color('║ ', color1) + color(low_2, color2) + color(' ║', color1)
    low_3 = color('║ ', color1) + color(low_3, color2) + color(' ║', color1)

    try:
        w = os.get_terminal_size().columns
    except OSError:
        w = 80
    dw = (w - len(low_1) // 2) // 2
    low_0 = ' ' * dw + low_0
    low_1 = ' ' * dw + low_1
    low_2 = ' ' * dw + low_2
    low_3 = ' ' * dw + low_3
    low_4 = ' ' * dw + low_4

    return '\n' + low_0 + '\n' + low_1 + '\n' + low_2 + '\n' + low_3 + '\n' + low_4 + '\n'


def lowpassfilter(input_vect, width=101):
    # Computes a low-pass filter of an input vector. This is done while properly handling
    # NaN values, but at the same time being reasonably fast.
    # Algorithm:
    #
    # provide an input vector of an arbitrary length and compute a running NaN median over a
    # box of a given length (width value). The running median is NOT computed at every pixel
    # but at steps of 1/4th of the width value. This provides a vector of points where
    # the nan-median has been computed (ymed) and mean position along the input vector (xmed)
    # of valid (non-NaN) pixels. This xmed/ymed combination is then used in a spline to
    # recover a vector for all pixel positions within the input vector.
    #
    # When there are no valid pixel in a 'width' domain, the value is skipped in the creation
    # of xmed and ymed, and the domain is splined over.

    # indices along input vector
    index = np.arange(len(input_vect))

    # placeholders for x and y position along vector
    xmed = []
    ymed = []

    # loop through the lenght of the input vector
    for i in np.arange(-width // 2, len(input_vect) + width // 2, width // 4):

        # if we are at the start or end of vector, we go 'off the edge' and
        # define a box that goes beyond it. It will lead to an effectively
        # smaller 'width' value, but will provide a consistent result at edges.
        low_bound = i
        high_bound = i + int(width)

        if low_bound < 0:
            low_bound = 0
        if high_bound > (len(input_vect) - 1):
            high_bound = (len(input_vect) - 1)

        pixval = index[low_bound:high_bound]

        if len(pixval) < 3:
            continue

        # if no finite value, skip
        if np.max(np.isfinite(input_vect[pixval])) == 0:
            continue

        # mean position along vector and NaN median value of
        # points at those positions
        xmed.append(np.nanmean(pixval))
        ymed.append(np.nanmedian(input_vect[pixval]))

    xmed = np.array(xmed, dtype=float)
    ymed = np.array(ymed, dtype=float)

    # we need at least 3 valid points to return a
    # low-passed vector.
    if len(xmed) < 3:
        return np.zeros_like(input_vect) + np.nan

    if len(xmed) != len(np.unique(xmed)):
        xmed2 = np.unique(xmed)
        ymed2 = np.zeros_like(xmed2)
        for i in range(len(xmed2)):
            ymed2[i] = np.mean(ymed[xmed == xmed2[i]])
        xmed = xmed2
        ymed = ymed2

    # splining the vector
    spline = ius(xmed, ymed, k=2, ext=3)
    lowpass = spline(np.arange(len(input_vect)))

    return lowpass


def snail(iter, desc='', leave=False):
    txt = [' ']
    for i in range(1000):
        txt.append('🐌')
    txt.append('_')
    # ,'🐌','🐌','🐌','🐌','🐌','🐌','🐌','🐌','_']
    return tqdm(iter, leave=leave, desc=desc,
                colour='green', ascii=txt)


def sigma(tmp):
    if type(tmp[0]) != np.float64:
        tmp = np.array(tmp, dtype='float64')
    # return a robust estimate of 1 sigma
    sig1 = 0.682689492137086
    p1 = (1 - (1 - sig1) / 2) * 100
    return (np.nanpercentile(tmp, p1) - np.nanpercentile(tmp, 100 - p1)) / 2.0


def doppler(wave, v):
    # velocity expressed in m/s
    # relativistic calculation

    v = np.array(v)
    wave = np.array(wave)
    return wave * np.sqrt((1 - v / constants.c.value) / (1 + v / constants.c.value))


def lin_mini(vector, sample):
    # wrapper function that sets everything for the @jit later
    # In particular, we avoid the np.zeros that are not handled
    # by numba

    # size of input vectors and sample to be adjusted
    sz_sample = sample.shape  # 1d vector of length N
    sz_vector = vector.shape  # 2d matrix that is N x M or M x N

    # define which way the sample is flipped relative to the input vector
    if sz_vector[0] == sz_sample[0]:
        case = 2
    elif sz_vector[0] == sz_sample[1]:
        case = 1
    else:
        emsg = ('Neither vector[0]==sample[0] nor vector[0]==sample[1] '
                '(function = {0})')
        print(emsg)
        raise ValueError(emsg.format(emsg))

    # we check if there are NaNs in the vector or the sample
    # if there are NaNs, we'll fit the rest of the domain
    isnan = (np.sum(np.isnan(vector)) != 0) or (np.sum(np.isnan(sample)) != 0)

    if case == 1:

        if isnan:
            # we create a mask of non-NaN
            keep = np.isfinite(vector) * np.isfinite(np.sum(sample, axis=0))
            # redefine the input vector to avoid NaNs
            vector = vector[keep]
            sample = sample[:, keep]

            sz_sample = sample.shape
            sz_vector = vector.shape

        # matrix of covariances
        mm = np.zeros([sz_sample[0], sz_sample[0]])
        # cross-terms of vector and columns of sample
        v = np.zeros(sz_sample[0])
        # reconstructed amplitudes
        amps = np.zeros(sz_sample[0])
        # reconstruted fit
        recon = np.zeros(sz_sample[1])

    if case == 2:
        # same as for case 1, but with axis flipped
        if isnan:
            # we create a mask of non-NaN
            keep = np.isfinite(vector) * np.isfinite(np.sum(sample, axis=1))
            vector = vector[keep]
            sample = sample[keep, :]

            sz_sample = sample.shape
            sz_vector = vector.shape

        mm = np.zeros([sz_sample[1], sz_sample[1]])
        v = np.zeros(sz_sample[1])
        amps = np.zeros(sz_sample[1])
        recon = np.zeros(sz_sample[0])

    # pass all variables and pre-formatted vectors to the @jit part of the code
    amp_out, recon_out = linear_minimization(vector, sample, mm, v, sz_sample, case,
                                             recon, amps)

    # if we had NaNs in the first place, we create a reconstructed vector
    # that has the same size as the input vector, but pad with NaNs values
    # for which we cannot derive a value
    if isnan:
        recon_out2 = np.zeros_like(keep) + np.nan
        recon_out2[keep] = recon_out
        recon_out = recon_out2

    return amp_out, recon_out


def load_yaml(file):
    """
    Load a yaml file

    :param file: string, name of the yaml file
    :return: dictionary
    """

    with open(file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return dict(data)


def save_yaml(file, data):
    """
    Save a dictionary to a yaml file

    :param file: string, name of the yaml file
    :param data: dictionary
    :return:
    """

    with open(file, 'w') as f:
        yaml.dump(data, f)


def get_from_ari():
    """
    Copy-paste the name of all files for the ARI interface "file_list" document.
    Press enter twice when done and files get copied to the current directory.
    Note that unless you have an ssh key, you will be prompted for your password
    for each file. This is a feature, not a bug. Look at this website for more
    information on how to set these keys: https://www.ssh.com/ssh/copy-id
    """

    printc('~', 'bad3')
    printc('This code will download files from the ARI interface.', 'bad3')
    printc('Copy-paste the name of all files. Press enter twice when done.\n', 'bad3')

    user_inputs = []
    user = ' '  # initialize user input
    i = 1  # initialize counter
    while user != '':
        user = input()
        user_inputs.append(user)
        i += 1

    # make a numpy array
    user_inputs = np.array(user_inputs)

    # remove entries that do not contain '.fits'
    valid = ['.fits' in user_input for user_input in user_inputs]
    user_inputs = user_inputs[valid]

    # remove entries that do not contain '.fits'
    valid = ['.fits' in user_input for user_input in user_inputs]
    user_inputs = user_inputs[valid]

    for i, user_input in enumerate(user_inputs):
        file_name = user_input.split('/')[-1]
        if os.path.exists(file_name):
            printc('File {} already exists. Skipping download.'.format(file_name), 'info')
            continue
        printc('Downloading [{}/{}] {}'.format(i + 1, len(user_inputs), user_input), 'info')
        cmd = 'scp -r spirou@maestria:{} .'.format(user_input)
        # printout for the user
        print(cmd)
        os.system(cmd)


def gauss(x, mu, sig, amp, zp):
    return zp + amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def mk_smart(folder, doplot=False, force=True):
    """
    This function will create a smart template for a given instrument. It will use the s1d files in the folder

    :param folder:
    :param doplot:
    :param force:

    *** How to use this code ***

    You need to create folders per target and put the s1d files in there. The folder name should be the DRSOBJN
    of the target. For example, if you want to create a smart template for the target HD 123456, you need to create
    a folder called HD123456 and put the s1d files in there. You then need to run the code and pass to mk_smart
    the path to the folder. For example, if you have a folder called /space/spirou/smart_template/HD123456, you
    need to run the code as follows:
         mk_smart('HD123456')

    You may  pass the argument doplot = True to see the results of the smart template. You may also pass force = True
    to force the code to recompute the smart template even if it already exists. This is useful if you want to
    recompute the smart template with a different set of s1d files.

    Some folders get created along the way:
        - per_file_info: contains a csv file per s1d file with the results of the smart template
        - reconstructed_templates_INSTRUMENT : contains the smart template for each target for a given instrument
        - templates_INSTRUMENT: contains the vetted stars used to create the smart template, this gets downloaded
        from the LBL website


    :return:
    """

    printc('We change folder to {}'.format(params['working_directory']))
    initial_folder = os.getcwd()
    printc('... we will go back to {} at the end of the code'.format(initial_folder))
    os.chdir(params['working_directory'])
    search_string = '{}/{}/*_pp_s1d_v_*.fits'.format(params['working_directory'], folder)
    printc('Search string : {}'.format(search_string))
    files = np.array(glob.glob(search_string))
    files = files[np.argsort(np.random.random(len(files)))]
    
    if len(files) == 0:
        print(art('grrrr... bad bad bad!', color1='red', color2='red'))
        printc('No files found in {}'.format(folder), 'bad3')

        return

    if not os.path.isdir('per_file_info/'):
        os.mkdir('per_file_info/')

    # read first s1d file to get the wavelength grid. Note that s1d wave grids are instrument-specific and will
    # therefore not change through the different s1d files
    tbl0 = Table.read(files[0], 1)
    h = fits.getheader(files[0])

    if 'nirps' in h['INSTRUME'].lower():
        instrument = 'nirps'
    elif 'spirou' in h['INSTRUME'].lower():
        instrument = 'spirou'
    else:
        printc('Instrument {} not supported'.format(h['INSTRUME']), 'bad3')
        return

    instrument = instrument.lower()

    if instrument not in ['nirps', 'spirou']:
        printc('Instrument {} not supported'.format(instrument), 'bad3')
        return

    print(art('Smart Template!'))

    # Define all relevant parameters for the smart template

    hpsize = 2001
    # domain for adjustement of the template
    wave_domain = (1450, 1750)

    outname = '{}/smart_templates_{}.tar'.format(params['working_directory'], instrument)
    if not os.path.exists(outname):
        url = 'http://www.astro.umontreal.ca/~artigau/lbl/smart_templates_{}.tar'.format(instrument)

        wget.download(url, outname)
        file = tarfile.open(outname)
        # extracting file
        file.extractall(params['working_directory'])
        file.close()

    tbl_template = Table.read('{}/templates_{}/template_list.csv'.format(params['working_directory'], instrument),
                              format='ascii')
    # list of vetted stars to use for the smart template
    vetted_star = np.array(tbl_template['star'])
    # radial velocity of the vetted stars
    vetted_star_rv = np.array(tbl_template['rv'])

    dir = '{}/reconstructed_templates_{}'.format(params['working_directory'], instrument)
    if not os.path.isdir(dir):
        os.mkdir(dir)
    outname = '{}/Template_s1dv_{}-VETTED_sc1d_v_file_A.fits'.format(dir, h['DRSOBJN'])

    if os.path.exists(outname) and not force:
        printc('The template {} already exists. We will skip this file.'.format(outname), 'bad1')
        return

    template_path = None
    if instrument == 'nirps':
        template_path = params['working_directory'] + '/templates_{}/Template_s1dv_{}_sc1d_v_file_A.fits'
        kw_mjd = 'MJD-OBS'

    if instrument == 'spirou':
        template_path = params['working_directory'] + '/templates_{}/Template_s1dv_{}_sc1d_v_file_AB.fits'
        kw_mjd = 'MJDATE'

    vsinis = np.append(0, 1.1 ** np.arange(0, 40))

    vv_name = '{}/model_{}.pkl'.format(params['working_directory'], instrument)

    redo_dict = False
    if os.path.exists(vv_name):
        printc('Reading the vetted star dictionary', 'info')
        vv_dict = read_pickle(vv_name)
        # we check that the vetted stars are still in the dictionary
        if False in [star for star in vv_dict['vetted_star'] if star in vetted_star]:
            printc('Some vetted stars are missing from the dictionary. We will recompute the dictionary.', 'bad2')
            redo_dict = True

        # we look for new files
        vv = vv_dict['vv']
        templates = vv_dict['templates']
        tapas_other_1 = vv_dict['tapas_other_1']
        tapas_water_1 = vv_dict['tapas_water_1']

    wave0 = np.array(tbl0['wavelength'])
    # define the domain for the smart template
    g_domain = (tbl0['wavelength'] > wave_domain[0]) & (tbl0['wavelength'] < wave_domain[1])
    velo_sampling = (c / 1000) / np.nanmedian(tbl0['wavelength'] / np.gradient(tbl0['wavelength']))

    if not os.path.exists(vv_name) or redo_dict:
        vv_dict = {}
        vv_dict['vetted_star'] = vetted_star
        vv_dict['vetted_star_rv'] = vetted_star_rv
        vv_dict['vsinis'] = vsinis

        if not os.path.isdir(params['working_directory'] + '/telluric_model'):
            os.mkdir(params['working_directory'] + '/telluric_model')

        telluric_model_full_path = params['working_directory'] + '/telluric_model/tapas_all_sp.fits.gz'

        if not os.path.isfile(telluric_model_full_path):
            url = 'http://www.astro.umontreal.ca/~artigau/lbl/tapas_all_sp.fits.gz'
            wget.download(url, telluric_model_full_path)

        # tapas telluric spectrum
        tapas = Table.read(telluric_model_full_path)
        combined = 'trans_o3', 'trans_n2o', 'trans_o2', 'trans_co2', 'trans_ch4'
        sp_other = tapas['trans_o3'] * tapas['trans_n2o'] * tapas['trans_o2'] * tapas['trans_co2'] * tapas['trans_ch4']
        sp_water = tapas['trans_h2o']
        tapas_wave = tapas['wavelength']

        # spline the tapas spectrum for water and try components
        sp_other = ius(tapas_wave, sp_other)
        sp_water = ius(tapas_wave, sp_water)

        # get the tapas spectrum in the domain of interest
        tapas_other_1 = sp_other(wave0[g_domain])
        tapas_water_1 = sp_water(wave0[g_domain])

        with warnings.catch_warnings(record=True) as _:
            vv_tapas1 = np.log(tapas_other_1) - lowpassfilter(np.log(tapas_other_1), hpsize)
            vv_tapas2 = np.log(tapas_water_1) - lowpassfilter(np.log(tapas_water_1), hpsize)

        vv = np.zeros([len(tbl0[g_domain]), len(vetted_star) + 2, len(vsinis)])

        templates = np.zeros([len(tbl0), len(vetted_star) + 2, len(vsinis)])
        for ivet in snail(range(len(vetted_star))):
            template_table = Table.read(template_path.format(instrument, vetted_star[ivet]), 1)
            # keep only valid values
            valid = np.isfinite(template_table['flux'])
            valid &= (template_table['flux'] > 0)
            # spline the template
            spl = ius(doppler(template_table['wavelength'][valid], vetted_star_rv[ivet]),
                      template_table['flux'][valid], k=1, ext=1)
            sp_tmp = spl(wave0[g_domain])
            sp_template = spl(wave0)

            for ivsini in snail(range(len(vsinis))):
                if ivsini == 0:
                    lsptmp = np.log(sp_tmp)
                    vv[:, ivet, ivsini] = lsptmp - lowpassfilter(lsptmp, hpsize)
                    templates[:, ivet, ivsini] = sp_template
                else:
                    # apply vsini broadening
                    npix = int(np.round(vsinis[ivsini] / velo_sampling)) * 2 + 5
                    kernel = np.zeros(npix)
                    kernel[npix // 2] = 1

                    kernel = rot_broad(15000 * (1 + 1000 * np.arange(npix) * velo_sampling / c), kernel, 0.6,
                                       vsinis[ivsini])
                    sp_tmp2 = np.zeros_like(sp_tmp)
                    template_tmp = np.zeros_like(sp_template)

                    for ikernel in range(npix):
                        sp_tmp2 += np.roll(sp_tmp, int(npix // 2 - ikernel)) * kernel[ikernel]
                        template_tmp += np.roll(sp_template, int(npix // 2 - ikernel)) * kernel[ikernel]

                    lsptmp = np.log(sp_tmp2)
                    vv[:, ivet, ivsini] = lsptmp - lowpassfilter(lsptmp, hpsize)
                    templates[:, ivet, ivsini] = template_tmp

                vv[:, -2, ivsini] = vv_tapas1
                vv[:, -1, ivsini] = vv_tapas2

        vv_dict['vv'] = vv
        vv_dict['templates'] = templates
        vv_dict['tapas_other_1'] = tapas_other_1
        vv_dict['tapas_water_1'] = tapas_water_1
        save_pickle(vv_name, vv_dict)

        # append for the linear system
        # sp_tmp = np.log(spl(wave0[g_domain]))
        # append for the linear system
        # vv.append(sp_tmp - lowpassfilter(sp_tmp, hpsize))
        # append for the template list
        # templates.append(spl(wave0))

    # into a friendly numpy array
    # templates = np.vstack(templates).T
    # add the tapas spectrum to the linear system
    # with warnings.catch_warnings(record=True) as _:
    #    vv.append()
    #    vv.append()

    # into a friendly numpy array
    # vv = np.vstack(vv).T

    # placeholder for the resulting amplitudes
    all_amps = np.zeros((len(files), len(vetted_star) + 2))
    # placeholder for the resulting radial velocities
    all_vsys = np.zeros(len(files)) + np.nan
    all_vsini = np.zeros(len(files))
    all_vsini_id = np.zeros(len(files), dtype=int)
    all_rms_contrasts = np.zeros(len(files))
    mjdate = np.zeros(len(files))

    for ifile, file_sp in enumerate(files):
        printc('~', 'bad3')
        printc('Processing file {} of {},  {}'.format(ifile + 1, len(files), h['DRSOBJN']), 'number')

        tmp_file = '{}/{}.yaml'.format('per_file_info', file_sp.split('/')[-1].split('.')[0])
        if os.path.exists(tmp_file):
            printc('File {} already exists. We will skip this file.'.format(tmp_file), 'bad1')

            dict_per_file = load_yaml(tmp_file)
            all_vsys[ifile] = dict_per_file['vsys']
            all_vsini[ifile] = dict_per_file['vsini']
            all_vsini_id[ifile] = dict_per_file['vsini_id']
            all_rms_contrasts[ifile] = dict_per_file['rms_contrast']
            all_amps[ifile] = np.array(dict_per_file['amps'])
            mjdate[ifile] = dict_per_file['mjdate']

            continue

        dict_per_file = dict()

        # read the s1d file
        tbl = Table.read(file_sp, 1)
        # get the header
        h = fits.getheader(file_sp)

        scan_range = -300, 300
        allow_skip = True
        if ifile <= 2:
            best_guess_vsini_id = 0
        else:
            best_guess_vsini_id = np.nanmedian(all_vsini_id[0:ifile - 1]).astype(int)

        printc('Best guess vsini id: {:.0f}, vsini = {:.2f} km/s'.format(best_guess_vsini_id,
                                                                         vsinis[best_guess_vsini_id]),
               'number')

        if ifile > 5:
            if sigma(all_vsys[0:ifile - 1]) < 0.5:  # better than 0.5 km/s
                vsys_guess = np.nanmedian(all_vsys[0:ifile - 1])
                printc('Best guess vsys: {:.2f} km/s'.format(vsys_guess), 'number')

                # largest of the two, vsini and 10 km/s
                width_range = np.max([20, 2 * vsinis[best_guess_vsini_id]])
                printc('Width range: {:.2f} km/s'.format(width_range), 'number')
                scan_range = vsys_guess - h['BERV'] - width_range, vsys_guess - h['BERV'] + width_range
                allow_skip = False

        # get the appropriate domain
        sp1 = np.array(tbl['flux'][g_domain])

        # number of steps to try on either side of RV=0. Steps are in
        # pixels but pixels are either 500 or 1000 m/s depending on the
        # instrument
        # step = int(150/velo_sampling)

        # take only one step every 2 km/s
        stride = int(2 / velo_sampling)

        i1, i2, nn = int(scan_range[0] // velo_sampling), int(scan_range[1] // velo_sampling), int(
            np.round(1 / velo_sampling))
        rv_offset = np.arange(i1, i2, nn)

        # only check domain that is nominally without too much telluric contamination
        valid = tapas_other_1 > 0.5
        valid &= tapas_water_1 > 0.5
        sp1[~valid] = np.nan
        with warnings.catch_warnings(record=True) as _:
            lsp1 = np.log(sp1)
        # remove low frequency variations
        lsp1 = lsp1 - lowpassfilter(lsp1, hpsize)
        # remove high-signal spikes
        with warnings.catch_warnings(record=True) as _:
            lsp1[np.abs(lsp1 / sigma(lsp1)) > 20] = np.nan

        # loop over all possible RV offsets
        sig = np.zeros_like(rv_offset, dtype=float) + np.nan
        doskip = np.array(len(rv_offset) * [False])
        for i in snail(range(len(rv_offset)), leave=False):
            if not doskip[i]:
                vv2 = np.array(vv[:, :, best_guess_vsini_id])

                lsp1b = lsp1.copy()
                lsp1b = lsp1b[::stride]
                vv2[:, 0:len(vetted_star)] = np.roll(vv2[:, 0:len(vetted_star)], rv_offset[i], axis=0)
                vv2 = vv2[::stride]

                amps, recon = lin_mini(lsp1b, vv2)
                diff = lsp1b - recon
                # keep the standard deviation of the residuals
                sig[i] = sigma(diff)
                nsig = np.abs(diff / sig[i])
                with warnings.catch_warnings(record=True) as _:
                    lsp1b[nsig > 5] = np.nan

                amps, recon = lin_mini(lsp1b, vv2)
                diff = lsp1b - recon
                sig[i] = sigma(diff)

                if i > 5 and allow_skip:
                    if sig[i] > 0.95 * np.nanmedian(sig[:i]):
                        # we skip the next 5 km/s
                        nskip = int(np.round(5 / velo_sampling))
                        doskip[i + 1:i + nskip] = True
        med_sig = np.nanmedian(sig)
        sig /= med_sig

        keep_sig = np.isfinite(sig)
        sig = sig[keep_sig]
        rv_offset = rv_offset[keep_sig]

        iminsig = np.argmin(sig)

        sig_vsini = np.zeros_like(vsinis, dtype=float) + np.nan
        # loop on vsini

        lsp1b = lsp1.copy()
        lsp1b = lsp1b[::stride]

        # sort vsini by distance to best guess
        ivsini_ord = np.argsort(np.abs(vsinis - vsinis[best_guess_vsini_id]))

        sig_clip = True
        for ivsini in ivsini_ord:
            vv2 = np.array(vv[:, :, ivsini])
            vv2[:, 0:len(vetted_star)] = np.roll(vv2[:, 0:len(vetted_star)], rv_offset[iminsig], axis=0)
            vv2 = vv2[::stride]

            amps, recon = lin_mini(lsp1b, vv2)
            diff = lsp1b - recon
            if sig_clip:
                # keep the standard deviation of the residuals
                sig_vsini[ivsini] = sigma(diff)
                nsig = np.abs(diff / sig_vsini[ivsini])
                with warnings.catch_warnings(record=True) as _:
                    lsp1b[nsig > 5] = np.nan
                sig_clip = False

            amps, recon = lin_mini(lsp1b, vv2)
            diff = lsp1b - recon
            sig_vsini[ivsini] = sigma(diff) / med_sig

        all_vsini_id[ifile] = np.argmin(sig_vsini)

        if all_vsini_id[ifile] == 0:
            all_vsini[ifile] = vsinis[0]
        elif all_vsini_id[ifile] == len(vsinis) - 1:
            all_vsini[ifile] = vsinis[-1]
        else:
            fit = np.polyfit(vsinis[all_vsini_id[ifile] - 1:all_vsini_id[ifile] + 2],
                             sig_vsini[all_vsini_id[ifile] - 1:all_vsini_id[ifile] + 2], 2)
            all_vsini[ifile] = -fit[1] / (2 * fit[0])

        #  mu, sig, amp, zp
        mu = rv_offset[iminsig] * velo_sampling
        zp = np.nanpercentile(sig, 90)
        amp = np.min(sig) - zp
        sig_gauss = 10.0
        try:
            fit, _ = curve_fit(gauss, rv_offset * velo_sampling, sig, p0=[mu, sig_gauss, amp, zp])
        except:
            fit = [-999, -1, 0, 1]
        all_rms_contrasts[ifile] = -amp / zp

        if doplot:
            fig, ax = plt.subplots(1, 2, sharey=True)
            ax[0].plot(rv_offset * velo_sampling, sig, '.-')
            ax[0].plot(rv_offset * velo_sampling, gauss(rv_offset * velo_sampling, *fit), 'r--')

            ax[1].plot(vsinis, sig_vsini)
            ax[1].set(xlabel='vsini (km/s)', ylabel='normalized RMS contrast')
            ax[0].set(xlabel='RV (km/s)', ylabel='normalized RMS contrast')
            plt.show()

        # save the results
        vsys = fit[0] + h['BERV']
        all_vsys[ifile] = vsys
        printc('\tRV {:.3f} km/s, BERV {:.3f} km/s, vsys {:.3f} km/s'.format(fit[0], h['BERV'], vsys), 'number')
        printc('\tFWHM : {:.3f} km/s'.format(fit[1] * 2.355), 'number')
        printc('\t vsini : {:.3f} km/s'.format(all_vsini[ifile]), 'number')
        printc('\t RMS contrast : {:.3f}'.format(all_rms_contrasts[ifile]), 'number')

        # find the best RV offset
        best_offset = rv_offset[np.argmin(sig)]
        lsp1b = lsp1.copy()

        # remove the best fit template
        vv2 = np.array(vv[:, :, all_vsini_id[ifile]])
        vv2[:, 0:len(vetted_star)] = np.roll(vv[:, 0:len(vetted_star), all_vsini_id[ifile]], best_offset, axis=0)
        amps, recon = lin_mini(lsp1b, vv2)
        diff = lsp1 - recon
        # sigma clip the residuals
        with warnings.catch_warnings(record=True) as _:
            bad = np.abs(diff) > 10 * sigma(diff)
        lsp1b[bad] = np.nan
        amps, recon = lin_mini(lsp1b, vv2)

        all_amps[ifile, :] = amps

        # plot the results
        if doplot:
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(wave0[g_domain], lsp1, label='data', color='black')
            ax[0].plot(wave0[g_domain], recon, label='model', color='red')
            tmp1 = vv[:, -2, all_vsini_id[ifile]] * amps[-2]
            tmp2 = vv[:, -1, all_vsini_id[ifile]] * amps[-1]
            tmp1[~np.isfinite(lsp1)] = np.nan
            tmp2[~np.isfinite(lsp1)] = np.nan
            ax[0].set(title='Template & Input Spectra', ylabel='log[flux]', xlabel='Wavelength [nm]')
            ax[1].set(title='residuals')
            ax[1].set(title='Template & Input Spectra', ylabel='log[flux]', xlabel='Wavelength [nm]')
            ax[1].set_xlabel('vsini (km/s)')

            ax[0].plot(wave0[g_domain], tmp1, label='Dry', color='orange')
            ax[0].plot(wave0[g_domain], tmp2, label='Water', color='cyan')
            ax[0].legend()
            ax[1].plot(wave0[g_domain], lsp1 - recon, label='residual')

            plt.show()

        dict_per_file = dict()
        dict_per_file['filename'] = str(file_sp.split('/')[-1])
        dict_per_file['vsys'] = float(vsys)
        dict_per_file['vsini'] = float(all_vsini[ifile])
        dict_per_file['vsini_id'] = int(all_vsini_id[ifile])
        dict_per_file['rms_contrast'] = float(all_rms_contrasts[ifile])
        dict_per_file['DRSOBJN'] = str(h['DRSOBJN'])

        dict_per_file['mjdate'] = float(h[kw_mjd])
        dict_per_file['amps'] = [float(a) for a in amps]
        dict_per_file['vetted_star'] = [str(t) for t in vetted_star]
        save_yaml(tmp_file, dict_per_file)

    nsig_vsys = np.abs(all_vsys - np.nanmedian(all_vsys)) / sigma(all_vsys)
    bad = nsig_vsys > 5

    printc('Bad files : '
           ' '.join([files[i] + '\n' for i in range(len(files)) if bad[i]]), 'bad2')

    # files = files[~bad]
    # all_vsys = all_vsys[~bad]
    # all_amps = all_amps[~bad,:]
    # all_vsini = all_vsini[~bad]
    # all_vsini_id = all_vsini_id[~bad]

    vsini = np.nanmedian(all_vsini)
    vsini_id = np.argmin(np.abs(vsinis - vsini))

    # get the median amplitudes for each template
    amps = all_amps[np.argmin([all_vsys - np.nanmedian(all_vsys)])][0:-2]

    rms_to_other_files = [
        np.nanmedian(np.abs(all_amps[:, :-2] - np.tile(all_amps[i, :-2], len(files)).reshape(all_amps[:, :-2].shape)))
        for i in range(len(files))]
    central_amp = np.argmin(rms_to_other_files)

    templates = templates[:, :-2, vsini_id]

    amps = all_amps[central_amp, :-2]
    printc(h['OBJECT'], 'info')
    for ivet in range(len(vetted_star)):
        # print the name of the template and the amplitude

        n1, med, p1 = np.nanpercentile(all_amps[:, ivet], [15, 50, 86])

        printc('{}\tadopted : {:.3f}, median : {:.3f} 1-sig : {:.3f}'.format(vetted_star[ivet], amps[ivet], med,
                                                                             (p1 - n1) / 2), 'number')

        with warnings.catch_warnings(record=True) as _:
            templates[:, ivet] = templates[:, ivet] ** amps[ivet]

    # construct the template
    template = np.product(templates, axis=1)
    # normalize the template
    template /= np.nanmedian(template)
    # get valid pixels
    valid = np.isfinite(template)
    # interpolate the template to the original wavelength grid and proper vsys
    sp_template = ius(doppler(wave0[valid], -np.nanmedian(all_vsys) * 1000), template[valid], k=1)(wave0)

    spl_mask = ius(doppler(wave0, -np.nanmedian(all_vsys) * 1000), np.array(valid), k=1)(wave0)
    sp_template[spl_mask < 0.5] = np.nan

    sp_template /= np.nanmedian(sp_template)

    template_table = Table()
    template_table['wavelength'] = wave0
    # save the template
    template_table['flux'] = sp_template
    template_table['rms'] = 1e-2
    template_table['rms'][~np.isfinite(template_table['flux'])] = np.nan

    template_table.write(outname, overwrite=True)

    tbl_quality = Table()
    tbl_quality['filename'] = files
    tbl_quality['vsys'] = all_vsys
    tbl_quality['vsini'] = all_vsini
    tbl_quality['vsini_id'] = all_vsini_id
    tbl_quality['rms'] = rms_to_other_files
    tbl_quality['rms_contrast'] = all_rms_contrasts
    tbl_quality['MJDATE'] = mjdate
    tbl_quality = tbl_quality[np.argsort(tbl_quality['MJDATE'])]

    path = 'reconstructed_templates_{}/'.format(instrument)
    if not os.path.isdir(path):
        os.mkdir(path)
    # create a MEF with tbl_quality as a 2nd extension
    hdu = fits.PrimaryHDU()
    hdu1 = fits.BinTableHDU(template_table)
    hdu1.name = 'TEMPLATE'
    hdu2 = fits.BinTableHDU(tbl_quality)
    hdu2.name = 'SCI_TABLE'
    # write hdu
    # name the extensions spectrum, smart_template_quality
    hdul = fits.HDUList([hdu, hdu1, hdu2])
    # write the file

    if instrument == 'nirps':
        suffix1 = 's1dv'
        suffix = 'A'
    elif instrument == 'spirou':
        suffix1 = 's1d'
        suffix = 'AB'
    else:
        printc('Instrument {} not supported'.format(instrument), 'bad3')
        return

    outname_template = '{}Template_{}_{}-VETTED_sc1d_v_file_{}.fits'.format(path, suffix1, h['DRSOBJN'], suffix)
    hdul.writeto(outname_template,
                 overwrite=True)

    tbl_quality.write('{}/Template_{}_{}-VETTED_sc1d_v_file_A_quality.csv'.format(
        path, suffix1, h['DRSOBJN']), overwrite=True)

    printc('Done', 'info')

    printc('Median vsini : {:.2f} km/s'.format(np.nanmedian(tbl_quality['vsini'])), 'number')

    # go back to initial folder
    os.chdir(initial_folder)

    return outname_template, tbl_quality

