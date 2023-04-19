#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Jianing Gou(goujianing19@mails.ucas.ac.cn)
# datetime:4/19/2023 11:19 PM
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sp
import matplotlib.dates as md
import gMetrology.Utils.noise.snm as snm

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    filename = '20210607_334append.tsf'
    data = snm.readTsf(filename)
    gravity = data['Gravity'] * 1000
    pressure = data['Pressure']
    time = []
    for i in range(len(data)):
        time.append(datetime.datetime(data['Year'][i], data['Month'][i], data['Day'][i], data['Hour'][i],
                                      data['Minute'][i],
                                      data['Second'][i]))

    title = filename[-22:-10] + "original data"
    ax = plt.gca()
    plt.title(title)
    plt.plot(time, gravity)
    plt.ylabel(r'gravity $10^{-8}m/s^2$')
    xfmt = md.DateFormatter('%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    figname = filename[-22:-10] + "original.png"
    plt.savefig('image/' + figname, dpi=600)
    plt.show()

    # remove linear drift
    grav_de = sp.detrend(gravity, type='linear', overwrite_data=False)
    ax = plt.gca()
    plt.ylabel(r'gravity $10^{-8}m/s^2$')
    xfmt = md.DateFormatter('%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.plot(time, grav_de)
    plt.title('remove drift')
    figname = filename[-22:-10] + "detrend.png"
    plt.savefig('image/' + figname, dpi=600)
    plt.show()

    Zone = 8
    Delt = 1.16
    Lat, Lon = [33.98, 116.79]
    TheoTide = snm.TheoryGravityTide(Lon, Lat, Zone, Delt, data['Year'], data['Month'], data['Day'], data['Hour'],
                                     data['Minute'],
                                     data['Second'])
    grav_detide = grav_de - TheoTide
    ax = plt.gca()
    plt.plot(time, TheoTide)
    plt.title('solid earth tide')
    xfmt = md.DateFormatter('%m-%d')
    plt.ylabel(r'gravity $10^{-8}m/s^2$')
    ax.xaxis.set_major_formatter(xfmt)
    figname = filename[-22:-10] + "solidearthtide.png"
    plt.savefig('image/' + figname, dpi=600)
    plt.show()

    ax = plt.gca()
    plt.plot(time, grav_detide)
    xfmt = md.DateFormatter('%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.title('remove solid earth tide')
    plt.ylabel(r'gravity $10^{-8}m/s^2$')
    figname = filename[-22:-10] + "detide.png"
    plt.savefig('image/' + figname, dpi=600)
    plt.show()

    # remove barometric effect

    grav_depre = grav_detide - snm.air_corr(pressure, 160)

    ax = plt.gca()
    plt.plot(time, grav_depre)
    xfmt = md.DateFormatter('%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.title('remove barometric effect')
    plt.ylabel(r'gravity $10^{-8}m/s^2$')
    figname = filename[-22:-10] + "pressure.png"
    plt.savefig('image/' + figname, dpi=600)
    plt.show()

    # fit the nine order polynomial
    grav_deploy = grav_depre - snm.nine_polynomial(grav_depre)
    ax = plt.gca()
    plt.plot(time, grav_deploy)
    xfmt = md.DateFormatter('%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.title('Fit with nine order polynomial')
    plt.ylabel(r'gravity $10^{-8}m/s^2$')
    figname = filename[-22:-10] + "ninepoly.png"
    plt.savefig('image/' + figname, dpi=600)
    plt.show()

    # Computer PSD
    days = int(len(grav_deploy) / 1440)
    print('average psd \tsnm ', '\n')
    fs_res = np.zeros([1025, 50])
    psd_res = np.zeros([1025, 50])
    snm_res = np.zeros([50, 2])
    for j in range(days):
        grav_deploy_day = grav_deploy[j * 1440:1440 * j + 1440]
        # step 1 : divide the data into single day
        window = sp.hann(len(grav_deploy_day))

        # 2、add hanning window to the gravity data

        grav_win = window * grav_deploy_day

        origin_signal_len = len(grav_win)
        new_signal_len = 0
        n = 0
        for i in range(1, 21):
            if 2 ** i > len(grav_win):
                n = i
                new_signal_len = 2 ** i
                break
            else:
                continue

        # 3 padding with zero
        zero_list = np.zeros(new_signal_len - origin_signal_len)
        grav_win_app = np.append(grav_win, zero_list)
        fs, psd = sp.periodogram(grav_win_app, fs=1 / 60, nfft=new_signal_len)
        sumfe = []
        for i in range(len(fs)):
            if 1 / 200 > fs[i] > 1 / 600:
                sumfe.append(psd[i])
        psd_mean = np.mean(sumfe)
        """
        Frequency period :  [1/600 - 1/200] s 
        """

        snm = np.log10(psd_mean) + 2.5
        print(psd_mean, snm)
        snm_res[j, 0] = psd_mean
        snm_res[j, 1] = snm
        fs_res[:, j] = fs
        psd_res[:, j] = psd

    # 4 save the PSD(power density result) : left: frequency, right: PSD value
    result = pd.DataFrame(data, columns=['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'])
    df = pd.DataFrame(np.squeeze(snm_res), columns=['MeanPSD', 'Snm'])

    # 5 find the quietest days for 3
    small_psd = df.nsmallest(3, 'Snm', keep='all')
    small_psd_save = (psd_res[:, small_psd.index[0]] + psd_res[:, small_psd.index[1]] + psd_res[:,
                                                                                        small_psd.index[2]]) / 3
    figname = filename[-22:-10] + "quiestday.txt"
    np.savetxt('image/' + figname, small_psd.index, fmt='%f')

    psd_save = [fs, small_psd_save]
    psd_save = np.transpose(psd_save)
    filename_save = filename[-22:-10] + "psd.txt"
    file = open(filename_save, "w")
    np.savetxt('image/' + filename_save, psd_save, fmt='%f')

    plt.figure()
    plt.loglog(fs, small_psd_save)
    title = filename[-22:-10] + "PSD "
    plt.title(title)
    NLNM = np.loadtxt('NLNM.txt')
    plt.loglog(1. / NLNM[:, 0], pow(10, (NLNM[:, 1] + 160) / 10), color='red')
    plt.xlabel("Frequency Hz")
    plt.ylabel(r'PSD $(nm/s^2)^2/Hz$')
    # plt.ylabel("PSD " + '$(nm/s^2)^2/Hz$', usetex=True)
    plt.xlim([1e-5, 1e-2])
    plt.ylim([1e-10, 1e5])
    plt.grid()
    plt.legend(["PSD", "NLNM"])
    figname = filename[-22:-10] + "PSD.png"
    plt.savefig('image/' + figname, dpi=600)
    plt.show()
