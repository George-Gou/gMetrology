#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Jianing Gou(goujianing19@mails.ucas.ac.cn)
# datetime:4/19/2023 11:01 PM

import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sp
import math
import matplotlib.dates as md


# load tsf file of gravimeter like superconductivity gravimeter and gPhone gravimeter.
def readTsf(file) -> object:
    f = pd.read_csv(file, sep='\s+', header=None,
                    names=['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Gravity', 'Pressure'])
    for g in range(len(f['Gravity'])):  # 去掉属性值
        if f['Year'][g] == '[DATA]':
            f = f[g + 1:]
            break
    f.index = range(len(f))  # 重置索引
    return f


def nine_polynomial(g):
    """
       Fit the gravity time series to a ninth order polynomial.
    """

    x = np.linspace(0, len(g), len(g))  # get the x-axis
    b = np.polyfit(x, g, 9)
    c = np.poly1d(b)
    polynome_result = np.array(c(x))

    return polynome_result


'''
Compute the earth tide given the [longitude, Latitude, delt, year, month, day, hour, minute, second]
which the "delta" is tidal gravity factor, we take 1.16 usually.
'''


def TheoryGravityTide(lon, lat, zone, delt, year, month, day, hour, minute, second):
    pi = 3.1415926535
    Deltath = delt  # 1.16
    tz = zone  # 8 in china
    Year = year
    Month = month
    Day = day
    Phi = lat
    Phip = Phi - 0.192424 * math.sin(2 * Phi * pi / 180.0)
    Longtitude = lon * pi / 180.0
    Latitude = lat * pi / 180.0

    Phi = Latitude
    Phip = Phip * pi / 180.0
    dg = [0] * len(Year)
    for i in range(len(dg)):
        t = int(hour[i]) + int(minute[i]) / 60.0 + int(second[i]) / 3600.0
        # T0 = Julian(Year(i), Month(i), Day(i))
        T0 = RULO(int(Year[i]), int(Month[i]), int(Day[i]))
        # 2415020.0
        T = (T0 - 2415020.0 + (t - tz) / 24.0) / 36525.0
        S = 270.43416 + 481267.8831 * T - 0.001133 * T * T + 0.000002 * T * T * T
        h = 279.696678 + 36000.738925 * T + 0.0003025 * T * T
        p = 334.3295556 + 4069.034033 * T - 0.010325 * T * T - 0.0000125 * T * T * T
        N = 259.183275 - 1934.142008 * T + 0.002077 * T * T + 0.000002 * T * T * T
        ps = 281.2208333 + 1.719175 * T + 0.0004527 * T * T + 0.000003 * T * T * T
        e = 23.452294 - 0.0130125 * T - 0.00000163889 * T * T + 0.0000005027 * T * T * T
        S = S * pi / 180.0
        h = h * pi / 180.0
        p = p * pi / 180.0
        N = N * pi / 180.0
        ps = ps * pi / 180.0
        e = e * pi / 180.0
        # ** ** ** * solve for moon ** ** ** * #
        crm = 1 + 0.0545 * math.cos(S - p) + 0.0030 * math.cos(2 * (S - p)) + 0.01 * math.cos(
            S - 2 * h + p) + 0.0082 * math.cos(2 * (S - h)) \
              + 0.0006 * math.cos(2 * S - 3 * h + ps) + 0.0009 * math.cos(3 * S - 2 * h - p)
        Lambdam = S + 0.0222 * math.sin(S - 2 * h + p) + 0.1098 * math.sin(S - p) + 0.0115 * math.sin(
            2 * S - 2 * h) + 0.0037 * math.sin(2 * S - 2 * p) \
                  - 0.0032 * math.sin(h - ps) - 0.001 * math.sin(2 * h - 2 * p) + 0.001 * math.sin(
            S - 3 * h + p + ps) + 0.0007 * math.sin(S - h - p + ps) \
                  - 0.0006 * math.sin(S - h) - 0.0005 * math.sin(S + h - p - ps) + 0.0008 * math.sin(
            2 * S - 3 * h + ps) - 0.002 * math.sin(2 * S - 2 * N) \
                  + 0.0009 * math.sin(3 * S - 2 * h - p)
        Beltam = 0.003 * math.sin(S - 2 * h + N) + 0.0895 * math.sin(S - N) + 0.0049 * math.sin(
            2 * S - p - N) - 0.0048 * math.sin(p - N) - 0.0008 * \
                 math.sin(2 * h - p - N) + 0.001 * math.sin(2 * S - 2 * h + p - N) + 0.0006 * math.sin(
            3 * S - 2 * h - N)
        Delta = math.sin(e) * math.sin(Lambdam) * math.cos(Beltam) + math.cos(e) * math.sin(Beltam)
        Theta = ((t - tz) * (15 * pi / 180.0)) + h + Longtitude - pi
        H = math.cos(Beltam) * math.cos(Lambdam) * math.cos(Theta) + math.sin(Theta) * (
                math.cos(e) * math.cos(Beltam) * math.sin(Lambdam) - math.sin(e) * math.sin(Beltam))
        Zm = math.sin(Phip) * Delta + math.cos(Phip) * H  # # Zm = cos(Zm)
        # # solve for sun # #
        crs = 1 + 0.0168 * math.cos(h - ps) + 0.0003 * math.cos(2 * h - 2 * ps)
        Lambdas = h + 0.0335 * math.sin(h - ps) + 0.0004 * math.sin(2 * h - 2 * ps)
        Beltas = 0.0
        Zs = math.sin(Phip) * math.sin(e) * math.sin(Lambdas) + math.cos(Phip) * (
                math.cos(Lambdas) * math.cos(Theta) + math.sin(Theta) * math.cos(e) * math.sin(Lambdas))
        # # Zs = cos(Zs)
        F = 0.998327 + 0.001676 * math.cos(2 * Phi)
        Gt = -165.17 * F * crm * crm * crm * (Zm * Zm - 1.0 / 3.0) - 1.3708 * F * F * crm * crm * crm * crm * Zm * (
                5 * Zm * Zm - 3) - 76.08 * F * crs * crs * crs * (Zs * Zs - 1.0 / 3.0)
        dg[i] = Deltath * Gt
    return dg


def RULO(Y, M, D):
    ND = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    l1 = Y - int(Y / 4) * 4
    if l1 == 0:
        ND[2] = 29
    K = 0
    for i in range(1, M):
        K = K + ND[i]
    R = float(365 * (Y - 1900) + int((Y - 1901) / 4) + K + D) + 15019.5 + 2400000
    return R


def air_corr(Pressure, Elevation):
    Pn = 1.01325e3 * (1 - 0.0065 * Elevation / 288.15) ** 5.2559
    out = -0.3 * (Pressure - Pn)
    return out



