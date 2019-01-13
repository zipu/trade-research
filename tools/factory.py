import os
#import base64
#from Crypto.Cipher import XOR

#from matplotlib.lines import Line2D
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

def rolling_window(arr, window):
    """
    numpy 어레이의 rolling window array를 반환함
    stride trick을 사용하기 때문에 메모리 사용에 효율적임
    ex)
    arr: [1,2,3,4,5,6,7]
    window: 3
    return: 
    [ [1,2,3],
      [2,3,4],
      [3,4,5],
      [4,5,6],
      [5,6,7] ]
    """

    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def set_ATR(metrics, span):
    """
    Average True Range를 구함
    metrics: ohlc를 포함한 데이터 프레임
    metrics 오브젝트에 ATR column를 추가함
    """

    df = pd.DataFrame()
    df['hl'] = metrics['high'] - metrics['low']
    df['hc'] = np.abs(metrics['high'] - metrics['close'].shift(1))
    df['lc'] = np.abs(metrics['low'] - metrics['close'].shift(1))
    df['TR'] = df.max(axis=1)
    metrics['ATR'] = df['TR'].ewm(span).mean()


def norm(data, ntype='abs_diff'):
    """
    Data Normalization
    머신러닝 훈련 데이터 만들때 주로 사용됨
    """
    if ntype=="abs_diff":
        """
        mean: 0
        scale factor: absolute diff mean
        """
        base = np.abs(data.diff()).mean()
        return (data-data.mean())/base

    if ntype=='minmax':
        """
        (data - min)/(max-min)
        """
        return (data-data.min())/(data.max()-data.min())

    if ntype=='zscore':
        return (data-data.mean())/data.std()


def split(arr):
    """
    arr의 연속된 값들을 그룹으로 묶음
    ex) 
    input: [1,2,3,6,7,8,10]
    output: [[1,2,3],[6,7,8],[10]]
    """
    return np.split(arr, np.where(np.diff(arr) != 1)[0]+1)

"""
바 차트
"""
def ohlc_chart(ax, quotes, linewidth=1, colors=['k','k'], background='lightgoldenrodyellow'):
    
    for idx, color in enumerate(colors):
        cond = quotes['close']>= quotes['open']
        data = quotes[cond] if idx == 0 else quotes[~cond]
        ohlc = data[['open','high','low','close']].values
        o,h,l,c = np.squeeze(np.split(ohlc, 4, axis=1))
        
        #x축 세팅
        dates =data.index.values
        
        if dates.dtype == 'M8[ns]':
            offset = np.timedelta64(8, 'h')
        elif dates.dtype == 'int64':
            offset = 0.3
        
        
        ax.vlines(dates, l, h, linewidth=linewidth, color=color)
        ax.hlines(o, dates-offset, dates, linewidth=linewidth, color=color)
        ax.hlines(c, dates, dates+offset, linewidth=linewidth, color=color)

    #style
    ax.grid(linestyle='--')
    ax.set_facecolor(background)
    ax.yaxis.tick_right()
    return ax

"""
캔들 차트
"""
def candle_chart(ax, quotes, linewidth=1, colors=['k','k'], background='lightgoldenrodyellow'):
    
    for idx, color in enumerate(colors):
        cond = quotes['close']>= quotes['open']
        data = quotes[cond] if idx == 0 else quotes[~cond]
        ohlc = data[['open','high','low','close']].values
        o,h,l,c = np.squeeze(np.split(ohlc, 4, axis=1))
        
        #x축 세팅
        dates =data.index.values
        
        if dates.dtype == 'M8[ns]':
            offset = np.timedelta64(8, 'h')
        elif dates.dtype == 'int64':
            offset = 0.3
        
        ax.vlines(dates, l, h, linewidth=linewidth, color=color)
        
        if idx == 0:
            ax.vlines(dates, o, c, linewidth=12, color=color)
        elif idx == 1:
            ax.vlines(dates, c, o, linewidth=12, color=color)
        
        #ax.hlines(o, dates-offset, dates, linewidth=linewidth, color=color)
        #ax.hlines(c, dates, dates+offset, linewidth=linewidth, color=color)

    #style
    ax.grid(linestyle='--')
    ax.set_facecolor(background)
    ax.yaxis.tick_right()
    return ax


"""
 암호화 매소드
"""
def encrypt(key, plaintext):
  cipher = XOR.new(key)
  return base64.b64encode(cipher.encrypt(plaintext))

def decrypt(key, ciphertext):
  cipher = XOR.new(key)
  return cipher.decrypt(base64.b64decode(ciphertext))