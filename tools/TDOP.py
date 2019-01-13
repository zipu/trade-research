import os
from datetime import datetime, timedelta
import numpy as np

#from .tools import rolling_window

def get_factors(table, pinfo):
    """
    density table로부터 주요 factor들을 반환
    x: 가격 array에 매핑되는 x축 index array
    columns: 가격에 매핑되는 x index의 시간별 리스트
    dates: 날짜 array
    """
    density = table.read()
    tickunit = pinfo['tick_unit']
    digit = pinfo['decimal_places']
    
    dates = density['date'].astype('M8[s]')
    prices = np.round(density['price'], digit)
    values = density['value']
    values[np.isinf(values)] = 0 #inf --> 0
    values[values> values.mean() * values.std()*5] = 0 ## larger than std * 15 --> 0
    values[np.isnan(values)] = 0 # nan --> 0
    
    min_price = prices.min()
    max_price = prices.max()
    
    x = np.arange(min_price, max_price+tickunit/2, tickunit).round(digit)
    columns = np.searchsorted(x, prices)
    
    return x, columns, values, dates



def create_tdop(columns, values, dates, now=None, period=None, decayfactor=1):
    """
    tdop array를 반환
    args:
     - columns: 가격의 열 번호를 가진 numpy array
     - values: 가격대별 거래량
     - dates: 날짜 array
     - now: 계산 시점
     - period: 계산 길이 (단위: 년)
     - decayfactor: 시간 감소 주기 (단위: 일)
     """
    if not now:
        now = np.datetime64(datetime.now() + timedelta(hours=1))
    
    if not period:
        start = dates[0]
    else:
        start = now - np.timedelta64(period * 365, 'D')
    
    exclude_idx=np.where((dates<start) | (now<dates))[0]
    values = values.copy()
    values[exclude_idx] = 0
    
    #scale factor: exp(-(date - date)/decayfactor )
    delta = (now - dates)/np.timedelta64(decayfactor,'D') # 시차(일수)
    delta[delta<0] = np.nan 
    #delta = delta +1 # 최소시차 = 1
    weight = values * np.exp(-delta)
    #weight = values * np.exp(-np.sqrt(delta))
    weight[np.isnan(weight)] = 0
    
    tdop = np.bincount(columns, weights=weight)
    return tdop

def get_SR(tdop, threshold):
    """
    tdop array로부터 지지저항선을 계산하여 반환
    threshold: normalized cum tdop의 SR 임계치 % ex)0.99 --> 99% 
    """
    th = (1-threshold)/2
    norm = tdop/tdop.sum()
    cum = norm.cumsum()
    args = np.where((cum > th) & (cum < 1-th))[0]

    if args.size > 0:
        return args.min(), args.max()
    else:
        return None, None


def get_resist(tdops, threshold):
    """
    tdops로부터 저항선을 계산하여 반환
    threshold: normalized cum tdop의 SR 임계치 % ex)0.99 --> 99%
    """
    th = (1-threshold) / 2
    values = tdops['tdop'].value
    dates = tdops['dates'].value.astype('M8[s]')
    price = tdops['prices'].value
    
    lower = []
    upper = []
    for date, value in zip(dates, values):
        
        tdop = value.cumsum()/value.sum() if value.sum() > 0 else value
        effective = np.where( (th < tdop) & (tdop < 1-th))[0]
        if effective.size> 0:
            lower.append(price[effective.min()])
            upper.append(price[effective.max()])
        
        else: #임계값이 없으면 마지막 값
            lower.append(lower[-1] if lower else np.nan)
            upper.append(upper[-1] if upper else np.nan)
    return (lower,upper)

def split(arr, idx):
    """
    입력받은 index array를 연속된 숫자들로 그룹으로 묶고
    idx에 가까운 순서로 sort하여 반환함
    """
    groups = np.split(arr, np.where(np.diff(arr) != 1)[0]+1)
    args = []
    for group in groups:
        args.append(np.abs(group - idx).min())
    sortedargs = np.argsort(args)
    
    return np.array(groups)[sortedargs]


def gaussian(x, weight, mu, sigma):
    """
    gaussian distribution
    """
    return weight * np.exp(-(x-mu)**2/(2.*sigma**2))

def mixture(x, parameters):
    """
    gaussian mixture
    """
    out = np.zeros_like(x.astype('float'))
    for (weight, mu, sigma) in parameters:
        out += gaussian(x, weight, mu, sigma)
    return out

def residual(parametersflat, y, x):
    """
    gaussian mixture와 real data 사이의 residual을 계산하여 반환
    scipy.optimize의 leastsq 함수의 cost function으로 사용
    """
    parameters = [tuple(r) for r in parametersflat.reshape(len(parametersflat)//3,3)]
    return y - mixture(x, parameters)

def get_tdop_array2(density, pinfo, decayfactor=1, period=None, now=None):
    """
    pytables density table로부터 정해진 날짜의 tdop array를 반환한다.
    args: 
     density: pytables group,
     info: 상품정보
     decayfactor: 시간 감소 주기 (일수)
     now: numpy datetime64 object
     period: time window 

    @@ 입력된 now 값을 기준으로 그 이후에 생성된 데이터는 무시함
       (과거 날짜 기준의 tdop 생성용)
    """
    
    tickunit = pinfo['tick_unit']
    digit = pinfo['decimal_places']
    name = pinfo['name']
    code = pinfo['symbol']
    
    min_price = density.read(field='price').min()
    max_price = density.read(field='price').max()
    x = np.arange(min_price, max_price+tickunit/2, tickunit).round(digit)
    
    if now == None:
        now = np.datetime64(datetime.now()+timedelta(hours=1)) #1시간 시차
    
    if period:
        start = (now - np.timedelta64(365*period, 'D'))\
                        .astype('M8[s]').astype('int64')

    else:
        start = 0

    end = now.astype('M8[s]').astype('int64')
    density = density.read_where('(start<=date)&(date<=end)')
        
    dates = density['date'].astype('M8[s]')
    prices = np.round(density['price'], digit)
    values = density['value']
    values[np.isinf(values)] = 0 #inf --> 0
    values[values> values.mean() * values.std()*5] = 0 ## larger than std * 15 --> 0
    values[np.isnan(values)] = 0 # nan --> 0
    
    #scale factor: sqrt(date - date)
    delta = (now - dates)/np.timedelta64(decayfactor,'D') # 시차(일수)
    delta = delta +1 # 최소시차 = 1
    weight = values/np.sqrt(delta)
    
    columns = np.searchsorted(x, prices)
    if not np.isin(x.argmax(), columns):
        columns = np.append(columns, x.argmax())
        weight = np.append(weight, 0)
    tdop = np.bincount(columns, weights=weight)
    
    return x, tdop, now

def get_tdop_array(density, info, period=1, now=None):
    """
    pytables density table로부터 정해진 날짜의 tdop array를 반환한다.
    args: 
     density: pytables group,
     info: 상품정보
     period: 시간 주기 (일수)
     now: numpy datetime64 object 
    
    @@ 입력된 now 값을 기준으로 그 이후에 생성된 데이터는 무시함
       (과거 날짜 기준의 tdop 생성용)
    """

    if now == None:
        now = np.datetime64(datetime.now()+timedelta(hours=1)) #1시간 시차
        
    tickunit = info['tick_unit']
    digit = info['decimal_places']
    name = info['name']
    code = info['symbol']

    #print("processing: ", name)
    # data preprocessing
    dates = density.DateMapper.read(field='date').astype('M8[s]')
    source = density.Minute.read()
    price = source['price'].round(digit)
    row_num = source['row']
    col_num = np.rint((price-price.min())/tickunit)
    value = source['value']
    value[np.isinf(value)] = 0 #inf --> 0
    value[value > value.std()*15] = 0 # larger than std * 15 --> 0
    value[np.isnan(value)] = 0 # nan --> 0
    
    
    #sparse matrix creation
    shape = (row_num.max()+1, col_num.max()+1)
    matrix = sp.csr_matrix((value, (row_num, col_num)), shape=shape)
    
    #scale factor: sqrt(date - date)
    delta = (now - dates)/np.timedelta64(period,'D') # 시차(일수)
    delta[delta<0] = np.nan 
    delta = delta +1 # 최소시차 = 1
    seq = 1/np.sqrt(delta)
    seq[np.isnan(seq)] = 0
    scale = sp.diags(seq) #diagonal matrix
    
    #normalized TDOP
    tdop = np.squeeze(np.asarray((scale*matrix).sum(axis=0)))
    #normed_tdop = tdop / tdop.sum()
    x_ticks = np.arange(price.min(), price.max()+tickunit/2, tickunit).round(digit)
    
    return x_ticks, tdop, now

def area_ratio(array, window):
    """
    중심값을 기준으로 좌우 window의 area ratio를 구함
    return:  log2(left area/ right area)
    """
    arr = np.copy(array)
    
    center = window + 1
    length = window*2 + 1
        
    arr[arr==0] = np.nan
    rolled = rolling_window(arr, length)
    left = rolled[:,:center].sum(axis=1)
    right = rolled[:,center+1:].sum(axis=1)
    
    #diff = left - right
    diff = np.log2(left/right)
    #diff = left/right
    diff[np.isnan(diff)] = 0
    diff = np.concatenate([np.zeros(window), diff, np.zeros(window)])
    return diff
