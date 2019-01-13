import os
import h5py
import tables as tb
import pandas as pd


from .config import DATADIR

def open_file(module, fname, mode='r', comp=False, force=False):
    """
    모듈별 file object를 생성
    """
    fpath = os.path.join(DATADIR, fname)
    
    if not force and mode=='w' and os.path.isfile(fpath):
        raise FileExistsError("file '%s' alreay exist"%fname)
    else:
        if module == 'h5py':
            return h5py.File(fpath, mode)

        elif module == 'tb':
            if comp:
                filters = tb.Filters(complib='blosc', complevel=9)
                return tb.open_file(fpath, mode=mode, filters=filters)
            else:
                return tb.open_file(fpath, mode=mode)


def load_products():
    import json

    fpath = os.path.join(DATADIR, 'products.json')
    fobj = open(fpath).read()
    return json.loads(fobj)


def dataframe(symbol, file):
    """
    hdf5에 저장된 numpy array를 pandas dataframe 형식으로 변환하여 리턴
    file.attrs['columns'] 에는 array의 각 column name을 포함하고 있어야함
    각 file[symbol]['name']에는 각 데이터 name을 가지고 있어야함
    """
    columns = file.attrs['columns'].split(';')
    df = pd.DataFrame(file[symbol].value, columns=columns)
    df['date'] = df['date'].astype('M8[s]')
    df.set_index('date', inplace=True)
    df.name = file[symbol].attrs['name']
    return df

#def product_info():
#    """
#    DB에서 종목정보 불러와 Dict type으로 리턴
#    """
#    fpath = os.path.join(DATADIR, 'db.sqlite3')
#    con = lite.connect(fpath)
#    products = pd.read_sql('select * from trading_product', con)
#    products.set_index(['group'], drop=False, inplace=True)
#    products = products.to_dict(orient='index')
#    return products