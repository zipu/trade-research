import tables as tb

class OHLC(tb.IsDescription):
    """
    일봉데이터
    Table structure:
        - date : POSIX 시간(초)을 Integer 형태로 저장
        - open : 시가
        - high: 고가
        - low: 저가
        - close: 종가
        - volume: 거래량
    """
    date = tb.Time32Col(pos=0)
    open = tb.Float64Col(pos=1)
    high = tb.Float64Col(pos=2)
    low = tb.Float64Col(pos=3)
    close = tb.Float64Col(pos=4)
    volume = tb.UInt64Col(pos=5)
    
class Density(tb.IsDescription):
    """
    trade density
    Table structure:
        - date: 날짜
        - price: 가격
        - value: 거래량
    """

    date = tb.Time32Col(pos=0)
    price = tb.Float64Col(pos=1)
    value = tb.Float64Col(pos=2)