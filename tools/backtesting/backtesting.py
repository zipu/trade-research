import pandas as pd
import numpy as np
import sqlite3 as lite
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import OrderedDict, defaultdict
import time

from ..tools import product_info

LONG = 1
SHORT = -1

class UnitTrade:
    """
    상품별로 전략을 테스트 하는 클래스
    """

    def __init__(self, feed, pinfo):
        self.feed = feed #pandas dataframe 상품 가격 데이터 
        self.pinfo = pinfo #xingapi로 받아온 상품 기초정보
        self._statement = [] #총평가기록
        self.tradelog = {} #매매기록

    def entry(self,entryid, date, price, position=None, lot=1):
        self.tradelog[entryid] = OrderedDict({
            'entry': [entryid, position, lot, date, price],
            'exit': []
        })
        

    def exit(self, entryid, date, price, lot=1):
        ex = [lot, date, price]
        self.tradelog[entryid]['exit'].append(ex)
        
        #진입 계약수보다 청산 총계약수가 많으면 에러
        tot_lot = 0
        for ex in self.tradelog[entryid]['exit']:
            tot_lot += ex[0]

        if tot_lot > self.tradelog[entryid]['entry'][2]:
            raise ValueError('total exit lots exceed entry lots')

        else:
            self._statement.append(self.tradelog[entryid]['entry'] + ex)

    def report(self):
        return BackTester.get_result(self.statement, self.info['name'])

    def plot(self):
        statement = self.statement
        fig, (ax) = plt.subplots(2,1, figsize=(12,10))
        ax[0].plot(self.data.close)
        for _,position, entrydate, exitdate in statement[['position','entrydate','exitdate']].itertuples():
            color = 'red' if position == 1 else 'green'
            ax[0].axvspan(entrydate, exitdate, facecolor=color, alpha=0.3)

        ax[1].plot(statement.entrydate, statement.cumprofit)
        ax[1].plot(statement.entrydate, statement.cumprofit.cummax())

        return ax[0]
    
    @property
    def statement(self):
        statement = pd.DataFrame(
            data = self._statement,
            columns=['entryid','position','entrylot', 'entrydate', 'entryprice','exitlot', 'exitdate', 'exitprice'])
        statement['profit'] = statement.position*(statement.exitprice - statement.entryprice)/self.pinfo['tick_unit']
        statement['cumprofit'] = statement.profit.cumsum()
        statement['drawdown'] = statement.cumprofit.cummax() - statement.cumprofit
        statement.insert(0, 'name', self.pinfo['name'])
        #receipt['drawdown'] = (1- receipt.cumprofit / receipt.cumprofit.cummax())*100 
        return statement
    


class BackTester:
    #INFO_PATH = '../data/db.sqlite3'

    def __init__(self, feed=None, strategy=None, name='STRATEGY', principal=1000):
        
        self.feed = feed
        self.columns = feed.attrs['columns'].split(';')
        self.name = name    
        
        self.meta = product_info()#self._get_meta(BackTester.INFO_PATH)

        self.principal = principal
        self.trades = []

        if strategy:
            self.strategy = strategy           
        else: 
            self.strategy = self.default_strategy

        
    @property
    def statement(self):
        return self._statement

    
    def report(self, level=0):
        if level == 0:
            report = BackTester.get_result(self._statement, self.name)
            
        elif level == 1:
            report = BackTester.get_result(self._statement, self.name)
            report = report.append(pd.concat([item.report() for item in self.trades]))
        
        return report
    
    
    def summary(self, level=0):
        self.plot()
        return self.report(level)
    
    def run(self):
        print("trading started. it takes few minutes...")
        for idx, raw in enumerate(self.feed.values()):
            symbol = raw.attrs['symbol']
            if symbol == 'None' or not symbol:
                continue

            else:
                data = pd.DataFrame(raw.value[:,1:], index=raw.value[:,0].astype('M8[s]'), columns=self.columns[1:])
                data.insert(0,'date', data.index)
                
                trade = UnitTrade(data, self.meta[symbol])
                self.strategy(trade)
                self.trades.append(trade)

                print(f'\rprocessing..({idx})', end='', flush=True)
        
        self._statement = pd.concat([trade.statement for trade in self.trades])[['name','entrydate','exitdate','profit']]
        self._statement.sort_values('entrydate', inplace=True)
        self._statement.reset_index(drop=True, inplace=True)
        self._statement['cumprofit'] = self._statement.profit.cumsum()
        self._statement['drawdown'] = self._statement.cumprofit.cummax() - self._statement.cumprofit
        
        print("\nDone")

    def _get_meta(self, infopath):
        con = lite.connect(infopath)
        info = pd.read_sql('select * from trading_product', con).set_index('group', drop=True)
        return info.to_dict(orient='index')
    
    def plot(self):
        x = self._statement.entrydate.values
        y1 = self._statement.cumprofit.values
        y2 = self._statement.cumprofit.cummax().values
        fig, ax = plt.subplots(1,1, figsize=(10, 8))
        ax.fill_between(x, 0, y1, where=y1>=0, facecolor='green',alpha=0.4, interpolate=True)
        ax.fill_between(x, 0, y1, where=y1<0, facecolor='red', alpha=0.8, interpolate=True)
        ax.fill_between(x,y1, y2, color='grey', alpha=0.2)

        #labels
        ax.set_title('Cumulative Profit')
        ax.set_xlabel('Date')
        ax.set_ylabel('Profit')
        ax.yaxis.set_label_position("right")
        
        #style
        ax.grid(linestyle='--')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.yaxis.tick_right()
        fig.autofmt_xdate()
        
        plt.show()    
    
    @staticmethod
    def get_result(statement, name):
        #receipt = self.receipt
        #mdd = 100*(statement.drawdown/statement.cumprofit.cummax()).max()
        mdd = statement.drawdown.max()
        cum_profit = statement.cumprofit.iloc[-1]
        ave_profit = statement.profit.mean()
        rng = statement.exitdate.max() - statement.entrydate.min()
        rng = rng.days/365
        cagr = pow(cum_profit, 1/rng) - 1 if cum_profit > 0 else 0
        win_rate = 100 * statement.profit[statement.profit >= 0].count() \
                    / statement.profit.count()
        max_loss = statement.profit.min()
        max_profit = statement.profit.max()
        #ave_win = statement.profit[receipt.profit >= 0].mean()
        #ave_lose = statement.profit[receipt.profit < 0].mean()
        profit_factor = abs(statement.profit[statement.profit >=0].sum()/\
                       statement.profit[statement.profit < 0].sum())
        num_trade = len(statement) / rng # 연평균 매매 횟수

        data = [[cum_profit, cagr, profit_factor, mdd, win_rate, max_profit, max_loss, num_trade]]
        columns = ['총손익(틱)','CAGR(%)','손익비','MDD(틱)','승률(%)', '최대수익(틱)','최대손실(틱)','연평균 매매횟수']
        return pd.DataFrame(data, index=[name], columns=columns)
    
    @staticmethod
    def default_strategy(trade):
        """
        default: buy only ma cross system
        진입: 20종가이평이 60종가이평 돌파상승 한 다음날 시가 진입
        청산: 20종가이평이 60종가이평 돌파하락 한 다음날 시가 청산
        """
        data = trade.feed
        data['ma20'] = data.close.rolling(20).mean()
        data['ma60'] = data.close.rolling(60).mean()
        data['cross'] = (data.ma20 > data.ma60).astype('int')
        data['signal'] = data.cross.diff().shift(1)
        data.dropna(inplace=True)
        signals = data[(data.signal == 1) | (data.signal == -1)]

        entryid= 0
        for date, price, signal in signals[['open','signal']].itertuples():
            if signal == 1:
                entryid += 1
                trade.entry(entryid, date, price,position=1, lot=1)
            elif signal == -1:
                trade.exit(entryid, date, price, lot=1)