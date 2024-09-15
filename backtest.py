########################################################################
# 技術分析指標
# from talib import abstract
import talib as ta
# 回測框架
from backtesting import Backtest, Strategy
from backtesting.test import SMA
from backtesting.lib import crossover
# YAHOO 財經 (數據爬蟲)
import yfinance as yf
# 數據處理
import pandas as pd
import numpy as np
# 基礎
import datetime
# 圖像化
import matplotlib.pyplot as plt
########################################################################


########################################################################
# # 設定時間區間
# start = datetime.datetime(2018,1,1)
# end = datetime.datetime(2024,9,11)
# # 從 YAHOO 下載歷史資料
# raw_data = yf.download("^TWII", start, end)
# # 轉存 CSV 檔
# raw_data.to_csv("data.csv")
########################################################################


########################################################################
# 讀取 CSV 檔
raw_data = pd.read_csv('data.csv')
# 對空值插值 (避免錯誤)
raw_data = raw_data.infer_objects(copy=False)
# 篩選目標區間
raw_data['Date'] = pd.to_datetime(raw_data['Date'])
dataframe = raw_data[(raw_data['Date']>='2023-09-01') & (raw_data['Date']<='2024-09-11')]
dataframe = dataframe.set_index('Date')
########################################################################


########################################################################
# 計算 ATR
def ATR(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    # 計算真實波幅 TR
    tr = pd.DataFrame({
        'high_low': high - low,
        'high_close': (high - close.shift(1)).abs(),
        'low_close': (low - close.shift(1)).abs()
    })
    
    # 真實波幅的最大值
    tr['TR'] = tr.max(axis=1)
    
    # 計算 ATR
    atr = tr['TR'].rolling(window=period).mean()
    return atr
########################################################################


########################################################################
# 自定義策略：布林通道
class BollingerBand(Strategy):
    def init(self):
        # TALIB：布林通道
        self.upperband, self.middleband, self.lowerband = self.I(
            ta.BBANDS, 
            self.data.Close, 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0
        )
        # 自定義變量：緩存止損價格
        self.stop_loss_price = None  
        # 自定義變量：動態計算 ATR
        self.atr = self.I(ATR, self.data.df)

    def next(self):
        # [布林上下軌進場穩定虧損 >> 反向證明可以做選擇權]
        if not self.position:
            if self.data.Close[-1] > self.upperband[-1]:
                self.sell()
                self.stop_loss_price = self.data.Close[-1] + self.atr[-1] * 0.5

            elif self.data.Close[-1] < self.lowerband[-1]:
                self.buy()
                self.stop_loss_price = self.data.Close[-1] - self.atr[-1] * 0.5

        elif self.position.is_long:
            if self.data.Close[-1] <= self.stop_loss_price:
                self.position.close()

            elif self.data.Close[-1] >= self.middleband[-1]:
                self.position.close()

        elif self.position.is_short:
            if self.data.Close[-1] >= self.stop_loss_price:
                self.position.close()
            
            elif self.data.Close[-1] <= self.middleband[-1]:
                self.position.close()


# 自定義策略：均線交叉
class SmaCrossover(Strategy):
    def init(self):
        # TALIB：簡單移動均線
        self.fast_line = self.I(SMA, self.data.Close, 5)
        self.slow_line = self.I(SMA, self.data.Close, 20)
        # 自定義變量：緩存止損價格
        self.stop_loss_price = None  
        # 自定義變量：動態計算 ATR
        self.atr = self.I(ATR, self.data.df)

    def next(self):
        # 均線交叉策略只適合做多，因為下跌通常又快又猛，做空會為賠
        if not self.position:
            if self.slow_line[-1] > self.slow_line[-2]:
                if crossover(self.fast_line, self.slow_line):
                    self.buy()
                    # self.stop_loss_price = self.data.Close[-1] - self.atr[-1] * 0.5
                    self.stop_loss_price = self.data.Close[-1] - 100

            # if self.slow_line[-1] < self.slow_line[-2]:
            #     if crossover(self.slow_line, self.fast_line):
            #         self.sell()
            #         self.stop_loss_price = self.data.Close[-1] + self.atr[-1] * 0.5

        elif self.position.is_long:
            if self.data.Close[-1] <= self.stop_loss_price:
                self.position.close()
            elif crossover(self.slow_line, self.fast_line):
                self.position.close()

        # elif self.position.is_short:
        #     if self.data.Close[-1] >= self.stop_loss_price:
        #         self.position.close()
        #     elif crossover(self.fast_line, self.slow_line):
        #         self.position.close()


# 自定義策略：週期突破
class PriceChannel(Strategy):
    def init(self):
        # 自定義變量：動態計算 ATR
        self.atr = self.I(ATR, self.data.df)

    def next(self):
        is_max = round(self.data.Close[-1], 4) == np.round(self.data.Close[-201:].max(), 4)
        is_min = round(self.data.Close[-1], 4) == np.round(self.data.Close[-201:].min(), 4)

        if is_max:
            self.buy()

        if is_min:
            self.sell()
#######################################################################


########################################################################
# 運行回測並圖像化
result = Backtest(
    # 數據
    dataframe,
    # 策略
    PriceChannel,
    # 資金
    cash=1000000,
    # 稅費
    commission=0.02,
    # 每次操作前自動關閉上次操作
    exclusive_orders=True,
    # 對沖策略
    hedging=False,
    # 收盤交易
    trade_on_close=True,
)
result.run()
result.plot(
    plot_width=None,
    plot_equity=True,
    plot_return=False,
    plot_pl=True,
    plot_volume=True,
    plot_drawdown=False,
    smooth_equity=False,
    relative_equity=True,
    superimpose=True,
    resample=True,
    reverse_indicators=False,
    show_legend=True,
    open_browser=True
)
########################################################################