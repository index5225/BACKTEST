########################################################################
# 技術分析指標
# from talib import abstract
import talib as ta
# 回測框架
from backtesting import Backtest, Strategy
from backtesting.test import SMA 
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
# 自定義參數
# re_download = True
re_download = False
########################################################################


########################################################################
# 重新下載歷史數據
if re_download == True:
    # 設定時間區間
    start = datetime.datetime(2018,1,1)
    end = datetime.datetime(2024,9,11)
    # 從 YAHOO 下載歷史資料
    dataframe = yf.download("^TWII", start, end)
    # 轉存 CSV 檔
    dataframe.to_csv("data.csv")
########################################################################


########################################################################
# 讀取 CSV 檔
dataframe = pd.read_csv('data.csv')
# 對空值插值 (避免錯誤)
dataframe = dataframe.infer_objects(copy=False)
# 篩選目標區間
dataframe['Date'] = pd.to_datetime(dataframe['Date'])
dataframe_filter = dataframe[(dataframe['Date']>='2023-01-01') & (dataframe['Date']<='2024-09-11')]
dataframe_filter = dataframe_filter.set_index('Date')
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
class BollingerBandsStrategy(Strategy):
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
        # TALIB：簡單移動均線
        # self.sma = self.I(SMA, self.data.Close, 20)
        # 自定義變量：緩存止損價格
        self.stop_loss_price = None  
        # 自定義變量：動態計算 ATR
        self.atr = self.I(ATR, self.data.df)

    def next(self):
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

        # data.Close[-1]：對應的是前一根K線的收盤價。
        # data.Close[0]：對應的是當前正在處理的K線的收盤價。
        # data.Close[1]：這個值會超出範圍，因為這是下一根的值，回測中不應訪問未來的數據。
########################################################################


########################################################################
# 運行回測並圖像化 (數據、策略、資金、稅費)
result = Backtest(dataframe_filter, BollingerBandsStrategy, cash=100000, commission=0.02)
result.run()
result.plot(relative_equity=True, plot_equity=True, plot_return=False, plot_volume=False, superimpose=True)
# relative_equity 權益及報酬 (True 百分比表示、False 金額表示)
# plot_equity 顯示權益曲線
# plot_return 顯示報酬率
# plot_volume 顯示量
# superimpose 顯示月k在背景
########################################################################