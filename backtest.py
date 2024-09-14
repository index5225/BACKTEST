########################################################################
# 技術分析指標
# from talib import abstract
import talib as ta
# 回測框架
from backtesting import Backtest, Strategy
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
dataframe_filter = dataframe[(dataframe['Date']>='2018-01-01') & (dataframe['Date']<='2024-09-11')]
dataframe_filter = dataframe_filter.set_index('Date')
########################################################################


########################################################################
# 自定義策略：布林通道上軌買入
class BollingerBandsStrategy(Strategy):
    def init(self):
        # 使用 talib 計算布林帶
        self.upperband, self.middleband, self.lowerband = self.I(
            ta.BBANDS, 
            self.data.Close, 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0
        )

    def next(self):
        if (self.middleband[-1] >= self.middleband[-2]):
            # 如果沒有頭寸，並且價格跌破下軌，買入
            if self.data.Close[-1] < self.lowerband[-1] and not self.position.is_long:
            # if self.data.Close[-1] < self.lowerband[-1]:
                self.buy()

            # 如果持有多頭倉位，並且價格回到中軌，賣出平倉
            elif self.position.is_long and self.data.Close[-1] >= self.middleband[-1]:
                self.position.close()

        elif (self.middleband[-1] <= self.middleband[-2]):
            # 如果沒有頭寸，並且價格突破上軌，賣出
            if self.data.Close[-1] > self.upperband[-1] and not self.position.is_short:
            # elif self.data.Close[-1] > self.upperband[-1]:
                self.sell()

            # 如果持有空頭倉位，並且價格回到中軌，買入平倉
            elif self.position.is_short and self.data.Close[-1] <= self.middleband[-1]:
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