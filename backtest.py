########################################################################
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from talib import abstract
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
########################################################################


########################################################################
# 自定義參數
re_download = False
########################################################################


########################################################################
# 重新下載歷史數據
if re_download == True:
    # 設定時間區間
    start = datetime.datetime(2018,1,1)
    end = datetime.datetime(2020,1,1)
    # 從 YAHOO 下載歷史資料
    df = yf.download("AAPL", start, end)
    # 轉存 CSV 檔
    df.to_csv("data.csv")
########################################################################


########################################################################
# 讀取 CSV 檔
df = pd.read_csv('data.csv')
# 對空值插值 (避免錯誤)
df = df.interpolate()
# 篩選目標區間
date_sel = df['Date']>='2018-01-01'
df_sel = df[date_sel]
df_sel['Date'] = pd.to_datetime(df_sel['Date'])
df_sel = df_sel.set_index('Date')
########################################################################


########################################################################
# 自定義指標：用 TALIB 函式庫計算 KD 值
df_tmp = df_sel
# 重新命名 TALIB 欄位
df_tmp.rename(columns = {'High':'high', 'Low':'low','Adj Close':'close','Close':'non_adj close'}, inplace = True) 
kd = abstract.STOCH(df_tmp)
kd.index = df_tmp.index
# 合併兩個資料表
fnl_df = df_tmp.join(kd).dropna()
# 重新命名回測欄位
fnl_df.rename(columns = {'high':'High', 'low':'Low','close':'Close'}, inplace = True)
########################################################################


########################################################################
# 自定義函數：跳過策略中的資料
def I_bypass(data):
    return data
########################################################################


########################################################################
# 自定義策略：KD 值
class KDCross(Strategy): 
    lower_bound = 20  
    upper_bound = 80  

    def init(self):
        # K
        self.k = self.I(I_bypass, self.data.slowk)
        # D
        self.d = self.I(I_bypass, self.data.slowd)

    def next(self):
        if crossover(self.k, self.d) and self.k < self.lower_bound and self.d < self.lower_bound and not self.position:
            self.buy() 
        elif crossover(self.d, self.k) and self.k > self.upper_bound and self.d > self.upper_bound: 
            if self.position and self.position.is_long:
                self.position.close()
########################################################################


########################################################################
# 運行回測
bt = Backtest(fnl_df, KDCross, cash = 10000, commission = .002)
rslt = bt.run()
print(rslt)
bt.plot()
########################################################################