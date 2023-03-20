import yfinance as yf
import pandas as pd
import numpy as np
import datetime, pytz
import matplotlib.pyplot as plt
from numpy import ndarray
from pandas import DataFrame
import requests
import matplotlib.transforms as mtransforms



# 收盤時間用變數
df: DataFrame = pd.DataFrame({'weight': {'AAPL': 12.43, 'MSFT': 12.07, 'AMZN': 6.05, '^NDX': float('nan')}})  # 當天更新權重

# # 設定初始參數 下載的股票、時間判別
time_USA = datetime.datetime.now(pytz.timezone('America/New_York'))  # 美國現在精準時間
start_time_USA = (time_USA).strftime("%Y/%m/%d 09:30:00-05:00")  # 美國開盤時間

# 開盤時間
df = df.join(yf.download(list(df.index), period='2d', interval='1d', rounding=2)['Close'].iloc[0].T)  # 匯入 df 昨收資料
all_stock: DataFrame = yf.download(list(df.index), period='1d', interval='1m')#[start_time_USA:now_time_1min_USA]

# 計算一點數影響大盤點數, 同時設定Dataframe
df.columns = ['weight', 'y_Close']  # 更改列名
df['1p_affect'] = round(
    (df['y_Close']['^NDX'] * df['weight'][0:len(df.index) - 1]) / (df['y_Close'][0:len(df.index) - 1] * 100)
    , 2)
df = df.sort_values(by='weight', ascending=False)
len_stock = len(df.index)

df['Open'] = all_stock['Open'].iloc[0]  # 加入 今日開盤價
df.loc['top3'] = [np.sum(df['weight'][0:3]) / 100] + [np.nan] * 3  # 加入 Top3 和


print(df.replace(np.nan, ''))
print(time_USA)


# 設定各權值股系列nparray : ((價格-開盤價)*影響點數 )/大盤價格
# type> nparray
# .dif_array                : 輸出 變化點數的 nparray
# .dif_point_array          : 輸出 影響大盤點數的 nparray
# .times_array(10) : 輸出 在'大盤10點震幅外的' 占大盤影響點數權重
class one_stock:
    def __init__(self, stock_name: str, ndx_difarray: ndarray, all_stock: DataFrame):
        self.stock_name = stock_name
        self.df: DataFrame = df
        self.NDX_difarray = ndx_difarray
        self.stock_nparray = np.array(all_stock[stock_name])

    def dif_array(self) -> ndarray:  # 輸出 變化點數的 nparray
        return self.stock_nparray - self.df['Open'][self.stock_name]

    def dif_point_array(self) -> ndarray:  # 輸出 影響大盤點數的 nparray
        return self.dif_array() * self.df['1p_affect'][self.stock_name]


# 設定畫布
def setup_figure_and_axes(xlim: int, normal_difdif: float):  # x = times_sum_top3

    if xlim < 15:
        ax1.set_xlim(0, 15)
        ax2.set_xlim(0, 15)
        ax3.set_xlim(0, 15)
    else:
        ax1.set_xlim(0, xlim)
        ax2.set_xlim(0, xlim)
        ax3.set_xlim(0, xlim)

    ax1.set_ylabel('NDX', color='green')
    ax1.xaxis.set_visible(False)
    ax1.tick_params(axis='y', labelcolor='green')

    ax2.set_ylabel('2dif point', color='blue')
    ax2.xaxis.set_visible(False)
    ax2.tick_params(axis='y', labelcolor='blue')

    ax3.set_xlabel('time')
    ax3.set_ylabel('BarC')
    ax3.set_ylim(-0.7, 0.7)
    ax3.set_yticks([normal_difdif, 0, -normal_difdif])


# 畫線
def plot_line(x: range, sup_sty: tuple,
              ax1, NDX_dif_array: ndarray,
              ax2, top3_sum_dif_array: ndarray,
              ax3, normal_difdif: ndarray):
    # 畫輔助線,   sup_sty = 自訂輔助線型style
    ax1.axhline(y=0.0, linestyle=sup_sty, color='gray', linewidth=0.5)  # 畫 y = 0. 參考線
    ax1.axhline(y=30.0, linestyle=sup_sty, color='magenta', linewidth=0.7)  # 畫 y = 0. 參考線
    ax1.axhline(y=-30.0, linestyle=sup_sty, color='magenta', linewidth=0.7)  # 畫 y = 0. 參考線
    ax1.axvline(x=61, linestyle=sup_sty, color='gray', linewidth=0.5)  # 畫 x = 61 參考線 (10:30分隔線)

    ax1.axhline(y=df['y_Close']['^NDX'] - df['Open']['^NDX'], linestyle='-', color='blue', linewidth=0.7,
                alpha=0.7)  # 昨收參考線 ( 或 跳空參考線)

    ax2.axhline(y=0.0, linestyle=sup_sty, color='gray', linewidth=0.5)  # 畫 y = 0 point 參考線

    ax3.axhline(y=normal_difdif, linestyle=sup_sty, color='magenta', linewidth=0.7)  # 畫 y = 0.4 參考線
    ax3.axhline(y=normal_difdif / 2, linestyle=sup_sty, color='gray', linewidth=0.7)  # 畫 y = 0.4 參考線
    ax3.axhline(y=-normal_difdif / 2, linestyle=sup_sty, color='gray', linewidth=0.7)  # 畫 y = -0.4 參考線
    ax3.axhline(y=-normal_difdif, linestyle=sup_sty, color='magenta', linewidth=0.7)  # 畫 y = -0.4 參考線
    ax3.axhline(y=0, linestyle=sup_sty, color='gray', linewidth=0.7)  # 畫 y = -0.4 參考線
    ax3.axvline(x=61, linestyle=sup_sty, color='gray', linewidth=0.5)  # 畫 x = 61 參考線 (10:30分隔線)

    # 化折線趨勢圖
    # ax1
    ax1.plot(x, NDX_dif_array, 'g-', linewidth=1)
    ax1.plot(x, top3_sum_dif_array, linestyle='-.', color='cornflowerblue', linewidth=1)
    ax1.plot(x, NDX_dif_array - top3_sum_dif_array, 'y-.', linewidth=1)

    # ax2
    # 參數設定
    dif_dif_point: ndarray = NDX_dif_array - top3_sum_dif_array * 2
    dif_dif_weight: ndarray = dif_dif_point / np.abs(NDX_dif_array)
    dif_dif_point_adjnormal: ndarray = dif_dif_point * normal_difdif / np.abs(dif_dif_weight)  # 矯正為 若比重為 0.4 時的影響點數
    Bt_adj_dif_point: ndarray = dif_dif_point_adjnormal - dif_dif_point

    # rgb 顏色設定
    rgba_colors: ndarray = np.where(
        np.expand_dims(dif_dif_point < 0, axis=1)
        , [0., 0.5, 0., 1.]
        , [1., 0., 0., 1.])  # 紅色綠色 匯入

    ax2.bar(x, dif_dif_point, color=rgba_colors)
    ax2.bar(x, dif_dif_point_adjnormal, color='gray', alpha=0.4, zorder=0)
    ax2.plot(x, Bt_adj_dif_point, color='blue', lw=0.8)

    # ax3
    # 邏輯設定
    logic = np.logical_or(np.abs(NDX_dif_array) >= 1, dif_dif_weight != float('nan'))
    # rgb 顏色設定--設定alpha值
    alpha_logic = np.abs(NDX_dif_array) < 15
    rgba_colors[..., 3] = np.where(alpha_logic, 0.2, 1.)

    ax3.bar(x, np.where(logic, dif_dif_weight, 0.05 * np.sign(dif_dif_weight)), color=rgba_colors)
    
    # 畫不合理範圍區塊顏色--normal_difdif 之外
    grayArea = np.where(np.abs(dif_dif_weight) > normal_difdif, np.where(alpha_logic, 0, 1), 0)
    trans2 = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
    trans3 = mtransforms.blended_transform_factory(ax3.transData, ax3.transAxes)
    ax2.fill_between(x, 0, 1, where=grayArea == 1, facecolor='gray', alpha=0.4, transform=trans2, zorder=0)
    ax3.fill_between(x, 0, 1, where=grayArea == 1, facecolor='gray', alpha=0.4, transform=trans3, zorder=0)


    # 傳送line 通知
    global FTTT
    if FTTT == True and dif_dif_weight[-1] * dif_dif_weight[-2] < 0:
        FTTT = False
        if dif_dif_weight[-2] < 0:
            requests.get("https://maker.ifttt.com/trigger/Monitor/with/key/5ZeIUkDLXYS9-cRBiGqcBE-qwY33akt4a0xIJpffcr?value1=:<br>dif_dif_weight color changed<br><br>Green to red")
        elif dif_dif_weight[-2] > 0:
            requests.get("https://maker.ifttt.com/trigger/Monitor/with/key/5ZeIUkDLXYS9-cRBiGqcBE-qwY33akt4a0xIJpffcr?value1=:<br>dif_dif_weight color changed<br><br>red to green")
        elif dif_dif_weight[-2] == 0:
            requests.get("https://maker.ifttt.com/trigger/Monitor/with/key/5ZeIUkDLXYS9-cRBiGqcBE-qwY33akt4a0xIJpffcr?value1=:<br>dif_dif_weight color compressed")
        else:
            requests.get("https://maker.ifttt.com/trigger/Monitor/with/key/5ZeIUkDLXYS9-cRBiGqcBE-qwY33akt4a0xIJpffcr?value1=:<br>bug happend")


# //////////////////////////////////////////////////////////////////////////////////
from matplotlib.animation import FuncAnimation


time_judge: str = "start"
FTTT: bool = False
symbol = ('AAPL', 'MSFT', 'AMZN', '^NDX')

def update_line(i):
    # 時間辨識
    global symbol, time_judge, FTTT
    if not FTTT and not datetime.datetime.now().strftime("%M") == time_judge:
        time_judge = datetime.datetime.now().strftime("%M")
        FTTT = True

    update_stock: DataFrame = yf.download(symbol, period='1d', interval='1m', rounding=2)['Close']

    # 做成四個 Series 再合併將其
    dropna_update: DataFrame = pd.concat(
        (
            update_stock['^NDX'][:-5]
                .append(update_stock['^NDX'][-5:].dropna(axis=0, how='any'))
                .reset_index(drop=True)
            , update_stock['AAPL'][:-5]
                .append(update_stock['AAPL'][-5:].dropna(axis=0, how='any'))
                .reset_index(drop=True)
            , update_stock['MSFT'][:-5]
                .append(update_stock['MSFT'][-5:].dropna(axis=0, how='any'))
                .reset_index(drop=True)
            , update_stock['AMZN'][:-5]
                .append(update_stock['AMZN'][-5:].dropna(axis=0, how='any'))
                .reset_index(drop=True)
        )
        , join='outer'
        , axis=1
    )
    # print(dropna_update)

    NDX_difarray: ndarray = np.array(dropna_update['^NDX'] - df['Open']['^NDX'])

    AAPL = one_stock('AAPL', NDX_difarray, dropna_update)
    MSFT = one_stock('MSFT', NDX_difarray, dropna_update)
    AMZN = one_stock('AMZN', NDX_difarray, dropna_update)

    ax1.cla()
    ax2.cla()
    ax3.cla()
    setup_figure_and_axes(xlim=len(NDX_difarray),
                          normal_difdif=round(1 - df['weight']['top3'] * 2, 2))

    plot_line(x=range(len(NDX_difarray)), sup_sty=(0, (5, 10)),
              ax1=ax1, NDX_dif_array=NDX_difarray,
              ax2=ax2, top3_sum_dif_array=AAPL.dif_point_array() + AMZN.dif_point_array() + MSFT.dif_point_array(),
              ax3=ax3, normal_difdif=round(1 - df['weight']['top3'] * 2, 2))



# 畫視覺圖
fig = plt.figure()
ax1 = fig.add_axes([0.06, 0.3, 0.9, 0.65])  # 第一條界線( 控制畫佈的左, 下, 寬度, 高度 )
ax2 = fig.add_axes([0.06, 0.2, 0.9, 0.1])  # 第二條界線
ax3 = fig.add_axes([0.06, 0.1, 0.9, 0.1])

ani = FuncAnimation(fig, update_line, blit=False, interval=5000, frames=10000)

plt.show()