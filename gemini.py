import os
import queue
from collections import deque  # 新增：用于高效存储 CSI 历史数据的定长队列
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit
from ecg_ui import Ui_ecg_get
import serial
from threading import Thread, Lock
import pyqtgraph as pg
import numpy as np
from datetime import datetime
import pandas as pd
from gnuradio import gr

########## 新增：引入共享队列与 GNU Radio 流图 ##########
import shared_data
from wifi_loopback_real_no_pyqt import wifi_loopback_real_no_pyqt


###########################################################

class MySignals(QObject):
    plain_text_print = pyqtSignal(QTextEdit, str)
    update_table = pyqtSignal(str)


class Stats(QMainWindow):
    def __init__(self):
        super().__init__()

        # 尽早初始化锁，确保任何操作都能安全调用
        self.data_lock = Lock()

        ########## 新增：初始化 GNU Radio 和 CSI 历史缓存 ##########
        self.tb = wifi_loopback_real_no_pyqt()
        # 使用 deque 存储最近的 10000 个 CSI 包 (大约可存 40 秒的 250Hz 数据)，防止内存溢出
        self.csi_buffer = deque(maxlen=10000)
        ##############################################################

        self.thread = None
        self.ms = MySignals()
        self.ser = None
        self.ui = Ui_ecg_get()
        self.ui.setupUi(self)

        # 绑定按键事件
        self.ui.start_serial.clicked.connect(self.handle_start_serial)
        self.ui.stop_serial.clicked.connect(self.handle_stop_serial)
        self.ui.save_data.clicked.connect(self.handle_save_data)
        self.ui.del_data.clicked.connect(self.handle_del_data)
        self.ui.load_list.clicked.connect(self.handle_load_list)
        self.ui.file_list.clicked.connect(self.handle_load_data)
        self.ms.plain_text_print.connect(self.printToGui)

        # 画图点数，数据及画笔初始化
        self.sample_rate = 250
        self.window_size = 12
        self.max_points = self.sample_rate * self.window_size

        self.x_data = np.linspace(0, self.window_size, self.max_points)
        self.ecg_data, self.breath_data, self.wifi_data = np.zeros(self.max_points), np.zeros(
            self.max_points), np.zeros(self.max_points)
        self.timestamp_ecg_data = np.zeros(self.max_points)

        # 画出现在和历史图画
        pen = pg.mkPen(color='red', width=2)
        self.curve_ecg = self.ui.real_time_ecg_plot.plot(self.x_data, self.ecg_data, pen=pen)
        self.hist_curve_ecg = self.ui.history_ecg_plot.plot(pen=pen)
        pen = pg.mkPen(color='blue', width=2)
        self.curve_breath = self.ui.real_time_breath_plot.plot(self.x_data, self.breath_data, pen=pen)
        self.hist_curve_breath = self.ui.history_breath_plot.plot(pen=pen)
        pen = pg.mkPen(color='green', width=2)
        self.curve_wifi = self.ui.real_time_wifi_plot.plot(self.x_data, self.wifi_data, pen=pen)
        self.hist_curve_wifi = self.ui.history_wifi_plot.plot(pen=pen)

        # 保存文件夹
        self.save_dir = "saved_datasets"
        os.makedirs(self.save_dir, exist_ok=True)

        # 图标更新策略
        pg.setConfigOptions(antialias=True)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

        ########## 新增：处理 CSI 队列的定时器 ##########
        self.csi_timer = QTimer()
        self.csi_timer.timeout.connect(self.process_csi_queue)
        #################################################

    def printToGui(self, fb, text):
        fb.append(str(text))
        fb.ensureCursorVisible()

    ########## 新增：从后台队列提取 USRP 数据并处理 ##########
    def process_csi_queue(self):
        while not shared_data.csi_queue.empty():
            try:
                # 接收到的是极速版的元组：(同一批的时间戳, N个包的二维矩阵)
                ts, csi_matrix = shared_data.csi_queue.get_nowait()
                n_packets = len(csi_matrix)

                # 计算这批数据每个包的平均幅值，用于在绿色的 wifi_plot 上画趋势折线
                mag_batch = np.mean(np.abs(csi_matrix), axis=1)

                with self.data_lock:
                    # 1. 把最原始的 52 载波复数存入历史缓存备用 (用于最终保存)
                    for i in range(n_packets):
                        self.csi_buffer.append((ts, csi_matrix[i]))

                    # 2. 极速移位更新一维画图数组，避免 for 循环
                    if n_packets >= self.max_points:
                        self.wifi_data[:] = mag_batch[-self.max_points:]
                    else:
                        self.wifi_data[:-n_packets] = self.wifi_data[n_packets:]
                        self.wifi_data[-n_packets:] = mag_batch
            except queue.Empty:
                break

    ##########################################################

    def update_plot(self):
        with self.data_lock:
            ecg_snap = self.ecg_data.copy()
            breath_snap = self.breath_data.copy()
            wifi_snap = self.wifi_data.copy()
        self.curve_ecg.setData(self.x_data, ecg_snap)
        self.curve_breath.setData(self.x_data, breath_snap)
        self.curve_wifi.setData(self.x_data, wifi_snap)

    def handle_start_serial(self):
        try:
            self.ser = serial.Serial(port=self.ui.serial_numcb.currentText(),
                                     baudrate=eval(self.ui.baud_cb.currentText()), timeout=None)
            self.ui.start_serial.setEnabled(False)
            self.ui.stop_serial.setEnabled(True)
            self.ui.serial_numcb.setEnabled(False)
            self.ui.baud_cb.setEnabled(False)
            self.ui.save_data.setEnabled(False)
            self.ui.del_data.setEnabled(False)
            self.ms.plain_text_print.emit(self.ui.conf_text,
                                          f'Successfully open {self.ser.name}, Baud rate: {self.ser.baudrate}')

            self.timer.start(15)

            ########## 修改：同时启动 CSI 定时器和 GNU Radio ##########
            self.csi_timer.start(10)
            self.tb.start()

            ###########################################################

            def threadFunc():
                while self.ser.is_open:
                    try:
                        line = self.ser.read_until(b'\n')
                        part = line.decode('utf-8', 'ignore').strip()
                        self.ms.plain_text_print.emit(self.ui.serial_plain_text, part)
                        ecg_new_data = part.split(',')[0]
                        breath_new_data = part.split(',')[1]
                        current_time_ecg = datetime.now().timestamp()

                        with self.data_lock:
                            self.ecg_data[:-1] = self.ecg_data[1:]
                            self.ecg_data[-1] = float(ecg_new_data)
                            self.breath_data[:-1] = self.breath_data[1:]
                            self.breath_data[-1] = float(breath_new_data)
                            self.timestamp_ecg_data[:-1] = self.timestamp_ecg_data[1:]
                            self.timestamp_ecg_data[-1] = current_time_ecg
                    except:
                        pass

            self.thread = Thread(target=threadFunc)
            self.thread.start()
        except Exception as e:
            self.ms.plain_text_print.emit(self.ui.conf_text, f'open serial failed\n,{e}')
            print('open serial failed\n', e)

    def handle_stop_serial(self):
        try:
            ########## 修改：停止串口前，先安全停止 USRP 硬件和 CSI 接收 ##########
            self.tb.stop()
            self.tb.wait()
            self.csi_timer.stop()
            ######################################################################

            self.ser.close()
            self.timer.stop()
            self.ui.start_serial.setEnabled(True)
            self.ui.stop_serial.setEnabled(False)
            self.ui.serial_numcb.setEnabled(True)
            self.ui.baud_cb.setEnabled(True)
            self.ui.save_data.setEnabled(True)
            self.ui.del_data.setEnabled(True)
            self.ms.plain_text_print.emit(self.ui.conf_text, 'close serial success')
        except Exception as e:
            self.ms.plain_text_print.emit(self.ui.conf_text, f'stop serial failed\n,{e}')
            print('stop serial failed\n', e)

    def handle_save_data(self):
        # 1. 冻结瞬间状态获取快照
        with self.data_lock:
            ecg_snap = self.ecg_data.copy()
            breath_snap = self.breath_data.copy()
            time_ecg_snap = self.timestamp_ecg_data.copy()
            ########## 新增：获取 CSI 历史数据快照 ##########
            csi_snap = list(self.csi_buffer)
            #################################################

        # 转换心电图时间戳为可视化字符串
        time_strings = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-2] if ts > 0 else "" for ts in
                        time_ecg_snap]

        ########## 新增：最核心的“基于时间戳对齐”算法 ##########
        # 将 CSI 列表解解包，便于 numpy 高速运算
        if len(csi_snap) > 0:
            csi_times = np.array([item[0] for item in csi_snap])
            csi_frames = np.array([item[1] for item in csi_snap])
        else:
            csi_times = np.array([])
            csi_frames = np.zeros((0, 52), dtype=np.complex64)

        aligned_csi_mag = []
        for ecg_t in time_ecg_snap:
            # 如果心电图有有效时间戳，且收到过 CSI 数据
            if ecg_t > 0 and len(csi_times) > 0:
                # 找到时间轴上离这个心电图采样点最近的一个 WiFi 数据包
                idx = np.abs(csi_times - ecg_t).argmin()

                # 同步容错率：如果两者时间差在 0.1 秒以内，认为对齐成功，提取 52 载波幅值
                if np.abs(csi_times[idx] - ecg_t) < 0.1:
                    aligned_csi_mag.append(np.abs(csi_frames[idx]))
                else:
                    aligned_csi_mag.append(np.zeros(52))  # 若未收到 WiFi 填 0
            else:
                aligned_csi_mag.append(np.zeros(52))

        aligned_csi_mag = np.array(aligned_csi_mag)  # 最终得到 (3000, 52) 的漂亮矩阵
        ########################################################

        # 利用 pandas 构建数据集，准备写入CSV
        data_dict = {
            'ECG_Timestamp': time_strings,
            'ECG': ecg_snap,
            'Breath': breath_snap
        }

        ########## 新增：将 52 个载波的幅值平铺为 52 列扩展进数据集 ##########
        for i in range(52):
            data_dict[f'CSI_Mag_{i}'] = aligned_csi_mag[:, i]
        #####################################################################

        df = pd.DataFrame(data_dict)

        # 生成以当前时间命名的文件名
        file_name = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = os.path.join(self.save_dir, file_name)

        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        self.ms.plain_text_print.emit(self.ui.conf_text, f'Save data successfully')

    def handle_del_data(self):
        self.ui.conf_text.clear()
        self.ui.serial_plain_text.clear()

        # 图标重新画
        with self.data_lock:
            self.x_data = np.linspace(0, self.window_size, self.max_points)
            self.ecg_data, self.breath_data, self.wifi_data = np.zeros(self.max_points), np.zeros(
                self.max_points), np.zeros(self.max_points)
            self.timestamp_ecg_data = np.zeros(self.max_points)
            ########## 新增：清空 CSI 历史缓存 ##########
            self.csi_buffer.clear()
            ###########################################

        self.curve_ecg.setData(self.x_data, self.ecg_data)
        self.curve_breath.setData(self.x_data, self.breath_data)
        self.curve_wifi.setData(self.x_data, self.wifi_data)
        self.ms.plain_text_print.emit(self.ui.conf_text, f'Delete data successfully')

    def handle_load_list(self):
        file_list = os.listdir(self.save_dir)
        csv_file = [f for f in file_list if os.path.splitext(f)[1] == '.csv']
        self.ui.file_list.addItems(csv_file)
        self.ms.plain_text_print.emit(self.ui.conf_text, f'Load list successfully')

    def handle_load_data(self):
        csv_file = self.ui.file_list.currentItem().text()
        df = pd.read_csv(os.path.join(self.save_dir, csv_file))
        if 'ECG' in df.columns and 'Breath' in df.columns:
            hist_ecg = df['ECG'].values
            hist_breath = df['Breath'].values

            x_hist = np.linspace(0, self.window_size, len(hist_ecg))

            self.hist_curve_ecg.setData(x_hist, hist_ecg)
            self.hist_curve_breath.setData(x_hist, hist_breath)

            self.ms.plain_text_print.emit(self.ui.conf_text, f'Displaying History: {csv_file}')


if __name__ == '__main__':
    ########## 新增：赋予整个应用实时抢占权限，消灭 USRP 溢出 ##########
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print("警告: 无法开启实时调度 (可能需要 root/sudo 权限)。")
    ###################################################################

    app = QApplication([])
    stats = Stats()
    stats.show()
    app.exec_()