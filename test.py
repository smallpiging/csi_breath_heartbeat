import queue
from gnuradio import gr
import time

import shared_data
from wifi_loopback_real_no_pyqt import wifi_loopback_real_no_pyqt

print(time.time())
if gr.enable_realtime_scheduling() != gr.RT_OK:
    print("警告: 无法开启实时调度 (可能需要 root/sudo 权限)。")
tb = wifi_loopback_real_no_pyqt()
tb.start()
num = 0

try:
    while True:
        while not shared_data.csi_queue.empty():
            event = shared_data.csi_queue.get()
            num = num + 1

except KeyboardInterrupt:
    print(num)
    print('关闭usrp')
finally:
    tb.stop()
    tb.wait()