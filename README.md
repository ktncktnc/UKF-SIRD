
# UKF-SIRD

### Setup

  

```bash

cd UKF_SIRD

pip install -r requirements.txt

```

  

### Run

Import mô hình

``` python

from model import SIRD

```

Tạo mô hình (country mặc định là Vietnam, có thể thay đổi bằng nước khác, mặc định lấy dữ liệu của tất cả các ngày
``` python

m = SIRD()

```
Lấy dữ liệu từ internet, reset lại các tham số cho mô hình.
``` python

m.pull_data()

```
Tối ưu tham số với dữ liệu hiện tại
``` python

m.solve(-1)

```
Vẽ đồ thị các tham số của mô hình
``` python

m.plot()

```
Dự đoán các giá trị tiếp theo của mô hình
``` python

m.original_pred(<số ngày>)

```

Vẽ đồ thị I,R lúc này:
``` python

m.plot_ir()

```
Vẽ đồ thị D lúc này:
``` python

m.plot_d()

```
Sau khi chạy xong, cần reset lại mô hình trước khi chạy lại
``` python

m.reset()

```