# Tạo Ảnh Động Hình Trái Tim từ Mã Python

Dự án này sử dụng mã Python để tạo ra một ảnh GIF động từ 30 hình ảnh được huấn luyện từ mã trong tệp `temp.txt`. Dự án này bao gồm thuật toán tạo ra hình trái tim và chuyển động mượt mà của các điểm theo thời gian, được hiển thị qua ảnh động.

## Tổng quan

Mã trong dự án này sẽ:
1. Huấn luyện hệ thống dựa trên 30 hình ảnh sử dụng mã được lưu trữ trong tệp `temp.txt`.
2. Tạo ra ảnh GIF động từ các hình ảnh đã được huấn luyện.

### Tính năng:
- Xử lý và hiển thị hình ảnh trái tim động.
- Mã huấn luyện được lưu trữ trong tệp `temp.txt`.
- Hiển thị các hình ảnh đã được lưu trữ, minh họa hình ảnh:
![Ảnh nhận dạng số 1]([https://github.com/nhut-share-code/ve_hoa_tao_anh_dong/blob/master/output/0.jpg])
## Yêu cầu

Để chạy mã này, bạn cần:
- Python 3.x
- Các thư viện sau:
  - `numpy`
  - `matplotlib`
  - `Pillow`
  - `opencv-python`
  - `scipy`

Bạn có thể cài đặt các thư viện cần thiết bằng lệnh:

```bash
pip install numpy matplotlib Pillow opencv-python scipy
