# ABSA-model

## Cài đặt thư viện cần thiết
chuyển tới thư mục src 
`cd src `

chạy lệnh để cài thư viện
`sh install.sh`

download dataset từ kaggle
`python absa_download_data.py`

tiến hành config lại path data vì hiện tại path data đang được fix cứng 

tiến hành train model 
`python main.py -epochs 10 -layer_bert 6 -cl_alpha 0.2 -log_path=path_here`

với log_path là path của log file trong quá trình train, evaluate 
