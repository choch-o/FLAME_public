python3 power_profiler.py --gpu --outfile gpu_avg_5_epochs_10 --epochs 10
sleep 1m
python3 power_profiler.py --gpu --outfile gpu_avg_5_epochs_20 --epochs 20
sleep 1m
python3 power_profiler.py --outfile cpu_avg_5_epochs_10 --epochs 10
sleep 1m
python3 power_profiler.py --outfile cpu_avg_5_epochs_20 --epochs 20
