cd ~/下載
unzip -q train-50k-2.zip -d /media/ee303/4T/SynTxt-Gen/
if [ $? -eq 0 ]; then
    echo "解壓縮完成！"
else
    echo "解壓縮失敗！" >&2
    exit 1
fi
unzip -q train-50k-3.zip -d /media/ee303/4T/SynTxt-Gen/
if [ $? -eq 0 ]; then
    echo "解壓縮完成！"
else
    echo "解壓縮失敗！" >&2
    exit 1
fi

cd /media/ee303/4T/SynTxt-Gen/
python Gen2.py --text_dir train-50k-2/txt --data_dir SynTxt3D_50k_gen2
python Gen2.py --text_dir train-50k-3/txt --data_dir SynTxt3D_50k_gen3
