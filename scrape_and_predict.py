import subprocess

def run_scripts():
    # scrapingloto6.pyを実行
    print("Running scrapingloto7.py...")
    subprocess.run(['python', 'scrapingloto7.py'], check=True)
    
    # lottery_prediction.pyを実行
    print("Running lottery_prediction.py...")
    subprocess.run(['python', 'lottery_prediction.py'], check=True)

if __name__ == '__main__':
    run_scripts()
