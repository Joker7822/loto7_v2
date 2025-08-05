# optimize_all_models.py が失われました。再アップロードしてください。


import subprocess

def push_results_to_github(commit_msg="Update optuna results"):
    try:
        subprocess.run(["git", "add", "optuna_results"], check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push"], check=True)
        print("[INFO] GitHub に optuna_results を push しました。")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] GitHub push 失敗: {e}")


if __name__ == "__main__":
    main()
    push_results_to_github("Auto-optuna update")
