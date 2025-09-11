import os
import subprocess
import time
import random

episode_ids = [
      7,  11,  13,  40,  42,  52,  70,  94, 116, 140, 150, 156, 166, 171, 176, 
    181, 187, 190, 191, 207, 218, 226, 232, 244, 247, 259, 265, 275, 308, 312, 
    321, 330, 338, 348, 362, 371, 377, 387, 403, 411, 423, 432, 439, 447, 454, 
    461, 469, 479, 513, 516, 526, 531, 546, 550, 559, 568, 576, 586, 602, 609, 
    620, 643, 655, 670, 677, 698, 705, 715, 721, 739, 743, 748, 755, 765, 781, 
    787, 804, 810, 821, 824, 842,1051,1056,1061,1071,1077,1084,1085,1087,1092,
   1106,1117,1133,1139,1142,1148,1284,1301,1307,1406
] 

MAX_RETRY = 1  

def start_ollama_in_terminals():
    subprocess.Popen([
        "gnome-terminal", "--", "bash", "-c",
        "ollama serve; exec bash"
    ])
    time.sleep(7)
    subprocess.Popen([
        "gnome-terminal", "--", "bash", "-c",
        "ollama run qwen3:32b; exec bash"
    ])
    time.sleep(15)
def stop_ollama():
    os.system("pkill -f 'ollama run qwen3:32b'")
    os.system("pkill -f 'ollama serve'")
    time.sleep(7)

def run_one_episode(ep_id):
    attempt = 1
    while attempt <= MAX_RETRY:
        print(f"\n Episode {ep_id} for the {attempt} time")

        start_ollama_in_terminals()

        cmd = [
            "python", "run.py",
            "--exp_name", f"cont-cwp-opennav-ori-{ep_id}",
            "--exp-config", "run_ReThinkNav.yaml",
            "--llm", "qwen3:32b",
            "--api_key", "not-needed",
            "--target_episode_ids", str(ep_id),
            "SIMULATOR_GPU_IDS", "[0]",
            "TORCH_GPU_ID", "0",
            "TORCH_GPU_IDS", "[0]",
            "EVAL.SPLIT", "val_unseen"
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        result = subprocess.run(cmd, env=env)

        stop_ollama()

        if result.returncode == 0:
            print(f"Episode {ep_id} is done.")
            return
        else:
            attempt += 1
            time.sleep(5)

if __name__ == "__main__":
    for ep_id in episode_ids:
        run_one_episode(ep_id)
        time.sleep(7)
