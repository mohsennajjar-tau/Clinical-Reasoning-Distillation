import subprocess, time
while True:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        parts = r.stdout.strip().split(", ")
        used, total, temp = float(parts[0]), float(parts[1]), float(parts[2])
        pct = used / total * 100
        status = "✅" if pct < 80 else "⚠️" if pct < 90 else "🔴 DANGER"
        print(f"{status} VRAM: {used:.0f}/{total:.0f} MB ({pct:.0f}%) | Temp: {temp}°C")
    except:
        print("nvidia-smi failed")
    time.sleep(5)

    