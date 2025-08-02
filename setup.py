import subprocess
import sys
import os

def main():
    env_name = "cleanwaterai_env"
    
    print("🔃 Processing...")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'venv', env_name
        ], check=True, capture_output=True)
        
        if os.name == 'nt':
            pip_path = os.path.join(env_name, 'Scripts', 'pip.exe')
        else:
            pip_path = os.path.join(env_name, 'bin', 'pip')
        print(f"✅ Using pip at: {pip_path}")
        print("🔃 Installing requirements...")
        subprocess.run([
            pip_path, 'install', '-r', 'requirements.txt'
        ], check=True, capture_output=True)
        print("✅ Requirements installed successfully.")
        print("🔃 Installing the package in editable mode...")
        subprocess.run([
            pip_path, 'install', '-e', '.'
        ], check=True, capture_output=True)
        print("✅ Project Setup Completed Successfully. Run the application using ```python main.py```.")
        
    except subprocess.CalledProcessError:
        sys.exit(1)

if __name__ == "__main__":
    main()