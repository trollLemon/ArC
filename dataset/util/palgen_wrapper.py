import subprocess
import numpy as np
import logging


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


class PalGenWrapper:
    def __init__(self, num_colors:int, num_jobs:int, logger: logging.Logger):
        self.num_colors = num_colors
        self.num_jobs = num_jobs
        self.logger = logger


    def run(self, image_path):
        self.logger.info(f"Running palgen on {image_path}")
        try:
            result = subprocess.run(['palgen', '-i', image_path, '-c', str(self.num_colors), '-j', str(self.num_jobs)], capture_output=True)
            self.logger.info(f"palgen returned {result.returncode}")
            self.logger.info(f"creating color array for {image_path}")
            palette = result.stdout.decode('utf-8').split('\n')
            palette_np = np.zeros((self.num_colors, 3))
            for i,color in enumerate(palette):
                if not color:
                    continue
                rgb = hex_to_rgb(color)

                rgb_np = np.array(rgb)

                rgb_np += 1 # add 1 since 0 is considered a NAN class, so 1,1,1 will be pitch black for example
                palette_np[i, :] = rgb_np
            self.logger.info(f"palgen created color array for {image_path}")
            return palette_np
        except FileNotFoundError as e:
            self.logger.error(f"Executable not found: {e}")
            return None
        except subprocess.TimeoutExpired as e:
            self.logger.error(f"Process timed out: {e}")
            return None
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Process failed (exit code {e.returncode}):\nSTDERR: {e.stderr}")
            return None


