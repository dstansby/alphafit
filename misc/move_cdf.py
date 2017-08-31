import os
import sys
sys.path.append('./fitting')
from config import get_dirs

output_dir = get_dirs()


for probe in ['1', '2']:
    for year in range(1974, 1986):
        csv_dir = os.path.join(output_dir, 'csv', 'helios' + probe, str(year))
        cdf_dir = os.path.join(output_dir, 'cdf', 'helios' + probe, str(year))
        if not os.path.isdir(cdf_dir):
            os.makedirs(cdf_dir)
        for dirpath, dirs, files in os.walk(csv_dir):
            for fname in files:
                if fname.endswith(".cdf"):
                    old_file = os.path.join(csv_dir, fname)
                    new_file = os.path.join(cdf_dir, fname)
                    os.rename(old_file, new_file)
                    print(new_file)
