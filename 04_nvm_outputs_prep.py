import os
from glob import glob
import subprocess
import argparse
import tarfile
import shutil

def convert_cif_to_pdb_execute(input_dir, output_dir):
    cif_files = glob(os.path.join(input_dir, '**', '*.cif'), recursive=True)
    # Filter out macOS AppleDouble metadata stubs
    cif_files = [f for f in cif_files if not os.path.basename(f).startswith('._')]
    if not cif_files:
        print("Brak plików .cif w podanym katalogu.")
        return
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for cif_file in cif_files:
        # Determine output path, preserving directory structure
        rel_path = os.path.relpath(cif_file, input_dir)
        pdb_rel = os.path.splitext(rel_path)[0] + '.pdb'
        output_file = os.path.join(output_dir, pdb_rel)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cmd = ['obabel', '-icif', cif_file, '-opdb', '-O', output_file]
        try:
            subprocess.run(cmd, check=True)
            print(f'Konwersja {cif_file} do {output_file} zakończona sukcesem.')
        except subprocess.CalledProcessError as e:
            print(f'Błąd konwersji {cif_file}: {e}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract a .tar.gz archive and convert contained .cif files to .pdb')
    parser.add_argument('archive', help='Path to the .tar.gz archive')
    args = parser.parse_args()

    tar_path = args.archive
    tar_dir = os.path.dirname(tar_path) or '.'
    base_name = os.path.basename(tar_path)
    if base_name.endswith('.tar.gz'):
        dir_name = base_name[:-7]
    else:
        dir_name = os.path.splitext(base_name)[0]

    # Extract the archive
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=tar_dir)

    # Remove macOS AppleDouble metadata files
    for root, dirs, files in os.walk(os.path.join(tar_dir, dir_name), topdown=False):
        for name in files:
            if name.startswith('._'):
                os.remove(os.path.join(root, name))
        for name in dirs:
            if name.startswith('._'):
                dirpath = os.path.join(root, name)
                try:
                    os.rmdir(dirpath)
                except OSError:
                    shutil.rmtree(dirpath)

    # Directory with extracted files
    extracted_dir = os.path.join(tar_dir, dir_name)

    # Rename any .dcd trajectory files to .xtc
    dcd_files = glob(os.path.join(extracted_dir, '**', '*.dcd'), recursive=True)
    if dcd_files:
        for dcd_file in dcd_files:
            xtc_file = os.path.splitext(dcd_file)[0] + '.xtc'
            os.rename(dcd_file, xtc_file)
            print(f'Renamed {dcd_file} to {xtc_file}')

    # Convert CIF to PDB in extracted directory
    convert_cif_to_pdb_execute(extracted_dir, extracted_dir)
