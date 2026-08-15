import redivis
import os

def download_merlin(scan_ids: list[str], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    table = redivis.table("aimi.merlin_abdominal_ct_dataset:exfv:v1_0.merlin_data:m1t0")

    downloaded_paths = []

    for scan_id in scan_ids:
        filename = f"{scan_id}.nii.gz"
        save_path = os.path.join(output_dir, filename)

        if os.path.exists(save_path):
            downloaded_paths.append(save_path)
            continue

        # with file.open() as f:
        #     """ Process file """

        # Any fsspec compatible code can also use the redivis URI scheme:
        # with fsspec.open(f"redivis://aimi.merlin_abdominal_ct_dataset:exfv:v1_0.merlin_data:m1t0/{scan_id}.nii.gz") as f:
        # 	""" Process file """

        file_ref = table.file(filename)
        file_ref.download(save_path)
        downloaded_paths.append(save_path)

    return downloaded_paths