# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
from pathlib import Path
from tqdm import tqdm


def download(url: str, filename: Path, chunk_size=1024):
    """
    Downloads a file from a url and saves it to a path.

    Displays a progress bar.
    """
    filename = Path(filename)  # Ensure filename is a Path object

    # Check if the file already exists
    if filename.exists():
        print(f"File already exists: {filename}")
        return

    else:  # Download the file

        # Create parent directory if it doesn't exist
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Download the header to get total size
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        print(f"Downloading {filename.name} from {url}")

        with (
            open(filename, "wb") as file,
            tqdm(
                desc=filename.name,
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):  # Uses a tqdm context manager to display a progress bar
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

        print(f"Download complete: {filename}")
