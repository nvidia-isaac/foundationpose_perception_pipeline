#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the commercial depth path on one scene.

    ./.venv/bin/python -m foundationpose_perception_pipeline.inference.stereo \\
        --scene-dir <dataset_root>/<dataset>/test/000000 \\
        --out-dir /tmp/fs --engine <path>.engine

Runs the same backend in isolation for one-scene debugging.

A separate module from `depth.py` so that `python -m ...stereo` does not re-execute a module the
package `__init__` has already imported, which makes runpy warn about unpredictable behaviour.
"""

from foundationpose_perception_pipeline.inference.stereo.depth import main

if __name__ == "__main__":
    main()
